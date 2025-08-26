from typing import Any, Dict, List, Tuple, Optional
from transformers import AutoConfig
import logging
import time
from estimator_utils import *
from hardware_specs import INSTANCE_SPEC
from cluster_pool import ClusterPool
import sys
import os
# Add parent directory to path for protocols import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protocols import OUT_OF_MEMORY

# Configure logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self):
        self.stages = [] # 여기에는 instance 이름들이 들어감.
        self.azs = [] # 여기에는 instance 들의 az 가 들어갈 것이다. 현재는 전부 dummy az 를 사용할 것
        self.layer_per_stage = [] # 각 stage 별로 몇개의 layer가 있는지.
        self.throughput = 0 # 전체 파이프라인의 throughput
        self.cost = 0 # 전체 파이프라인의 비용
        self.global_batch_size = 0
        self.num_cache_blocks = 0 # 캐시 블록의 개수
        self.latency_per_global_batch = 0 # 파이프라인의 추론 최소 지연 시간
    
    def __repr__(self):
        """Pipeline 객체의 문자열 표현"""
        if not self.stages:
            return "Pipeline(empty)"
        
        # throughput/cost ratio 계산 (0으로 나누기 방지)
        ratio = self.throughput / self.cost if self.cost > 0 else 0
        
        # 각 스테이지 정보를 간단히 표현
        stage_info = []
        for instance, layers in zip(self.stages, self.layer_per_stage):
            stage_info.append(f"{instance}:{layers}L")
        
        return (f"Pipeline(stages=[{', '.join(stage_info)}], "
                f"throughput={self.throughput:.3f}, "
                f"cost=${self.cost:.3f}, "
                f"ratio={ratio:.3f}, "
                f"latency_per_global_batch={self.latency_per_global_batch:.0f}ms)")
    
    def set_cost(self, cost: float):
        """파이프라인의 총 비용을 설정합니다 (hourly cost)."""
        self.cost = cost
    
    def calculate_throughput(self, config: Dict[str, Any]) -> float:
        """파이프라인의 throughput을 계산하고 self.throughput에 저장합니다."""
        node_layer_comb = []
        for i, (instance, layer_count) in enumerate(zip(self.stages, self.layer_per_stage)):
            node_layer_comb.append((instance, self.azs[i], layer_count))
        
        throughput, total_latency_per_global_batch = get_throughput(
            avg_input_len=config["expected_input_len"],
            avg_output_len=config["expected_output_len"],
            max_model_len=config["max_model_len"],
            hidden_dim=config["hidden_size"],
            num_attention_head=config["num_attention_heads"],
            num_kv_cache_head=config["num_key_value_heads"],
            total_num_layers=config["num_layers"],
            vocab_size=config["vocab_size"],
            intermediate_dim=config["intermediate_size"],
            gpu_mem_utilization=config["gpu_mem_utilization"],
            node_layer_comb=node_layer_comb,
            dtype=config["dtype"]
        )
        
        self.throughput = throughput
        self.latency_per_global_batch = total_latency_per_global_batch
        return self.throughput


class DPOptimizer:
    """Dynamic Programming based optimizer for throughput/cost optimization"""
    
    def __init__(self, config: Dict[str, Any], budget: float, latency_slo: float, cluster_pool: ClusterPool, max_stages: Optional[int] = None):
        """
        Args:
            config: 모델 및 시스템 설정
            budget: 예산 상한 (hourly cost in dollars)
            latency_slo: 지연시간 SLO (milliseconds)
            cluster: 클러스터 리소스 관리자 (필수)
            max_stages: 파이프라인의 최대 스테이지 수 제한 (None이면 제한 없음)
        """
        self.config = config
        self.budget = budget
        self.latency_slo = latency_slo
        self.num_layers = config["num_layers"]
        self.instance_types = list(INSTANCE_SPEC.keys())
        self.cluster_pool = cluster_pool
        
        # 최대 스테이지 수 설정 (기본값: num_layers, 즉 제한 없음)
        # 실용적으로는 더 작은 값으로 제한하여 파이프라인 오버헤드를 줄임
        self.max_stages = max_stages if max_stages is not None else self.num_layers
        
        # DP 캐시: dp[layer_idx][num_stages] = Pipeline
        # 최대 stage 수를 num_layers로 설정 (최악의 경우 각 layer가 하나의 stage)
        self.dp = [[None for _ in range(self.num_layers + 1)] for _ in range(self.num_layers + 1)]
    
    def _check_budget_constraint(self, pipeline: Pipeline) -> bool:
        """파이프라인이 예산 제약을 만족하는지 확인"""
        return pipeline.cost <= self.budget
    
    def _create_pipeline(self, stages: List[Tuple[str, int]]) -> Pipeline:
        """스테이지 정보로부터 Pipeline 객체 생성"""
        pipeline = Pipeline()
        if len(stages) <= 0: # stages 가 비어있다면 (초기화시 가능함)
            return pipeline
        
        for instance, layer_count in stages:
            pipeline.stages.append(instance)
            pipeline.azs.append("dummy-az")
            pipeline.layer_per_stage.append(layer_count)
        
        # 비용 계산 - ClusterPool의 가격 사용
        total_cost = 0
        for instance in pipeline.stages:
            total_cost += self.cluster_pool.get_instance_price(instance)
        pipeline.set_cost(total_cost)
        
        # Throughput 및 latency 계산
        pipeline.calculate_throughput(self.config)
        
        return pipeline
    
    def optimize(self) -> Optional[Pipeline]:
        """
        Dynamic Programming을 사용하여 throughput/cost를 최대화하는 최적의 파이프라인을 찾습니다.
        
        목적:
        - 주어진 예산(budget)과 지연시간 SLO 제약 하에서
        - throughput/cost 비율을 최대화하는 파이프라인 구성을 찾습니다
        
        알고리즘:
        1. DP 상태: dp[i][j] = i개의 레이어를 j개의 스테이지로 처리하는 최적 파이프라인
        2. 전이: 새로운 인스턴스를 추가하여 더 많은 레이어를 처리
        3. 제약조건 체크: 예산, 지연시간, 메모리 제약을 만족하는지 확인
        
        Returns:
            최적의 Pipeline 객체 또는 실행 가능한 파이프라인이 없으면 None
        """
        
        # Base case: 0개 레이어, 0개 스테이지로 시작
        self.dp[0][0] = self._create_pipeline([])
        
        # DP 수행: 1개부터 전체 레이어 수까지 순차적으로 최적해를 구함
        for current_layer in range(1, self.num_layers + 1):
            # 현재 스테이지에서 사용할 수 있는 모든 인스턴스 타입 탐색
            for instance in self.instance_types:
                # 이 인스턴스가 처리할 레이어 수를 1부터 current_layer까지 시도
                # (한 인스턴스가 모든 레이어를 처리할 수도 있음)
                for layers_to_process in range(1, current_layer + 1):
                    # 이전 상태의 레이어 수 계산
                    prev_layer = current_layer - layers_to_process
                    
                    # 이전 상태들을 확인 (최대 스테이지 수 제한 적용)
                    # prev_num_stages가 max_stages-1이면, 새 스테이지 추가 시 max_stages가 됨
                    for prev_num_stages in range(self.max_stages):
                        prev_pipeline = self.dp[prev_layer][prev_num_stages]
                        if prev_pipeline is None:
                            continue
                        
                        # 새로운 스테이지 수 계산
                        new_num_stages = prev_num_stages + 1
                        
                        # 마지막 스테이지인 경우, 남은 모든 레이어를 처리해야 함
                        if new_num_stages == self.max_stages:
                            if layers_to_process != (self.num_layers - prev_layer):
                                continue
                        
                        # 새로운 스테이지 구성 생성
                        if prev_num_stages == 0:  # 첫 번째 스테이지인 경우
                            new_stages = [(instance, layers_to_process)]
                        else:
                            # 기존 파이프라인에 새로운 스테이지 추가
                            new_stages = []
                            for i in range(len(prev_pipeline.stages)):
                                new_stages.append((prev_pipeline.stages[i], prev_pipeline.layer_per_stage[i]))
                            new_stages.append((instance, layers_to_process))
                        
                        # 클러스터 리소스 제약 확인 (cluster가 설정된 경우)
                        if self.cluster_pool and not self.cluster_pool.check_feasibility(new_stages):
                            continue
                        
                        # 새로운 파이프라인 생성 및 평가
                        new_pipeline = self._create_pipeline(new_stages)
                        
                        # 제약 조건 확인
                        # 메모리 제약: OUT_OF_MEMORY 상태인 경우 스킵
                        if new_pipeline.throughput == OUT_OF_MEMORY:
                            continue

                        # 유효성 검사: throughput과 cost가 양수여야 함
                        if new_pipeline.throughput <= 0 or new_pipeline.cost <= 0:
                            continue

                        # 예산 제약: 총 비용이 budget을 초과하면 스킵
                        if not self._check_budget_constraint(new_pipeline):
                            continue
                        
                        # 지연시간 제약: 최소 지연시간이 SLO를 초과하면 스킵
                        if new_pipeline.latency_per_global_batch > self.latency_slo:
                            continue
                        
                        # 새로운 스테이지 수
                        new_num_stages = len(new_stages)
                        
                        # 현재 상태(current_layer, new_num_stages)에서의 최적해 업데이트
                        # throughput/cost 비율이 더 높은 파이프라인으로 교체
                        current_best = self.dp[current_layer][new_num_stages]
                        if current_best is None or (new_pipeline.throughput / new_pipeline.cost) > (current_best.throughput / current_best.cost):
                            self.dp[current_layer][new_num_stages] = new_pipeline
        
        # 최종 결과 선택: 모든 레이어를 처리하는 파이프라인 중에서
        # throughput/cost 비율이 가장 높은 것을 선택
        best_pipeline = None
        best_ratio = 0
        
        # 모든 가능한 스테이지 수에 대해 확인
        for num_stages in range(1, self.num_layers + 1):
            pipeline = self.dp[self.num_layers][num_stages]
            if pipeline and pipeline.throughput > 0 and pipeline.cost > 0:
                ratio = pipeline.throughput / pipeline.cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pipeline = pipeline
        
        return best_pipeline
    
    def print_dp_table(self, layer_range: Tuple[int, int] = None, stage_range: Tuple[int, int] = None):
        """
        디버깅용: DP 테이블의 내용을 출력합니다
        
        Args:
            layer_range: (start, end) 레이어 범위. None이면 전체 범위
            stage_range: (start, end) 스테이지 범위. None이면 전체 범위
        """
        # 기본값 설정
        if layer_range is None:
            layer_start, layer_end = 0, self.num_layers
        else:
            layer_start, layer_end = layer_range
            layer_start = max(0, layer_start)
            layer_end = min(self.num_layers, layer_end)
        
        if stage_range is None:
            stage_start, stage_end = 0, self.num_layers
        else:
            stage_start, stage_end = stage_range
            stage_start = max(0, stage_start)
            stage_end = min(self.num_layers, stage_end)
        
        logger.info(f"=== DP Table (layers {layer_start}-{layer_end}, stages {stage_start}-{stage_end}) ===")
        
        # 헤더 출력
        header = f"{'Layer':>6} |"
        for stage in range(stage_start, stage_end + 1):
            header += f" Stage {stage:<8} |"
        logger.info(header)
        logger.info("-" * len(header))
        
        # 데이터 출력
        for layer in range(layer_start, layer_end + 1):
            row = f"{layer:>6} |"
            for stage in range(stage_start, stage_end + 1):
                if stage < len(self.dp[layer]):
                    pipeline = self.dp[layer][stage]
                    if pipeline is None:
                        row += f"{'None':^15} |"
                    else:
                        # 간단한 요약 정보: cost(throughput)
                        summary = f"${pipeline.cost:.2f}({pipeline.throughput:.2f})"
                        row += f"{summary:^15} |"
                else:
                    row += f"{'N/A':^15} |"
            logger.info(row)
    
    def get_all_valid_pipelines(self) -> List[Pipeline]:
        """디버깅용: DP 테이블에서 유효한 모든 파이프라인을 반환합니다"""
        valid_pipelines = []
        
        for layer in range(len(self.dp)):
            for stage in range(len(self.dp[layer])):
                pipeline = self.dp[layer][stage]
                if pipeline and pipeline.throughput > 0 and pipeline.cost > 0:
                    valid_pipelines.append(pipeline)
        
        return valid_pipelines
    
    def get_ranked_pipelines(self, max_rank: int = 10, only_complete: bool = True) -> List[Tuple[int, Pipeline, float]]:
        """
        디버깅용: throughput/cost 비율에 따라 파이프라인을 랭킹으로 반환합니다
        
        Args:
            max_rank: 반환할 최대 랭킹 수
            only_complete: True면 모든 레이어를 처리하는 파이프라인만, False면 모든 유효한 파이프라인
        
        Returns:
            List of (rank, pipeline, throughput/cost ratio)
        """
        if only_complete:
            # 모든 레이어를 처리하는 파이프라인만 가져오기
            complete_pipelines = []
            if self.num_layers in range(len(self.dp)):
                for stage in range(len(self.dp[self.num_layers])):
                    pipeline = self.dp[self.num_layers][stage]
                    if pipeline and pipeline.throughput > 0 and pipeline.cost > 0:
                        complete_pipelines.append(pipeline)
            pipelines_to_rank = complete_pipelines
        else:
            # 모든 유효한 파이프라인 가져오기
            pipelines_to_rank = self.get_all_valid_pipelines()
        
        # throughput/cost 비율로 정렬
        pipeline_ratios = []
        for pipeline in pipelines_to_rank:
            ratio = pipeline.throughput / pipeline.cost if pipeline.cost > 0 else 0
            pipeline_ratios.append((pipeline, ratio))
        
        # 비율 내림차순으로 정렬
        pipeline_ratios.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 max_rank개만 반환
        ranked_pipelines = []
        for rank, (pipeline, ratio) in enumerate(pipeline_ratios[:max_rank], 1):
            ranked_pipelines.append((rank, pipeline, ratio))
        
        return ranked_pipelines
    
    def print_ranked_pipelines(self, max_rank: int = 10, only_complete: bool = True):
        """
        디버깅용: 랭킹된 파이프라인들을 출력합니다
        
        Args:
            max_rank: 출력할 최대 랭킹 수
            only_complete: True면 모든 레이어를 처리하는 파이프라인만, False면 모든 유효한 파이프라인
        """
        ranked_pipelines = self.get_ranked_pipelines(max_rank, only_complete)
        
        if not ranked_pipelines:
            logger.info("No valid pipelines found for ranking")
            return
        
        scope = "complete pipelines" if only_complete else "all valid pipelines"
        logger.info(f"=== Top {len(ranked_pipelines)} {scope} (by throughput/cost ratio) ===")
        
        for rank, pipeline, ratio in ranked_pipelines:
            total_layers = sum(pipeline.layer_per_stage) if pipeline.layer_per_stage else 0
            logger.info(f"Rank {rank:2d}: {pipeline} (ratio: {ratio:.3f}, layers: {total_layers})")
        
        logger.info("")


def run_test_case(
        config: Dict, 
        budget: float, 
        latency_slo: float, 
        cluster_pool: ClusterPool,
        look_rank: int = 5,
        max_stages: Optional[int] = None) -> Tuple[Pipeline, DPOptimizer, float]:
    """
    Run a single test case with given budget and SLO.
    
    Args:
        config: Model configuration dictionary
        budget: Budget in USD per hour
        latency_slo: Latency SLO in milliseconds
        look_rank: Number of top pipelines to show
        max_stages: Maximum number of stages allowed (None for no limit)
    
    Returns:
        Tuple of (Optimal Pipeline, DPOptimizer, optimization_time)
    """
    optimizer = DPOptimizer(config, budget=budget, latency_slo=latency_slo, cluster_pool=cluster_pool, max_stages=max_stages)
    
    start_time = time.time()
    result = optimizer.optimize()
    optimization_time = time.time() - start_time
    
    if result:
        logger.info(f"✓ Found optimal pipeline:")
        logger.info(f"  {result}")
        logger.info(f"  - Stages: {result.stages}")
        logger.info(f"  - Layers per stage: {result.layer_per_stage}")
        logger.info(f"  - Total layers: {sum(result.layer_per_stage)}")
        
        # Show top ranked pipelines
        optimizer.print_ranked_pipelines(max_rank=look_rank, only_complete=True)
    else:
        logger.info("✗ No feasible pipeline found within constraints")
    
    # Show DP table for debugging
    optimizer.print_dp_table(layer_range=(config['num_layers']-4, config['num_layers']), stage_range=(1, 4))
    logger.info(f"⏱️  Optimization time: {optimization_time:.3f} seconds")
    logger.info("")
    
    return result, optimizer, optimization_time


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_config = AutoConfig.from_pretrained(model_name)

    look_rank = 5
    max_stages = 10

    config = {
        "expected_input_len": 512,  # 입력 시퀀스 길이
        "expected_output_len": 128,  # 출력 시퀀스 길이
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_hidden_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
        "intermediate_size": model_config.intermediate_size,
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "dtype": torch.float16,
        "max_model_len": 4096,
        "gpu_mem_utilization": 0.9
    }

    available_spot_nodes = {
        "(spot)g4dn.xlarge":      100,
        "(spot)g4dn.12xlarge":    100,
        "(spot)g4dn.metal":       100,
        "(spot)g5.xlarge":        30,
        "(spot)g5.12xlarge":      100,
        "(spot)g5.48xlarge":      100,
        "(spot)g6.xlarge":        40,
        "(spot)g6.12xlarge":      100,
        "(spot)g6.48xlarge":      100,
        "(spot)g6e.xlarge":       35,
        "(spot)g6e.12xlarge":     100,
        "(spot)g6e.48xlarge":     100,
        "(spot)p4d.24xlarge":     0,
        "(spot)p4de.24xlarge":    0,
        "(spot)p5.4xlarge":       0,
        "(spot)p5.48xlarge":      0,
        "(spot)p5e.48xlarge":     0,
        "(spot)p5en.48xlarge":    0,
        "(spot)p6-b200.48xlarge": 0,
    }
    spot_prices = {
        "(spot)p5.4xlarge":       9999, # spot 없음.
        "(spot)p5e.48xlarge":     22,
    }
    cluster_pool = ClusterPool(available_spot_nodes=available_spot_nodes, spot_prices=spot_prices)


    logger.info("=" * 80)
    logger.info("            Dynamic Programming Optimizer Test")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info("")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)
    logger.info("")

    # 시나리오 1, 2, 3 : 같은 예산에서 다른 SLO
    # 시나리오 4, 5, 6 : 같은 SLO 에서 다른 예산
    # 시나리오 7 : 예산 무제한, SLO 없음

    # 테스트 시나리오 1: 
    logger.info("-" * 80)
    logger.info("Test Case 1: Budget ($60/hour), Latency (20s)")
    logger.info("-" * 80)
    result1, optimizer1, optimization_time1 = run_test_case(config, budget=60.0, latency_slo=20000, look_rank=look_rank, cluster_pool=cluster_pool, max_stages=max_stages)

    # 테스트 시나리오 2: 
    logger.info("-" * 80)
    logger.info("Test Case 2: Budget ($60/hour), Latency (10s)")
    logger.info("-" * 80)
    result2, optimizer2, optimization_time2 = run_test_case(config, budget=60.0, latency_slo=10000, look_rank=look_rank, cluster_pool=cluster_pool, max_stages=max_stages)

    # 테스트 시나리오 3: 
    logger.info("-" * 80)
    logger.info("Test Case 3: Budget ($60/hour), Latency (5s)")
    logger.info("-" * 80)
    result3, optimizer3, optimization_time3 = run_test_case(config, budget=60.0, latency_slo=5000, look_rank=look_rank, cluster_pool=cluster_pool, max_stages=max_stages)

    # 테스트 시나리오 4: 
    logger.info("-" * 80)
    logger.info("Test Case 4: Budget ($60/hour), Latency (30s)")
    logger.info("-" * 80)
    result4, optimizer4, optimization_time4 = run_test_case(config, budget=60.0, latency_slo=30000, look_rank=look_rank, cluster_pool=cluster_pool, max_stages=max_stages)

    # 테스트 시나리오 5: 
    logger.info("-" * 80)
    logger.info("Test Case 5: Budget ($30/hour), Latency (30s)")
    logger.info("-" * 80)
    result5, optimizer5, optimization_time5 = run_test_case(config, budget=30.0, latency_slo=30000, look_rank=look_rank, cluster_pool=cluster_pool, max_stages=max_stages)

    # 테스트 시나리오 6: 
    logger.info("-" * 80)
    logger.info("Test Case 6: Budget ($10/hour), Latency (30s)")
    logger.info("-" * 80)
    result6, optimizer6, optimization_time6 = run_test_case(config, budget=10.0, latency_slo=30000, look_rank=look_rank, cluster_pool=cluster_pool, max_stages=max_stages)

    # 테스트 시나리오 7: 예산 무한, 지연시간 무제한
    logger.info("-" * 80)
    logger.info("Test Case 7: Unlimited budget, unlimited latency")
    logger.info("-" * 80)
    result7, optimizer7, optimization_time7 = run_test_case(config, budget=9999, latency_slo=9999999999, look_rank=look_rank, cluster_pool=cluster_pool, max_stages=max_stages)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("            Test Summary")
    logger.info("=" * 80)
    
    # 결과 요약
    results = [
        ("Test 1", result1, optimization_time1),
        ("Test 2", result2, optimization_time2),
        ("Test 3", result3, optimization_time3),
        ("Test 4", result4, optimization_time4),
        ("Test 5", result5, optimization_time5),
        ("Test 6", result6, optimization_time6)
    ]
    
    for test_name, result, opt_time in results:
        if result:
            logger.info(f"{test_name}: SUCCESS - {result} (⏱️ {opt_time:.3f}s)")
        else:
            logger.info(f"{test_name}: FAILED - No feasible solution (⏱️ {opt_time:.3f}s)")
    
    # 최적화 시간 통계
    logger.info("")
    optimization_times = [optimization_time1, optimization_time2, optimization_time3, optimization_time4, optimization_time5, optimization_time6]
    total_optimization_time = sum(optimization_times)
    avg_optimization_time = total_optimization_time / 6
    logger.info(f"Optimization time statistics:")
    logger.info(f"  Total optimization time: {total_optimization_time:.3f} seconds")
    logger.info(f"  Average optimization time: {avg_optimization_time:.3f} seconds")
    logger.info(f"  Fastest optimization: {min(optimization_times):.3f} seconds")
    logger.info(f"  Slowest optimization: {max(optimization_times):.3f} seconds")
