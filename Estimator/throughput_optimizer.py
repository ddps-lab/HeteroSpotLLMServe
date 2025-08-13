from typing import Any, Dict, List, Tuple, Optional
from transformers import AutoConfig
import logging
import time
from estimator_utils import *
from hardware_specs import INSTANCE_SPEC
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
        self.lowest_inference_latency = 0 # 파이프라인의 추론 최소 지연 시간
    
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
                f"latency={self.lowest_inference_latency:.0f}ms)")
    
    def calculate_cost(self) -> float:
        """파이프라인의 총 비용을 계산하고 self.cost에 저장합니다 (hourly cost)."""
        total_cost = 0
        for instance in self.stages:
            if instance in INSTANCE_SPEC:
                total_cost += INSTANCE_SPEC[instance]["ondemand_price"]
        self.cost = total_cost
        return self.cost
    
    def calculate_throughput(self, config: Dict[str, Any]) -> float:
        """파이프라인의 throughput을 계산하고 self.throughput에 저장합니다."""
        node_layer_comb = []
        for i, (instance, layer_count) in enumerate(zip(self.stages, self.layer_per_stage)):
            node_layer_comb.append((instance, self.azs[i], layer_count))
        
        throughput = get_throughput(
            avg_input_len=config["expected_input_len"],
            avg_output_len=config["expected_output_len"],
            max_model_len=config["max_model_len"],
            hidden_dim=config["hidden_size"],
            num_attention_head=config["num_attention_heads"],
            num_kv_cache_head=config["num_key_value_heads"],
            total_layer_num=config["num_layers"],
            total_model_mem=config["total_model_mem"],
            gpu_mem_utilization=config["gpu_mem_utilization"],
            node_layer_comb=node_layer_comb,
            dtype=config["dtype"]
        )
        
        self.throughput = throughput
        return self.throughput

# 아래 인자는 Latency 계산을 미리 캐싱해두는 역할을 맡을 것이며, 나중에 Pipeline 의 latency 하한을 맡을 것
instance_computation_latency_per_layer_cache = {} # instance 별로 캐시된 computation latency (Batch size 1 기준)

def initialize_computation_latencies(config: Dict[str, Any]):
    """
    파이프라인의 각 stage 별로 computation latency를 초기화합니다.
    """
    # config에서 자주 사용되는 값들을 미리 추출
    expected_input_len = config["expected_input_len"]
    expected_output_len = config["expected_output_len"]
    hidden_size = config["hidden_size"]
    dtype = config["dtype"]
    
    for instance in INSTANCE_SPEC.keys():
        gpu_type = INSTANCE_SPEC[instance]["gpu_type"]
        gpu_count = INSTANCE_SPEC[instance]["gpu_count"]

        prefill_computation_latency_per_layer = get_prefill_computation_latency_per_layer(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            input_len=expected_input_len,
            hidden_dim=hidden_size,
            batch_size=1,
            intermediate_dim=None,
            dtype=dtype,
        )

        decode_computation_latency_per_layer = get_decoding_computation_latency_per_layer(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            input_len=expected_input_len,
            output_len=expected_output_len,
            hidden_dim=hidden_size,
            batch_size=1,
            intermediate_dim=None,
            dtype=dtype,
        )

        instance_computation_latency_per_layer_cache[instance] = {
            "prefill": prefill_computation_latency_per_layer,
            "decode": decode_computation_latency_per_layer
        }

def get_minimum_latency(pipeline: Pipeline, config: Dict[str, Any]):
    """
    파이프라인의 최소 지연 시간을 계산합니다.
    """
    # config에서 자주 사용되는 값들을 미리 추출
    expected_input_len = config["expected_input_len"]
    expected_output_len = config["expected_output_len"]
    hidden_size = config["hidden_size"]
    dtype = config["dtype"]
    
    total_prefill_computation_latency = 0
    total_decode_computation_latency = 0
    total_pp_communication_latency = 0
    total_tp_communication_latency = 0
    
    for stage_idx in range(len(pipeline.stages)):
        instance = pipeline.stages[stage_idx]
        layer_count = pipeline.layer_per_stage[stage_idx]
        gpu_count = INSTANCE_SPEC[instance]["gpu_count"]
        p2p_bandwidth = INTERCONNECT_SPEC[INSTANCE_SPEC[instance]["interconnect"]]["bandwidth"]

        prefill_computation_latency_per_layer = instance_computation_latency_per_layer_cache[instance]["prefill"]
        decode_computation_latency_per_layer = instance_computation_latency_per_layer_cache[instance]["decode"]

        if stage_idx != len(pipeline.stages) - 1:
            # 마지막 stage 가 아니라면 다음 stage 에게 send 해야함
            prefill_pp_communication_send_latency = get_pp_communication_latency_send_recv(
                batch_size=1,
                sequence_len=expected_input_len,
                hidden_dim=hidden_size,
                inter_node_latency_ms=None,  # intra region latency
                inter_node_bandwidth=None,  # intra region bandwidth
                dtype=dtype
            )
            decode_pp_communication_send_latency = get_pp_communication_latency_send_recv(
                batch_size=1,
                sequence_len=1,
                hidden_dim=hidden_size,
                inter_node_latency_ms=None,  # intra region latency
                inter_node_bandwidth=None,  # intra region bandwidth
                dtype=dtype
            ) * expected_output_len
        else:
            # 마지막 stage 는 다음 stage 가 없으므로 통신 지연이 없음
            prefill_pp_communication_send_latency = 0
            decode_pp_communication_send_latency = 0

        if stage_idx != 0 and gpu_count > 1:  # 첫 번째 Stage 가 아니고 tp size 가 1보다 크면 broadcast 를 해야함.
            prefill_pp_communication_broadcast_latency = get_pp_communication_latency_broadcast(
                batch_size=1,
                sequence_len=expected_input_len,
                hidden_dim=hidden_size,
                tp_size=gpu_count,
                p2p_bandwidth=p2p_bandwidth,
                p2p_latency_ms=None,
                dtype=dtype
            )
            decode_pp_communication_broadcast_latency = get_pp_communication_latency_broadcast(
                batch_size=1,
                sequence_len=1,
                hidden_dim=hidden_size,
                tp_size=gpu_count,
                p2p_bandwidth=p2p_bandwidth,
                p2p_latency_ms=None,
                dtype=dtype
            ) * expected_output_len
        else:
            prefill_pp_communication_broadcast_latency = 0
            decode_pp_communication_broadcast_latency = 0
        
        prefill_tp_communication_latency = get_tp_communication_latency_per_layer(
            tp_size=gpu_count,
            batch_size=1,
            sequence_len=expected_input_len,
            hidden_dim=hidden_size,
            p2p_bandwidth=p2p_bandwidth,
            dtype=dtype
        ) * layer_count
        decode_tp_communication_latency = get_tp_communication_latency_per_layer(
            tp_size=gpu_count,
            batch_size=1,
            sequence_len=1,
            hidden_dim=hidden_size,
            p2p_bandwidth=p2p_bandwidth,
            dtype=dtype
        ) * layer_count * expected_output_len

        total_prefill_computation_latency += prefill_computation_latency_per_layer * layer_count
        total_decode_computation_latency += decode_computation_latency_per_layer * layer_count
        total_pp_communication_latency += (prefill_pp_communication_send_latency + prefill_pp_communication_broadcast_latency)
        total_pp_communication_latency += (decode_pp_communication_send_latency + decode_pp_communication_broadcast_latency)
        total_tp_communication_latency += prefill_tp_communication_latency + decode_tp_communication_latency

    total_latency = total_prefill_computation_latency + total_decode_computation_latency + total_pp_communication_latency + total_tp_communication_latency
    return total_latency


class DPOptimizer:
    """Dynamic Programming based optimizer for throughput/cost optimization"""
    
    def __init__(self, config: Dict[str, Any], budget: float, latency_slo: float):
        """
        Args:
            config: 모델 및 시스템 설정
            budget: 예산 상한 (hourly cost in dollars)
            latency_slo: 지연시간 SLO (milliseconds)
        """
        self.config = config
        self.budget = budget
        self.latency_slo = latency_slo
        self.num_layers = config["num_layers"]
        self.instance_types = list(INSTANCE_SPEC.keys())
        
        # DP 캐시: dp[layer_idx][num_stages] = Pipeline
        # 최대 stage 수를 num_layers로 설정 (최악의 경우 각 layer가 하나의 stage)
        self.dp = [[None for _ in range(self.num_layers + 1)] for _ in range(self.num_layers + 1)]
        
        # computation latency 초기화
        initialize_computation_latencies(config)
    
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
        
        # 비용 계산
        pipeline.calculate_cost()
        
        # Throughput 계산
        pipeline.calculate_throughput(self.config)
        
        # Latency 계산
        pipeline.lowest_inference_latency = get_minimum_latency(pipeline, self.config)
        
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
                    
                    # 이전 상태들을 확인 (모든 가능한 스테이지 수에 대해)
                    for prev_num_stages in range(self.num_layers):
                        prev_pipeline = self.dp[prev_layer][prev_num_stages]
                        if prev_pipeline is None:
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
                        if new_pipeline.lowest_inference_latency > self.latency_slo:
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


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_config = AutoConfig.from_pretrained(model_name)

    look_rank = 5

    config = {
        "expected_input_len": 900,  # 입력 시퀀스 길이
        "expected_output_len": 1024,  # 출력 시퀀스 길이
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_hidden_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
        "intermediate_size": model_config.intermediate_size,
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "dtype": torch.float16,
        "max_model_len": 8192,
        "gpu_mem_utilization": 0.9,
        "total_model_mem": 140 * 10**9,  # 16GB Model Memory
    }

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

    # 테스트 시나리오 1: 낮은 예산, 느슨한 지연시간
    logger.info("-" * 80)
    logger.info("Test Case 1: Low budget ($15/hour), relaxed latency (60s)")
    logger.info("-" * 80)
    optimizer1 = DPOptimizer(config, budget=15.0, latency_slo=60000)
    
    start_time = time.time()
    result1 = optimizer1.optimize()
    optimization_time1 = time.time() - start_time
    
    if result1:
        logger.info(f"✓ Found optimal pipeline:")
        logger.info(f"  {result1}")
        logger.info(f"  - Stages: {result1.stages}")
        logger.info(f"  - Layers per stage: {result1.layer_per_stage}")
        logger.info(f"  - Total layers: {sum(result1.layer_per_stage)}")
        
        # 디버깅: 상위 look_rank 개 완전한 파이프라인 랭킹
        optimizer1.print_ranked_pipelines(max_rank=look_rank, only_complete=True)
    else:
        logger.info("✗ No feasible pipeline found within constraints")
    # 디버깅: DP 테이블 확인
    optimizer1.print_dp_table(layer_range=(config['num_layers']-4, config['num_layers']), stage_range=(1, 4))
    logger.info(f"⏱️  Optimization time: {optimization_time1:.3f} seconds")
    logger.info("")

    # 테스트 시나리오 2: 중간 예산, 중간 지연시간
    logger.info("-" * 80)
    logger.info("Test Case 2: Medium budget ($20/hour), moderate latency (5s)")
    logger.info("-" * 80)
    optimizer2 = DPOptimizer(config, budget=20.0, latency_slo=5000)
    
    start_time = time.time()
    result2 = optimizer2.optimize()
    optimization_time2 = time.time() - start_time
    
    if result2:
        logger.info(f"✓ Found optimal pipeline:")
        logger.info(f"  {result2}")
        logger.info(f"  - Stages: {result2.stages}")
        logger.info(f"  - Layers per stage: {result2.layer_per_stage}")
        logger.info(f"  - Total layers: {sum(result2.layer_per_stage)}")

        # 디버깅: 상위 look_rank 개 완전한 파이프라인 랭킹
        optimizer2.print_ranked_pipelines(max_rank=look_rank, only_complete=True)
    else:
        logger.info("✗ No feasible pipeline found within constraints")
    # 디버깅: DP 테이블 확인
    optimizer2.print_dp_table(layer_range=(config['num_layers']-4, config['num_layers']), stage_range=(1, 4))
    logger.info(f"⏱️  Optimization time: {optimization_time2:.3f} seconds")
    logger.info("")

    # 테스트 시나리오 3: 높은 예산, 엄격한 지연시간
    logger.info("-" * 80)
    logger.info("Test Case 3: High budget ($50/hour), strict latency (2s)")
    logger.info("-" * 80)
    optimizer3 = DPOptimizer(config, budget=50.0, latency_slo=2000)
    
    start_time = time.time()
    result3 = optimizer3.optimize()
    optimization_time3 = time.time() - start_time
    
    if result3:
        logger.info(f"✓ Found optimal pipeline:")
        logger.info(f"  {result3}")
        logger.info(f"  - Stages: {result3.stages}")
        logger.info(f"  - Layers per stage: {result3.layer_per_stage}")
        logger.info(f"  - Total layers: {sum(result3.layer_per_stage)}")

        # 디버깅: 상위 look_rank 개 완전한 파이프라인 랭킹
        optimizer3.print_ranked_pipelines(max_rank=look_rank, only_complete=True)
    else:
        logger.info("✗ No feasible pipeline found within constraints")
    # 디버깅: DP 테이블 확인
    optimizer3.print_dp_table(layer_range=(config['num_layers']-4, config['num_layers']), stage_range=(1, 4))
    logger.info(f"⏱️  Optimization time: {optimization_time3:.3f} seconds")
    logger.info("")

    # 테스트 시나리오 4: 매우 낮은 예산 (실패 케이스)
    logger.info("-" * 80)
    logger.info("Test Case 4: Very low budget ($1/hour), moderate latency (5s)")
    logger.info("-" * 80)
    optimizer4 = DPOptimizer(config, budget=1.0, latency_slo=5000)
    
    start_time = time.time()
    result4 = optimizer4.optimize()
    optimization_time4 = time.time() - start_time
    
    if result4:
        logger.info(f"✓ Found optimal pipeline:")
        logger.info(f"  {result4}")
        
        # 디버깅: 상위 look_rank 개 완전한 파이프라인 랭킹
        optimizer4.print_ranked_pipelines(max_rank=look_rank, only_complete=True)
    else:
        logger.info("✗ No feasible pipeline found within constraints (expected)")
    # 디버깅: DP 테이블 확인
    optimizer4.print_dp_table(layer_range=(config['num_layers']-4, config['num_layers']), stage_range=(1, 4))
    logger.info(f"⏱️  Optimization time: {optimization_time4:.3f} seconds")
    logger.info("")

    # 테스트 시나리오 5: 매우 엄격한 지연시간 (실패 케이스)
    logger.info("-" * 80)
    logger.info("Test Case 5: Medium budget ($20/hour), very strict latency (500ms)")
    logger.info("-" * 80)
    optimizer5 = DPOptimizer(config, budget=20.0, latency_slo=500)
    
    start_time = time.time()
    result5 = optimizer5.optimize()
    optimization_time5 = time.time() - start_time
    
    if result5:
        logger.info(f"✓ Found optimal pipeline:")
        logger.info(f"  {result5}")

        # 디버깅: 상위 look_rank 개 완전한 파이프라인 랭킹
        optimizer5.print_ranked_pipelines(max_rank=look_rank, only_complete=True)
    else:
        logger.info("✗ No feasible pipeline found within constraints (expected)")
    # 디버깅: DP 테이블 확인
    optimizer5.print_dp_table(layer_range=(config['num_layers']-4, config['num_layers']), stage_range=(1, 4))
    logger.info(f"⏱️  Optimization time: {optimization_time5:.3f} seconds")
    logger.info("")

    # 테스트 시나리오 6: 매우 높은 예산, 매우 엄격한 지연시간
    logger.info("-" * 80)
    logger.info("Test Case 6: Very high budget ($90/hour), very strict latency (500ms)")
    logger.info("-" * 80)
    optimizer6 = DPOptimizer(config, budget=90.0, latency_slo=500)
    
    start_time = time.time()
    result6 = optimizer6.optimize()
    optimization_time6 = time.time() - start_time
    
    if result6:
        logger.info(f"✓ Found optimal pipeline:")
        logger.info(f"  {result6}")
        logger.info(f"  - Stages: {result6.stages}")
        logger.info(f"  - Layers per stage: {result6.layer_per_stage}")
        logger.info(f"  - Total layers: {sum(result6.layer_per_stage)}")

        # 디버깅: 상위 look_rank 개 완전한 파이프라인 랭킹
        optimizer6.print_ranked_pipelines(max_rank=look_rank, only_complete=True)
    else:
        logger.info("✗ No feasible pipeline found within constraints")
    # 디버깅: DP 테이블 확인
    optimizer6.print_dp_table(layer_range=(config['num_layers']-4, config['num_layers']), stage_range=(1, 4))
    logger.info(f"⏱️  Optimization time: {optimization_time6:.3f} seconds")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("            Test Summary")
    logger.info("=" * 80)
    
    # 결과 요약
    results = [
        ("Test 1 (Low budget, relaxed latency)", result1, optimization_time1),
        ("Test 2 (Medium budget, moderate latency)", result2, optimization_time2),
        ("Test 3 (High budget, strict latency)", result3, optimization_time3),
        ("Test 4 (Very low budget)", result4, optimization_time4),
        ("Test 5 (Very strict latency)", result5, optimization_time5),
        ("Test 6 (Very high budget, very strict latency)", result6, optimization_time6)
    ]
    
    for test_name, result, opt_time in results:
        if result:
            logger.info(f"{test_name}: SUCCESS - {result} (⏱️ {opt_time:.3f}s)")
        else:
            logger.info(f"{test_name}: FAILED - No feasible solution (⏱️ {opt_time:.3f}s)")
    
    # 유효한 파이프라인 개수 출력
    logger.info(f"Valid pipelines found per test:")
    optimizers = [
        ("Test 1", optimizer1),
        ("Test 2", optimizer2),
        ("Test 3", optimizer3),
        ("Test 4", optimizer4),
        ("Test 5", optimizer5),
        ("Test 6", optimizer6)
    ]
    
    total_valid_count = 0
    for test_name, opt in optimizers:
        valid_count = len(opt.get_all_valid_pipelines())
        total_valid_count += valid_count
        logger.info(f"  {test_name}: {valid_count} pipelines")
    
    logger.info(f"  Total: {total_valid_count} pipelines")
    
    # 최적화 시간 통계
    optimization_times = [optimization_time1, optimization_time2, optimization_time3, optimization_time4, optimization_time5, optimization_time6]
    total_optimization_time = sum(optimization_times)
    avg_optimization_time = total_optimization_time / 6
    logger.info(f"Optimization time statistics:")
    logger.info(f"  Total optimization time: {total_optimization_time:.3f} seconds")
    logger.info(f"  Average optimization time: {avg_optimization_time:.3f} seconds")
    logger.info(f"  Fastest optimization: {min(optimization_times):.3f} seconds")
    logger.info(f"  Slowest optimization: {max(optimization_times):.3f} seconds")
