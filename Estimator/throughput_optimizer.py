import copy
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
        self.num_blocks = 0 # 파이프라인의 block 개수
        self.block_size = 16
        self.single_request_latency = float('inf')
    
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
                f"latency_per_global_batch={self.latency_per_global_batch:.0f}ms, "
                f"single_request_latency={self.single_request_latency:.0f}ms, "
                f"num_blocks={self.num_blocks})")

    def set_cost(self, cost: float):
        """파이프라인의 총 비용을 설정합니다 (hourly cost)."""
        self.cost = cost
    
    def calculate_throughput(self, config: Dict[str, Any]) -> float:
        """파이프라인의 throughput을 계산하고 self.throughput에 저장합니다."""
        node_layer_comb = []
        for i, (instance, layer_count) in enumerate(zip(self.stages, self.layer_per_stage)):
            node_layer_comb.append((instance, self.azs[i], layer_count))
        
        throughput, total_latency_per_global_batch, num_blocks = get_throughput(
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
        self.num_blocks = num_blocks
        self.global_batch_size = num_blocks * self.block_size // (config["expected_input_len"] + config["expected_output_len"])
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


class BeamSearchDPOptimizer:
    """Dynamic Programming based optimizer for throughput/cost optimization"""
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 budget: float, 
                 latency_slo: float, 
                 cluster_pool: ClusterPool, 
                 max_stages: Optional[int] = None,
                 top_k: int = 5):
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

        # Beam Search 를 도입한다. Top-k 설정.
        # k 가 높아지면 나중에 heap 으로 변환하는 것이 효율적이다.
        # 하지만 현재 k 는 5 이하로 설정할 것이기 때문에 그냥 naive 한 기법을 사용한다.
        self.top_k = top_k

        # DP 캐시: dp[layer_idx][num_stages] = Pipelines
        # 최대 stage 수를 num_layers로 설정 (최악의 경우 각 layer가 하나의 stage)
        # Beam 은 Dictionary 형태를 유지한다. (같은 인스턴스 구성이면 중복을 제거한다.)
        self.dp: List[List[Dict[Pipeline]]] = [[{} for _ in range(self.num_layers + 1)] for _ in range(self.num_layers + 1)]
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
        
        # 비용 계산 - ClusterPool의 가격 사용
        total_cost = 0
        for instance in pipeline.stages:
            total_cost += self.cluster_pool.get_instance_price(instance)
        pipeline.set_cost(total_cost)
        
        # Throughput 및 latency 계산
        pipeline.calculate_throughput(self.config)
        pipeline.single_request_latency = get_minimum_latency(pipeline, self.config)
        
        return pipeline

    def _add_new_stage(self, pipeline: Pipeline, instance: str, layers: int) -> Pipeline:
        """기존 파이프라인에 새로운 스테이지를 추가하여 새로운 파이프라인 생성"""
        if layers <= 0:
            assert False, "layers must be positive"
        
        new_pipeline = copy.deepcopy(pipeline)
        new_pipeline.stages.append(instance)
        new_pipeline.azs.append("dummy-az")
        new_pipeline.layer_per_stage.append(layers)

        # add new stage 부터 throughput 계산을 해버리면 feasibility check 전에 매번 계산해야하므로
        # 시간이 매우 오래걸릴 것이다. 따라서 calculate 하는 경우에는 함수를 분리해서 feasibility check 이후에 한다.
        # new_pipeline.calculate_throughput(self.config)

        new_pipeline.set_cost(new_pipeline.cost + self.cluster_pool.get_instance_price(instance))
        # single request latency 의 경우 slo constraint 에 박혀있기 때문에 필요하다.
        new_pipeline.single_request_latency = get_minimum_latency(new_pipeline, self.config)

        return new_pipeline

    def _recalculate_pipeline_throughput(self, pipeline: Pipeline) -> Pipeline:
        """파이프라인의 throughput을 재계산"""
        pipeline.calculate_throughput(self.config)
        return pipeline

    
    def _check_feasibility_pipeline(self, pipeline: Pipeline, slo: int, budget: float) -> bool:
        """파이프라인이 주어진 SLO 및 예산 제약을 만족하는지 확인"""
        if pipeline is None:
            return False
        if pipeline.single_request_latency > slo:
            return False
        if pipeline.cost > budget:
            return False
        if pipeline.throughput == OUT_OF_MEMORY:
            return False
        return True
    
    def optimize(self) -> Optional[Pipeline]:
        """
        Dynamic Programming을 사용하여 throughput/cost를 최대화하는 최적의 파이프라인을 찾습니다.
        
        목적:
        - 주어진 예산(budget)과 지연시간 SLO 제약 하에서
        - throughput/cost 비율을 최대화하는 파이프라인 구성을 찾습니다
        
        알고리즘:
        1. DP 상태: dp[i][j] = i개의 레이어를 j개의 스테이지로 처리하는 최적 파이프라인 Beam
        2. 전이: 새로운 인스턴스를 추가하여 더 많은 레이어를 처리
        3. 제약조건 체크: 예산, 지연시간, 메모리 제약을 만족하는지 확인
        
        Returns:
            최적의 Pipeline 객체 또는 실행 가능한 파이프라인이 없으면 None
        """
        # Base case: 0개 레이어, 0개 스테이지로 시작
        self.dp[0][0][()] = self._create_pipeline([])
        
        # DP 수행: 1개부터 전체 레이어 수까지 순차적으로 최적해를 구함
        for pivot_layer in range(1, self.num_layers + 1):
            print(f"Populate DP table for {pivot_layer} pivot layer...")
            for prev_num_layer in range(0, pivot_layer):
                new_num_layer = pivot_layer - prev_num_layer
                for prev_num_stage in range(0, min(prev_num_layer + 1, self.max_stages)):
                    # beam search, 여기서 signature 는 stages 의 튜플이다.
                    # 아래 for 문이 최대 k 번만큼 순회함
                    for pipeline_signature, prev_pipeline in self.dp[prev_num_layer][prev_num_stage].items():
                        current_num_stage = prev_num_stage + 1

                        # 새로운 스테이지 추가
                        for new_stage_instance in self.instance_types:
                            # 새로운 인스턴스 추가시 cluster 의 가용성을 체크
                            if not self.cluster_pool.check_cluster_availability(prev_pipeline.stages + [new_stage_instance]):
                                continue

                            new_pipeline = self._add_new_stage(prev_pipeline, new_stage_instance, new_num_layer)
                            new_pipeline_signature = tuple(sorted(new_pipeline.stages))
                            # 제약조건을 만족하지 못하면 건너뜀
                            if not self._check_feasibility_pipeline(new_pipeline, self.latency_slo, self.budget):
                                continue
                            new_pipeline = self._recalculate_pipeline_throughput(new_pipeline)
                                
                            # 이미 같은 signature 가 dp table 안에 존재하는 경우 더 좋은 것만 남김
                            if new_pipeline_signature in self.dp[pivot_layer][current_num_stage].keys():
                                existing_pipeline = self.dp[pivot_layer][current_num_stage][new_pipeline_signature]
                                if (new_pipeline.throughput / new_pipeline.cost) > (existing_pipeline.throughput / existing_pipeline.cost):
                                    self.dp[pivot_layer][current_num_stage][new_pipeline_signature] = new_pipeline
                            # 해당 signature 가 dp table 에 존재하지 않는 경우 추가
                            # 단 beam search 를 고려하여 top-k 개수만 유지해야 함
                            else:
                                if len(self.dp[pivot_layer][current_num_stage].keys()) < self.top_k:
                                    self.dp[pivot_layer][current_num_stage][new_pipeline_signature] = new_pipeline
                                else:
                                    min_ratio = float('inf')
                                    min_signature = None
                                    for sig, pl in self.dp[pivot_layer][current_num_stage].items():
                                        ratio = pl.throughput / pl.cost
                                        if ratio < min_ratio:
                                            min_ratio = ratio
                                            min_signature = sig
                                    # 새로운 파이프라인이 기존의 최소 ratio 보다 크면 교체
                                    if (new_pipeline.throughput / new_pipeline.cost) > min_ratio:
                                        del self.dp[pivot_layer][current_num_stage][min_signature]
                                        self.dp[pivot_layer][current_num_stage][new_pipeline_signature] = new_pipeline

        
        # 최종 결과 선택: 모든 레이어를 처리하는 파이프라인 중에서
        # throughput/cost 비율이 가장 높은 것을 선택
        best_pipeline = None
        best_ratio = 0
        
        # 모든 가능한 스테이지 수에 대해 확인
        for num_stages in range(1, min(self.num_layers + 1, self.max_stages + 1)):
            pipelines = self.dp[self.num_layers][num_stages]
            for _, pipeline in pipelines.items():
                ratio = pipeline.throughput / pipeline.cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pipeline = pipeline
        
        return best_pipeline

    def show_pipeline_ranking(self, look_rank: int = 5):
        """DP 테이블에 저장된 파이프라인들을 throughput/cost 비율 기준으로 정렬하여 출력"""
        import heapq
        ranked_pipelines = []
        for stage in range(1, self.max_stages + 1):
            # search beam
            for _, pipeline in self.dp[self.num_layers][stage].items():
                ratio = pipeline.throughput / pipeline.cost
                if len(ranked_pipelines) < look_rank:
                    heapq.heappush(ranked_pipelines, (ratio, pipeline))
                else:
                    heapq.heapreplace(ranked_pipelines, (ratio, pipeline))

        sorted_by_ratio = sorted(ranked_pipelines, key=lambda x: x[0], reverse=True)
        logger.info(f"Top-{look_rank} Pipelines in DP Table (by throughput/cost ratio):")
        for rank, (ratio, pipeline) in enumerate(sorted_by_ratio, start=1):
            logger.info(f"Rank {rank}: {pipeline} with ratio={ratio:.3f}")



def run_test_case(
        config: Dict, 
        budget: float, 
        latency_slo: float, 
        cluster_pool: ClusterPool,
        top_k: int,
        look_rank: int = 5,
        max_stages: Optional[int] = None) -> Tuple[Pipeline, BeamSearchDPOptimizer, float]:
    """
    Run a single test case with given budget and SLO.
    
    Args:
        config: Model configuration dictionary
        budget: Budget in USD per hour
        latency_slo: Latency SLO in milliseconds
        look_rank: Number of top pipelines to show
        max_stages: Maximum number of stages allowed (None for no limit)
    
    Returns:
        Tuple of (Optimal Pipeline, BeamSearchDPOptimizer, optimization_time)
    """
    optimizer = BeamSearchDPOptimizer(config, budget=budget, latency_slo=latency_slo, cluster_pool=cluster_pool, max_stages=max_stages, top_k=top_k)
    
    start_time = time.time()
    result: Pipeline = optimizer.optimize()
    optimization_time = time.time() - start_time
    
    if result:
        logger.info(f"✓ Found optimal pipeline:")
        logger.info(f"  {result}")
        logger.info(f"  - Stages: {result.stages}")
        logger.info(f"  - Layers per stage: {result.layer_per_stage}")
        logger.info(f"  - Total layers: {sum(result.layer_per_stage)}")
        logger.info(f"  - Throughput : {result.throughput:.2f} req/s")
        logger.info(f"  - Cost : {result.cost:.2f} USD/h")
        logger.info(f"  - Global Batch : {result.global_batch_size}")
        logger.info(f"  - E2E Latency per Global Batch : {result.latency_per_global_batch:.2f} ms")
        logger.info(f"  - Single Request E2E Latency : {result.single_request_latency:.2f} ms")
        logger.info(f"  - Num Available Blocks : {result.num_blocks}")
    else:
        logger.info("✗ No feasible pipeline found within constraints")
    
    # Show DP table for debugging
    # TODO: Show Pipeline Ranking
    optimizer.show_pipeline_ranking(look_rank=look_rank)
    logger.info(f"⏱️  Optimization time: {optimization_time:.3f} seconds")
    logger.info("")
    
    return result, optimizer, optimization_time


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_config = AutoConfig.from_pretrained(model_name)

    look_rank = 5
    max_stages = 10
    top_k = 5

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
        "max_model_len": 8192,
        "gpu_mem_utilization": 0.9
    }

    available_spot_nodes = {
        "(spot)g4dn.xlarge":      10,
        "(spot)g4dn.12xlarge":    10,
        "(spot)g4dn.metal":       10,
        "(spot)g5.xlarge":        10,
        "(spot)g5.12xlarge":      10,
        "(spot)g5.48xlarge":      10,
        "(spot)g6.xlarge":        16,
        "(spot)g6.12xlarge":      4,
        "(spot)g6.48xlarge":      2,
        "(spot)g6e.xlarge":       10,
        "(spot)g6e.12xlarge":     4,
        "(spot)g6e.48xlarge":     1,
        "(spot)p4d.24xlarge":     0,
        "(spot)p4de.24xlarge":    0,
        "(spot)p5.4xlarge":       0,    
        "(spot)p5.48xlarge":      0,
        "(spot)p5e.48xlarge":     0,
        "(spot)p5en.48xlarge":    0,
        "(spot)p6-b200.48xlarge": 0,
    }
    # us-west-2 의 2025-08-26 00:00 에서의 가격 기록
    # AZ 별로 가격이 약간씩 상이하나 인스턴스가 존재하는 첫 번째 AZ 기준으로 설정
    spot_prices = {
        "(spot)g4dn.xlarge":      0.2523,
        "(spot)g4dn.12xlarge":    1.4673,
        "(spot)g4dn.metal":       3.4434,
        "(spot)g5.xlarge":        0.6424,
        "(spot)g5.12xlarge":      2.4761,
        "(spot)g5.48xlarge":      6.3587,
        "(spot)g6.xlarge":        0.4207,
        "(spot)g6.12xlarge":      2.1210,
        "(spot)g6.48xlarge":      5.3874,
        "(spot)g6e.xlarge":       0.9613,
        "(spot)g6e.12xlarge":     5.0399,
        "(spot)g6e.48xlarge":     13.2044,
        "(spot)p4d.24xlarge":     10.7828,
        "(spot)p4de.24xlarge":    14.6877,
        "(spot)p5.4xlarge":       9999, # no available spot on us-west-2
        "(spot)p5.48xlarge":      18.1301,
        "(spot)p5e.48xlarge":     21.6759,
        "(spot)p5en.48xlarge":    22.5218,
        "(spot)p6-b200.48xlarge": 29.2084,
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

    # 기준 SLO (p4d.24xlarge -> 8xA100 에서 512+128 길이의 요청을 처리할 때의 single request latency latency) (ms)
    baseline_SLO = 512

    # 테스트 시나리오
    logger.info("-" * 80)
    logger.info("Test Case 1: Budget ($20/hour), 10x SLO")
    logger.info("-" * 80)
    run_test_case(config, budget=20, latency_slo=baseline_SLO * 30, look_rank=look_rank, cluster_pool=cluster_pool, max_stages=max_stages, top_k=top_k)
    
