import copy
from typing import Any, Dict, List, Tuple, Optional
from transformers import AutoConfig
import logging
import time
from estimator_utils import *
from cluster_pool import ClusterPool
import heapq

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self):
        self.stages = []
        self.azs = []
        self.layer_per_stage = []
        self.throughput = 0
        self.cost = 0
        self.global_batch_size = 0
        self.num_cache_blocks = 0
        self.latency_per_global_batch = 0
        self.num_blocks = 0
        self.block_size = 16
        self.single_request_latency = float('inf')

    def __repr__(self):
        """String representation of the Pipeline object"""
        if not self.stages:
            return "Pipeline(empty)"

        stage_info = []
        for instance, layers in zip(self.stages, self.layer_per_stage):
            stage_info.append(f"{instance}:{layers}L")

        return (f"Pipeline(stages=[{', '.join(stage_info)}],\n"
                f"\tthroughput={self.throughput:.3f},\n"
                f"\tcost=${self.cost:.3f},\n"
                f"\tlatency_per_global_batch={self.latency_per_global_batch:.0f}ms,\n"
                f"\tsingle_request_latency={self.single_request_latency:.0f}ms,\n"
                f"\tnum_blocks={self.num_blocks},\n"
                f"\tglobal_batch_size={self.global_batch_size})")

    def set_cost(self, cost: float):
        """Sets the total cost of the pipeline (hourly cost)."""
        self.cost = cost

    def calculate_throughput(self, config: Dict[str, Any]) -> float:
        """Calculates the pipeline's throughput and stores it in self.throughput."""
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

def get_minimum_latency(pipeline: Pipeline, config: Dict[str, Any]):
    """Calculates the minimum latency of the pipeline."""
    node_layer_comb = []
    for i, (instance, layer_count) in enumerate(zip(pipeline.stages, pipeline.layer_per_stage)):
        node_layer_comb.append((instance, pipeline.azs[i], layer_count))

    latency = get_single_request_latency(
        avg_input_len=config["expected_input_len"],
        avg_output_len=config["expected_output_len"],
        hidden_dim=config["hidden_size"],
        num_attention_head=config["num_attention_heads"],
        num_kv_cache_head=config["num_key_value_heads"],
        total_num_layers=config["num_layers"],
        intermediate_dim=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        node_layer_comb=node_layer_comb,
        dtype=config["dtype"],
        head_dim=config.get("head_dim"),
    )

    return latency


cache_hit_count = 0
cache_miss_count = 0

latency_cache_hit_count = 0
latency_cache_miss_count = 0

class BeamSearchDPOptimizer:
    """Dynamic Programming based optimizer with aggressive pruning"""

    def __init__(self,
                 config: Dict[str, Any],
                 budget: float,
                 latency_slo: float,
                 cluster_pool: ClusterPool,
                 max_stages: Optional[int] = None,
                 top_k: int = 5,
                 optimization_mode: str = "soft_slo"):

        self.config = config
        self.budget = budget
        self.latency_slo = latency_slo
        self.num_layers = config["num_layers"]
        self.cluster_pool = cluster_pool
        self.optimization_mode = optimization_mode

        # Pre-filter instances based on budget
        self.instance_types = []
        for instance in cluster_pool.available_resources.keys():
            price = self.cluster_pool.get_instance_price(instance)
            if price <= self.budget and self.cluster_pool.available_resources[instance] > 0: # Only consider affordable instances
                self.instance_types.append(instance)

        logger.debug(f"Filtered to {len(self.instance_types)} affordable instances from {len(cluster_pool.available_resources)}")

        # Cache for expensive computations - use sorted key for better hit rate
        self._latency_cache = {}
        self._throughput_cache = {}

        self.max_stages = max_stages if max_stages is not None else self.num_layers
        self.top_k = top_k

        # DP cache
        self.dp: List[List[Dict[Pipeline]]] = [[{} for _ in range(self.num_layers + 1)] for _ in range(self.num_layers + 1)]

    def _get_cache_key(self, pipeline: Pipeline) -> tuple:
        """Generate cache key with sorted stages for better hit rate"""
        # Sort by (instance, layers) to maximize cache hits
        stage_info = list(zip(pipeline.stages, pipeline.layer_per_stage))
        return tuple(sorted(stage_info))

    def _create_pipeline(self, stages: List[Tuple[str, int]]) -> Pipeline:
        """Creates a Pipeline object from stage information"""
        pipeline = Pipeline()
        if len(stages) <= 0:
            return pipeline

        for instance, layer_count in stages:
            pipeline.stages.append(instance)
            pipeline.azs.append("dummy-az")
            pipeline.layer_per_stage.append(layer_count)

        total_cost = 0
        for instance in pipeline.stages:
            total_cost += self.cluster_pool.get_instance_price(instance)
        pipeline.set_cost(total_cost)

        pipeline.calculate_throughput(self.config)
        pipeline.single_request_latency = get_minimum_latency(pipeline, self.config)

        return pipeline

    def _add_new_stage(self, pipeline: Pipeline, instance: str, layers: int) -> Pipeline:
        """Creates a new pipeline by adding a new stage to an existing pipeline"""
        if layers <= 0:
            return None

        # Early termination: check cost before creating new pipeline
        new_cost = pipeline.cost + self.cluster_pool.get_instance_price(instance)
        if new_cost > self.budget:
            return None

        new_pipeline = copy.deepcopy(pipeline)
        new_pipeline.stages.append(instance)
        new_pipeline.azs.append("dummy-az")
        new_pipeline.layer_per_stage.append(layers)
        new_pipeline.set_cost(new_cost)

        global latency_cache_hit_count, latency_cache_miss_count

        # Use cached latency
        cache_key = self._get_cache_key(new_pipeline)
        if cache_key not in self._latency_cache:
            self._latency_cache[cache_key] = get_minimum_latency(new_pipeline, self.config)
            latency_cache_miss_count += 1
        new_pipeline.single_request_latency = self._latency_cache[cache_key]
        latency_cache_hit_count += 1

        return new_pipeline

    def _recalculate_pipeline_throughput(self, pipeline: Pipeline) -> Pipeline:
        """Recalculates the pipeline's throughput"""
        cache_key = self._get_cache_key(pipeline)

        global cache_hit_count, cache_miss_count
        if cache_key not in self._throughput_cache:
            pipeline.calculate_throughput(self.config)
            self._throughput_cache[cache_key] = (
                pipeline.throughput,
                pipeline.latency_per_global_batch,
                pipeline.num_blocks,
                pipeline.global_batch_size
            )
            cache_miss_count += 1
        else:
            cached_values = self._throughput_cache[cache_key]
            pipeline.throughput = cached_values[0]
            pipeline.latency_per_global_batch = cached_values[1]
            pipeline.num_blocks = cached_values[2]
            pipeline.global_batch_size = cached_values[3]
            cache_hit_count += 1
        return pipeline

    def _check_feasibility_pipeline(self, pipeline: Pipeline, slo: int, budget: float) -> bool:
        """Checks whether the pipeline satisfies the given SLO and budget constraints"""
        if pipeline is None:
            return False
        if pipeline.cost > budget:
            return False
        # Throughput check is performed in _evaluate_pipeline
        if self.optimization_mode == "hard_slo" and pipeline.single_request_latency > slo:
            return False
        return True

    def _evaluate_pipeline(self, pipeline: Pipeline):
        """
        Pipeline evaluation with multi-objective optimization.

        optimization_mode options:
        - "hard_slo": Simple throughput/cost ratio (respects hard latency constraint)
        - "soft_slo": Linear penalty for SLO exceedance with single hyperparameter α

        For soft_slo:
        - If latency ≤ SLO: score = efficiency (no penalty)
        - If latency > SLO: score = efficiency × (1 - α × (latency - SLO) / SLO)
        - α → ∞: behaves like hard_slo
        """
        if pipeline.cost == 0:
            return 0

        if pipeline.throughput <= 0:
            return 0

        efficiency = pipeline.throughput / pipeline.cost

        if self.optimization_mode == "hard_slo":
            return efficiency

        elif self.optimization_mode == "soft_slo":
            alpha = 1.0
            latency = pipeline.single_request_latency
            slo = self.latency_slo
            slo_excess_ratio = max(0, (latency - slo) / slo)
            penalty_term = alpha * slo_excess_ratio
            score = efficiency * (1 - penalty_term)
            return score

        elif self.optimization_mode == "only_throughput":
            return pipeline.throughput

        # Default fallback
        return efficiency

    def optimize(self) -> List[Pipeline]:
        # Base case
        self.dp[0][0][()] = self._create_pipeline([])

        # Measure algorithm overhead
        check_cluster_availability_time = 0
        add_new_stage_time = 0
        sort_pipeline_signature_time = 0
        check_feasibility_time = 0
        recalculate_throughput_time = 0
        count_add_new_stage = 0
        count_recalculate_pipeline_throughput = 0

        # Track top 5 pipelines using heap (min-heap)
        top_pipelines = []  # List of (score, counter, pipeline) tuples
        entry_counter = 0  # Counter for tie-breaking

        # Cache statistics
        global cache_hit_count, cache_miss_count, latency_cache_hit_count, latency_cache_miss_count
        cache_hit_count = 0
        cache_miss_count = 0
        latency_cache_hit_count = 0
        latency_cache_miss_count = 0

        # Perform DP
        for pivot_layer in range(1, self.num_layers + 1):
            if pivot_layer % 10 == 0 or pivot_layer == self.num_layers:
                logger.info(f"Processing layer {pivot_layer}/{self.num_layers}...")
            else:
                logger.debug(f"Processing layer {pivot_layer}/{self.num_layers}...")

            for prev_num_layer in range(0, pivot_layer):  # Limit lookback range
                new_num_layer = pivot_layer - prev_num_layer

                for prev_num_stage in range(0, min(prev_num_layer + 1, self.max_stages)):
                    if len(self.dp[prev_num_layer][prev_num_stage]) == 0:
                        continue

                    current_num_stage = prev_num_stage + 1

                    # Only process top pipelines from previous stage
                    sorted_pipelines = sorted(
                        self.dp[prev_num_layer][prev_num_stage].items(),
                        key=lambda x: self._evaluate_pipeline(x[1]),
                        reverse=True
                    )[:self.top_k]

                    for pipeline_signature, prev_pipeline in sorted_pipelines:
                        for new_stage_instance in self.instance_types:
                            start_time = time.time()
                            if not self.cluster_pool.check_cluster_availability(prev_pipeline.stages + [new_stage_instance]):
                                end_time = time.time()
                                check_cluster_availability_time += (end_time - start_time)
                                continue
                            end_time = time.time()
                            check_cluster_availability_time += (end_time - start_time)

                            start_time = time.time()
                            new_pipeline = self._add_new_stage(prev_pipeline, new_stage_instance, new_num_layer)
                            count_add_new_stage += 1
                            if new_pipeline is None:
                                end_time = time.time()
                                add_new_stage_time += (end_time - start_time)
                                continue
                            end_time = time.time()
                            add_new_stage_time += (end_time - start_time)

                            start_time = time.time()
                            if not self._check_feasibility_pipeline(new_pipeline, self.latency_slo, self.budget):
                                end_time = time.time()
                                check_feasibility_time += (end_time - start_time)
                                continue
                            end_time = time.time()
                            check_feasibility_time += (end_time - start_time)

                            start_time = time.time()
                            new_pipeline = self._recalculate_pipeline_throughput(new_pipeline)
                            end_time = time.time()
                            recalculate_throughput_time += (end_time - start_time)
                            count_recalculate_pipeline_throughput += 1
                            new_pipeline_score = self._evaluate_pipeline(new_pipeline)
                            

                            # Track top 5 pipelines for final layer
                            if pivot_layer == self.num_layers:
                                # Use a counter as tie-breaker to avoid Pipeline comparison
                                entry_counter += 1
                                if len(top_pipelines) < 5:
                                    heapq.heappush(top_pipelines, (new_pipeline_score, entry_counter, copy.deepcopy(new_pipeline)))
                                elif new_pipeline_score > top_pipelines[0][0]:
                                    heapq.heapreplace(top_pipelines, (new_pipeline_score, entry_counter, copy.deepcopy(new_pipeline)))

                            new_pipeline_signature = tuple(sorted(new_pipeline.stages))

                            # Update DP table
                            if new_pipeline_signature in self.dp[pivot_layer][current_num_stage]:
                                existing_pipeline = self.dp[pivot_layer][current_num_stage][new_pipeline_signature]
                                if new_pipeline_score > self._evaluate_pipeline(existing_pipeline):
                                    self.dp[pivot_layer][current_num_stage][new_pipeline_signature] = new_pipeline
                            else:
                                if len(self.dp[pivot_layer][current_num_stage]) < self.top_k:
                                    self.dp[pivot_layer][current_num_stage][new_pipeline_signature] = new_pipeline
                                else:
                                    # Find and replace minimum
                                    min_score = float('inf')
                                    min_sig = None
                                    for sig, pl in self.dp[pivot_layer][current_num_stage].items():
                                        pl_score = self._evaluate_pipeline(pl)
                                        if pl_score < min_score:
                                            min_score = pl_score
                                            min_sig = sig

                                    if new_pipeline_score > min_score:
                                        del self.dp[pivot_layer][current_num_stage][min_sig]
                                        self.dp[pivot_layer][current_num_stage][new_pipeline_signature] = new_pipeline

        logger.debug(f"✓ DP Table populated.")
        logger.debug(f"  - check_cluster_availability_time: {check_cluster_availability_time:.3f} seconds")
        logger.debug(f"  - add_new_stage_time: {add_new_stage_time:.3f} seconds")
        logger.debug(f"  - sort_pipeline_signature_time: {sort_pipeline_signature_time:.3f} seconds")
        logger.debug(f"  - check_feasibility_time: {check_feasibility_time:.3f} seconds")
        logger.debug(f"  - recalculate_throughput_time: {recalculate_throughput_time:.3f} seconds")
        logger.debug(f"  - count_add_new_stage: {count_add_new_stage}")
        logger.debug(f"  - count_recalculate_pipeline_throughput: {count_recalculate_pipeline_throughput}")
        logger.debug(f"  - recalculate throughput time per call: {recalculate_throughput_time / max(1, count_recalculate_pipeline_throughput):.6f} seconds")
        logger.debug(f"  - add new stage time per call: {add_new_stage_time / max(1, count_add_new_stage):.6f} seconds")
        logger.debug(f"  - Throughput Cache Hits: {cache_hit_count}, Misses: {cache_miss_count}")
        logger.debug(f"  - Latency Cache Hits: {latency_cache_hit_count}, Misses: {latency_cache_miss_count}")
        logger.debug(f"  - Throughput Cache Size: {len(self._throughput_cache)}")
        logger.debug(f"  - Latency Cache Size: {len(self._latency_cache)}, Throughput Cache Size: {len(self._throughput_cache)}")

        # Sort top 5 pipelines by score (descending) and return just the pipelines
        sorted_pipelines = sorted(top_pipelines, key=lambda x: x[0], reverse=True)
        return [pipeline for score, _, pipeline in sorted_pipelines]


def run_test_case(config: Dict, budget: float, latency_slo: float, cluster_pool: ClusterPool,
                  top_k: int, look_rank: int = 5, max_stages: Optional[int] = None
                  , optimization_mode: str = "soft_slo") -> Tuple[List[Pipeline], BeamSearchDPOptimizer, float]:
    optimizer = BeamSearchDPOptimizer(config, budget=budget, latency_slo=latency_slo,
                                      cluster_pool=cluster_pool, max_stages=max_stages, top_k=top_k,
                                      optimization_mode=optimization_mode)

    start_time = time.time()
    results: List[Pipeline] = optimizer.optimize()
    optimization_time = time.time() - start_time

    if results:
        logger.debug(f"✓ Found top {len(results)} pipelines:")
        for i, pipeline in enumerate(results, 1):
            score = optimizer._evaluate_pipeline(pipeline)
            logger.debug(f"\nRank {i}:")
            logger.debug(f"  {pipeline}")
            logger.debug(f"  - Score: {score:.4f}")
            logger.debug(f"  - Throughput : {pipeline.throughput:.2f} req/s")
            logger.debug(f"  - Cost : {pipeline.cost:.2f} USD/h")
            logger.debug(f"  - Single Request Latency : {pipeline.single_request_latency:.2f} ms")
        logger.debug(f"\n⏱️  Optimization time: {optimization_time:.3f} seconds")
    else:
        logger.debug("✗ No feasible pipeline found within constraints")

    return results, optimizer, optimization_time


if __name__ == "__main__":
    # Test with same configuration as original
    model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_config = AutoConfig.from_pretrained(model_name)

    config = {
        "expected_input_len": 763,
        "expected_output_len": 232,
        "hidden_size": model_config.hidden_size,
        "num_layers": model_config.num_hidden_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": getattr(model_config, "num_key_value_heads", model_config.num_attention_heads),
        "intermediate_size": model_config.intermediate_size,
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "dtype": torch.float16,
        "max_model_len": 8192,
        "gpu_mem_utilization": 0.85
    }

    optimization_mode = "soft_slo"

    available_nodes = {
        "(spot)g5.12xlarge": 2,
        "(spot)g6.12xlarge": 3,
        "(spot)g6e.xlarge": 4,
    }

    prices = {
        "(spot)g5.12xlarge":  2.2915,
        "(spot)g6.12xlarge":  1.9445,
        "(spot)g6e.xlarge":   0.7040,
    }

    cluster_pool = ClusterPool(available_spot_nodes=available_nodes, spot_prices=prices)

    logger.debug("=" * 80)
    logger.debug("Optimized Dynamic Programming Test")
    logger.debug("=" * 80)

    run_test_case(config, budget=9999, latency_slo=99999999, cluster_pool=cluster_pool,
                  max_stages=13, top_k=1, optimization_mode=optimization_mode)