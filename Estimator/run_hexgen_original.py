#!/usr/bin/env python3
"""
Run HEXGEN with original functions, modified only for cluster configuration
"""

import sys
import os
import copy
import random
import numpy as np

# Add HEXGEN path
sys.path.append(os.path.join(os.path.dirname(__file__), 'hexgen'))

# Import all original HEXGEN modules
from deap import base, creator, tools, algorithms
from itertools import product
from operator import attrgetter
from gen_plan import generate_unique_combinations
from predict_cost import predict_cost
from validate_and_adjust import validate_and_adjust
from cost_model_impl import compute_costs, tp_communication_costs, inter_device_communication_cost
from cost_function_for_mutation import cost_function
from simulator_v2 import Simulator

# Set random seed
random.seed(123)
np.random.seed(123)

# Re-create DEAP types if they exist
if hasattr(creator, "FitnessMin"):
    del creator.FitnessMin
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", object, fitness=creator.FitnessMin)

# Global configuration - can be modified for different clusters
initial_array = []
restriction_array = []
gpu_mem_limit_list = []
comp_abilities = []
comm_abilities = []

def set_cluster_config(config):
    """Set global variables for HEXGEN"""
    global initial_array, restriction_array, gpu_mem_limit_list, comp_abilities, comm_abilities
    
    initial_array = config['initial_array']
    restriction_array = config.get('restriction_array', initial_array)
    gpu_mem_limit_list = config['gpu_mem_limit_list']
    comp_abilities = config.get('comp_abilities', [[1.0] * len(group) for group in initial_array])
    comm_abilities = config.get('comm_abilities', [[1.0] * len(group) for group in initial_array])

# Copy original functions from genetic_algorithm.py
def compare_arrays_and_calculate_cost(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    comparison = array1 > array2
    cost = np.sum(comparison) * 1e12
    return cost

def generate_gpu_memory_limits(gpu_array, gpu_limit):
    result = []
    for i, subgroup in enumerate(gpu_array):
        limit = gpu_limit[i % len(gpu_limit)]
        total_gpus_in_subgroup = sum(subgroup)
        result.extend([limit] * total_gpus_in_subgroup)
    return result

def initialize_individual():
    ind = creator.Individual()
    # Initialize with the cluster configuration
    # Start with all nodes in a single stage per group
    ind.genes = [[group[:]] for group in initial_array]  # Deep copy of initial_array
    ind.bias = [0] * len(ind.genes)
    ind.batch = [1] * len(ind.genes)
    ind.iter = 0
    ind.goodput_store = 0
    
    return ind

def evaluate(individual): 
    group_plan_list = []
    group_cost_list = []
    group_mem_list = []
    group_pp_partition_list = []
    group_batch_list = []
    
    for sub_group_index in range(len(individual.genes)):
        individual_genes = individual.genes[sub_group_index] 
        # print(individual_genes) # [[4, 2, 4, 2], [4, 2, 4, 2]]
        sub_group_plan_list = []
        sub_group_cost_list = []
        sub_group_mem_list = []
        sub_group_pp_partition_list = []
        sub_batch_list = []
        if len(individual.bias) < len(individual_genes):
            # init new sub group's partition coe as 0
            individual.bias = individual.bias + [0] * (len(individual_genes) - len(individual.bias))
            # init new sub group's batch as 1
            individual.batch = individual.batch + [1] * (len(individual_genes) - len(individual.batch))
        for sub_group, bias_value, bsz in zip(individual_genes, individual.bias, individual.batch):
            # print(f"sub_group: {sub_group}, bias_value: {bias_value}, bsz: {bsz}") # sub_group: [4, 2, 4, 2], bias_value: -1, bsz: 1
            all_unique_combinations = [generate_unique_combinations(num_gpus) for num_gpus in sub_group]
            # all_unique_combinations: [[(2, 2), (1, 1, 1, 1), (4,), (1, 1, 2)], [(1, 1), (2,)], [(2, 2), (1, 1, 1, 1), (4,), (1, 1, 2)], [(1, 1), (2,)]]
            # print(f"all_unique_combinations: {all_unique_combinations}")
            final_combinations = []
            for combination in product(*all_unique_combinations):
                final_combinations.append(list(combination))
            cost_list = []
            # print(final_combinations)
            for alloc in final_combinations:
                # Each alloc is a sub_group plan: [(1, 2, 1), (1, 2, 1), (1, 2, 1)]
                # Convert alloc to parallel config list: [1, 2, 1, 1, 2, 1, 1, 2, 1]
                # print(f"alloc: {alloc}") # [(2, 2), (1, 1), (2, 2), (1, 1)]
                parallel_config = [item for sublist in alloc for item in sublist]
                # print(f"parallel_config: {parallel_config}") # [2, 2, 1, 1, 2, 2, 1, 1]
                # If the parallel_config is NULL, we return a huge value and continue searching
                if len(parallel_config) == 0:
                    cost_list.append([1e9, alloc])
                    continue
                # Obtain computation cost and memory cost for each stage
                stage_costs, mem_costs, pp_layer_list =  compute_costs(parallel_config, bias_value, bsz)
                # Obtain communication cost for each stage
                comm_costs = tp_communication_costs(pp_layer_list, parallel_config, bsz)
                # Obtain intermediate communication cost between devices: poor bandwidth condition
                inter_device_comm_cost = inter_device_communication_cost(bsz)
                # Predicted cost for this sub_group plan
                cost = predict_cost(alloc, comp_abilities[sub_group_index], comm_abilities[sub_group_index], stage_costs, comm_costs, inter_device_comm_cost)
                # Form memory limit for each stage: different devices have different memory limit
                gpu_limits = list(np.array(generate_gpu_memory_limits(alloc, gpu_mem_limit_list[sub_group_index])))
                # If memory is over limit, make cost overflow
                cost += compare_arrays_and_calculate_cost(mem_costs, gpu_limits)
                cost_list.append([cost, alloc, mem_costs, pp_layer_list, bsz])
            min_item = min(cost_list, key=lambda x: x[0])
            sub_group_cost = min_item[0]
            # Select the best sub_group plan
            sub_group_plan = min_item[1]
            sub_group_mem = min_item[2]
            sub_group_pp_partition = min_item[3]
            sub_batch = min_item[4]
            sub_group_cost_list.append(sub_group_cost)
            sub_group_plan_list.append([item for sublist in sub_group_plan for item in sublist])
            sub_group_mem_list.append(sub_group_mem)
            sub_group_pp_partition_list.append(sub_group_pp_partition)
            sub_batch_list.append(sub_batch)
        group_cost_list.append(sub_group_cost_list)
        group_plan_list.append(sub_group_plan_list)
        group_mem_list.append(sub_group_mem_list)
        group_pp_partition_list.append(sub_group_pp_partition_list)
        group_batch_list.append(sub_batch_list)

    # print(f"group_pp_partition_list: {group_pp_partition_list}")

    # Calculate fitness
    summation_group_cost = sum(sum(inner_list) for inner_list in group_cost_list)
    
    # if summation_group_cost > 1e9:
    #     # Memory overflow, return high penalty
    #     # 기존 코드는 왜 group_plan_list 를 반환할까? -> format 이 일정하지 않은 문제 발생;; 그래서 고쳤다.
    #     return 1e9, individual.genes
    
    # 원본 hexgen genetic algorithm logic 복구
    set_interval_for_global_search = 10
    if individual.iter % set_interval_for_global_search == 0 and individual.iter >= 100:
        flattened_plan_list = [sublist for inner_list in group_plan_list for sublist in inner_list]
        simulator = Simulator(flattened_plan_list, individual.bias, individual.batch, slo=0.05) # In current impl, slo can be of any value.
        goodput = simulator.exec()
        # print(goodput)
        individual.goodput_store = goodput
        fitness = 1 / goodput
        print(f"Goodput of Simulator : {goodput:10.10f} / Fitness : {fitness:10.10f}")
    else:
        # # Local optimal strategy search
        # fitness =  1 / sum(len(inner_list) for inner_list in individual.genes)
        flattened_cost_list = [sublist for inner_list in group_cost_list for sublist in inner_list] 
        fitness = 1 / (sum(flattened_cost_list) / len(flattened_cost_list))
    return fitness, individual.genes, group_pp_partition_list # Return the average cost as fitness

def is_group_valid(group, gpu_mem_list):
    """Check if the given group is valid based on the multiplication criterion"""
    MULTIPLIER_ARRAY = gpu_mem_list
    VALID_THRESHOLD = 144
    total = sum(a*b for a, b in zip(group, MULTIPLIER_ARRAY))
    return total >= VALID_THRESHOLD

def mutate(individual):
    """Original mutation logic from genetic_algorithm.py"""
    for sub_group_index in range(len(individual.genes)):
        genes = copy.deepcopy(individual.genes[sub_group_index])
        
        initial_array_ = initial_array[sub_group_index]
        restriction_array_ = restriction_array[sub_group_index]
        gpu_mem_limit_list_ = gpu_mem_limit_list[sub_group_index]
        comp_abilities_ = comp_abilities[sub_group_index]
        
        # Implementing Hill Climbing to guide mutation process
        max_attempts = 1
        attempts = 0
        pre_cost = cost_function(initial_array_, restriction_array_, genes, comp_abilities_, gpu_mem_limit_list_)
        
        while attempts < max_attempts:
            # Start mutation for genes
            valid_mutation = False
            validate_and_adjust(genes, initial_array_)
            
            while not valid_mutation:
                mutation_type = random.choice(["adjust_composition", "adjust_groups"])
                
                if mutation_type == "adjust_composition" and len(genes) > 1:
                    group_idx1, group_idx2 = random.sample(range(len(genes)), 2)
                    group1 = genes[group_idx1]
                    group2 = genes[group_idx2]
                    
                    # Get the dominant type (dimension) in group2
                    dominant_dim_group2 = group2.index(max(group2))
                    
                    # Check if group1 has any of that dominant type to give
                    if group1[dominant_dim_group2] > 0:
                        dim = dominant_dim_group2
                    else:
                        # If not, just pick any type from group1 that it can give
                        valid_dims = [i for i, v in enumerate(group1) if v > 0]
                        if not valid_dims:
                            continue
                        dim = random.choice(valid_dims)
                    
                    # Transfer a unit of the chosen type from group1 to group2
                    group1[dim] -= 1
                    group2[dim] += 1
                    
                    if is_group_valid(group1, gpu_mem_limit_list_) and is_group_valid(group2, gpu_mem_limit_list_):
                        valid_mutation = True
                    else:
                        # Revert the change
                        group1[dim] += 1
                        group2[dim] -= 1
                
                elif mutation_type == "adjust_groups":
                    if len(genes) > 1 and random.random() < 0.001:
                        # Merge two groups
                        idx1, idx2 = random.sample(range(len(genes)), 2)
                        genes[idx1] = [x + y for x, y in zip(genes[idx1], genes[idx2])]
                        del genes[idx2]
                        valid_mutation = True
                    else:
                        # Split a group
                        if len(genes) > 1:
                            idx = random.randrange(len(genes))
                        else:
                            idx = 0
                            valid_mutation = True
                        group = genes[idx]

                        # new_group = [x // 2 for x in group]
                        # if is_group_valid(new_group, gpu_mem_limit_list_):
                        #     genes[idx] = [x - x // 2 for x in group]
                        #     genes.insert(idx + 1, new_group)
                        #     valid_mutation = True

                        # 위 방법 실제 HEXGEN 의 알고리즘이다.
                        # 변경하는 이유는 만약 애초에 초기 노드 그룹이 모든 Node 가 단일한 GPU 를 채택한다면
                        # 절대 그룹을 분할할 수 없는 상황이 오기 때문이다.
                        # 이 경우를 해결하기 위해서, Single GPU Node 가 1개만 존재하는 경우 
                        # 그룹 분할을 // 2 를 통해 하는 것이 아니라 random 하게 포함 여부를 선택하게 한다.
                        new_group = []
                        for node_count in group:
                            # 2개 이상인 경우 기존 방식을 채택한다.
                            if node_count > 1:
                                new_group.append(node_count // 2)
                            elif 0 <= node_count <= 1: # 0 일 경우도 아래의 로직을 통해 함께 처리될 수 있다.
                                # 50% 확률을 통해 random 으로 넣을지 말지 결정
                                if random.random() < 0.5:
                                    new_group.append(node_count)
                                else:
                                    new_group.append(0)
                            else: # 이 경우 node_count 가 음수가 되었다는 얘기인데 무언가 문제가 발생했다는 것이다.
                                raise ValueError(f"Node count is negative. group:{group}, new_group:{new_group}")
                        
                        # 기존에는 업데이트시에 다시 계산했지만, 이미 생성된 group 을 통해서 기존 노드 수를 감소시키는 것으로 대체한다.
                        parent_group = [x - y for x, y in zip(group, new_group)]
                        # 이 경우 사실 기존에 쪼개져버린 원본 그룹도 체크를 해주어야 한다. 원본 HEXGEN 코드에 버그가 있던 것.
                        if is_group_valid(new_group, gpu_mem_limit_list_) and is_group_valid(parent_group, gpu_mem_limit_list_):
                            genes[idx] = parent_group
                            genes.insert(idx + 1, new_group)
                            valid_mutation = True

                validate_and_adjust(genes, initial_array_)
                
                # Ensure no sub-array is all zeros
                valid_mutation = valid_mutation and all(any(value > 0 for value in group) for group in genes)
            
            # Check if mutation improved cost
            post_cost = cost_function(initial_array_, restriction_array_, genes, comp_abilities_, gpu_mem_limit_list_)
            
            if post_cost < pre_cost:
                break
            attempts += 1
        
        # Finish mutation for genes
        individual.genes[sub_group_index] = genes
    
    # Start mutation for bias
    bias = copy.deepcopy(individual.bias)
    batch = copy.deepcopy(individual.batch)
    
    if random.random() < 0.5:
        for i in range(len(individual.bias)):
            # Determine the extent of bias change
            bias_change = random.choice([-1, 1])
            bias[i] += bias_change
    
    # Finish mutation for bias and batch
    individual.bias = bias
    individual.batch = batch
    
    return (individual,)

def validate_individual(individual):
    """Validation function from original genetic_algorithm.py"""
    genes = individual.genes
    for sub_group_index in range(len(genes)):
        initial_array_ = initial_array[sub_group_index]
        genes_ = genes[sub_group_index]
        sums = [0] * len(initial_array_)
        for sub_array in genes_:
            for i in range(len(sub_array)):
                sums[i] += sub_array[i]
        
        for sub_array in genes_:
            if all(element == 0 for element in sub_array):
                return False
    return sums == initial_array_

def selValidTournament(individuals, k, tournsize):
    """Custom selection function from original genetic_algorithm.py"""
    chosen = []
    for i in range(k):
        aspirants = tools.selRandom(individuals, tournsize)
        valid_aspirants = [ind for ind in aspirants if validate_individual(ind)]
        if valid_aspirants:
            chosen.append(max(valid_aspirants, key=attrgetter('fitness')))
        else:
            chosen.append(tools.selRandom(individuals, 1)[0])
    return chosen

def run_hexgen_ga(config, population_size=50, generations=100, verbose=True):
    """Run HEXGEN genetic algorithm with custom configuration"""
    
    # Set the cluster configuration
    set_cluster_config(config)
    
    if verbose:
        print("=" * 80)
        print("Running HEXGEN Genetic Algorithm (Original Functions)")
        print("=" * 80)
        print(f"Population size: {population_size}")
        print(f"Generations: {generations}")
        print(f"Cluster config: {initial_array}")
        print()
    
    # Create toolbox (exactly like original)
    toolbox = base.Toolbox()
    toolbox.register("individual", initialize_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", selValidTournament, k=10, tournsize=3)
    
    # Create initial population
    population = toolbox.population(n=population_size)
    
    # Parameters from original
    cxpb = 0.0  # Probability of mating two individuals
    mutpb = 1.0  # Probability of mutating an individual
    
    # Setup statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else []) + ['plan']
    
    # Track best individual across all generations
    best_individual = None
    best_fitness = float('inf')
    best_gen_iteration = 0
    best_gen_pp_partition = None
    
    # Run genetic algorithm (exactly like original)
    for gen in range(generations):
        offspring = algorithms.varOr(population, toolbox, lambda_=10, cxpb=cxpb, mutpb=mutpb)
        
        eval_data = list(map(toolbox.evaluate, offspring))
        for ind, data in zip(offspring, eval_data):
            fit, plan, pp_partition = data
            ind.fitness.values = fit,
            ind.plan = [fit, plan]
            ind.iter = gen + 1
            
            # Track best individual
            if fit < best_fitness:
                best_fitness = fit
                best_individual = copy.deepcopy(ind)
                best_gen_iteration = ind.iter
                best_gen_pp_partition = pp_partition

        population = toolbox.select(offspring, k=3)
        
        # Record the plan
        plans = [ind.plan for ind in population]
        min_pair = min(plans, key=lambda pair: pair[0])
        plan = min_pair[1]
        record = stats.compile(population) if stats else {}
        record['plan'] = plan
        logbook.record(gen=gen, nevals=len(offspring), **record)
        
        # Print every generation like original
        print(logbook.stream)
    
    # Extract final results (exactly like original)
    gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
    
    print("\n" + "=" * 80)
    print(f"Optimization Complete! Final Min: {min_[-1]:.6f}")
    print(f"Best Iteration of Generation : {best_gen_iteration}")
    print(f"PP Partition list : {best_gen_pp_partition}")
    print("=" * 80)
    
    # Return statistics and best individual
    return gen, avg, min_, max_, best_individual, best_gen_pp_partition


def main():
    # Test 1: Single group (all nodes in one pipeline)
    print("\n" + "=" * 80)
    print("Test 1: Single Group Configuration")
    print("=" * 80)
    
    # single_model_config = {
    #     'initial_array': [[8, 4, 8, 4]],  # All nodes in one group
    #     'gpu_mem_limit_list': [[40, 24, 40, 24]],
    #     'comp_abilities': [[1.0, 0.75, 1.0, 0.75]],
    #     'comm_abilities': [[1.0, 1.0, 1.0, 1.0]]
    # }

    cluster = [
        {
            "instance_type": "g6e.xlarge",
            "num_instances": 3,
            "num_gpu_per_instance": 1,
            "memory_limit": 44,
            "computation_ability": 1.0,
            "communication_ability": 1.0
        },
        {
            "instance_type": "g6e.12xlarge",
            "num_instances": 1,
            "num_gpu_per_instance": 4,
            "memory_limit": 44,
            "computation_ability": 1.0,
            "communication_ability": 1.0
        },
        {
            "instance_type": "g5.12xlarge",
            "num_instances": 3,
            "num_gpu_per_instance": 4,
            "memory_limit": 22,
            "computation_ability": 0.5,
            "communication_ability": 1.0
        }
    ]

    cluster_config_flatten = []

    single_model_config = {
        "initial_array": [[]],
        "gpu_mem_limit_list": [[]],
        "comp_abilities": [[]],
        "comm_abilities": [[]],
    }

    for node_info in cluster:
        node_count = node_info["num_instances"]
        num_gpu = node_info["num_gpu_per_instance"]
        memory_limit = node_info["memory_limit"]
        comp_ability = node_info["computation_ability"]
        comm_ability = node_info["communication_ability"]
        for i in range(node_count):
            single_model_config["initial_array"][0].append(num_gpu)
            single_model_config["gpu_mem_limit_list"][0].append(memory_limit)
            single_model_config["comp_abilities"][0].append(comp_ability)
            single_model_config["comm_abilities"][0].append(comm_ability)

            cluster_config_flatten.append(node_info["instance_type"])

    result1 = run_hexgen_ga(
        single_model_config,
        population_size=100,
        generations=300
    )
    gen1, avg1, min1, max1, best_ind1, best_gen_pp_partition = result1

    # Display best individual from Test 1
    print("\n" + "=" * 80)
    print("Best Individual from Single Group:")
    print("=" * 80)
    print(f"  Fitness:   {best_ind1.fitness.values[0]:.6f}")
    print(f"  Genes:     {best_ind1.genes}")
    print(f"  PP layers: {best_gen_pp_partition}")
    print(f"  Bias:      {best_ind1.bias}")
    print(f"  Batch:     {best_ind1.batch}")

    for i, node_group in enumerate(best_ind1.genes[0]):
        group_config = {} # instance_type : [PP_strategy]
        for j, node_count in enumerate(node_group):
            if node_count == 0:
                continue
            elif node_count < 0:
                raise ValueError("Node Count of Final gene < 0")
            instance_type = cluster_config_flatten[j]
            if not group_config.get(instance_type):
                group_config[instance_type] = []

            group_config[instance_type].append(node_count)

        print(f"Pipeline {i + 1}")
        for instance_type, PP_strategy in group_config.items():
            print(f"  {instance_type}: {PP_strategy}")
        print(f"  PP Layers: {best_gen_pp_partition[0][i]}")
        print(f"  Bias: {best_ind1.bias[i]}")

if __name__ == "__main__":
    # Check for deap
    try:
        import deap
    except ImportError:
        print("Installing deap...")
        os.system("pip install deap")
        print("Please run again after installation.")
        sys.exit(1)
    
    main()