from cost_model_impl import compute_costs
# Example of using the function:
alloc = [(1, 1, 1, 1), (1, 1), (2,)]
alloc = [(4,), (2,), (2,)]
alloc = [(4, 1, 2, 1), (1, 2, 1), (1, 2, 1)]
alloc = [(1,)]
parallel_config = [item for sublist in alloc for item in sublist]
bias = 0  # Default bias value
bsz = 1   # Default batch size
comp_cost_list, mem_cost_list, pp_layer_list = compute_costs(parallel_config, bias, bsz)

# The input should be a certain plan
print('INOUT:')
print(alloc)
# The output should be two lists
print('OUTPUTS:')
print(comp_cost_list)
print(mem_cost_list)


