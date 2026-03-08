"""The model placement policy"""

from alpaserve_lib.placement_policy.base_policy import ModelData, ClusterEnv
from alpaserve_lib.placement_policy.model_parallelism import (
    ModelParallelismILP, ModelParallelismRR,
    ModelParallelismGreedy, ModelParallelismSearch,
    ModelParallelismEqual)
from alpaserve_lib.placement_policy.selective_replication import (
    SelectiveReplicationILP, SelectiveReplicationGreedy,
    SelectiveReplicationUniform, SelectiveReplicationSearch,
    SelectiveReplicationReplacement)
