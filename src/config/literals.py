from typing import Literal

MOE_TYPES = Literal["all", "softmoe", "noisytopk", "expert_choice", "expert_segmentation", "switch_transformer", "round_robin", "ffn", "lory"]
DATASET_TYPES = Literal["mnist", "cifar10", "cinic10"]
OPTIMIZER_TYPES = Literal["adam", "muon", "sgd"]
