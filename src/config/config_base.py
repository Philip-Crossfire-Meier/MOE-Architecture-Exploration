from typing import Literal

import torch

import src.config.literals as lt
from src.expert_capacity_strategy import ExpertCapacityStrategy


class ConfigBase():
    
    expert_scaling_factor: float = 1.0
    
    def __init__(self):
        self.moe_type: lt.MOE_TYPES = "softmoe" # The MoE architecture this config belongs to.
        self.device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # Use cuda if available, otherwise cpu.
        self.seed: int = 42 # Random seed for reproducibility
        self.num_epochs: int = 20 # Number of epochs to train the model
        self.hidden_layers: int = 2 # Number of hidden layers in the MoE model
        self.learning_rate: float = 0.0001 # Learning rate for the optimizer
        self.num_experts: int = 8 # Number of experts in the MoE model
        self.batch_size: int = 25 # Batch size for training
        self.topk_experts: int = 8 # Number of top-k experts to use
        self.dataset_split: float = 0.3 # Fraction of data to use for validation
        self.expert_capacity_factor: float = 1.0 # Capacity factor for experts, used as a multiplicative hyperparameter.
        self.slots_per_expert: int = 1 # SoftMoE specific
        self.log_interval: int = 10 # Interval for logging training progress
        self.min_activated_experts: int = 0 # Minimum number of experts that should be activated, only used for expert segmentation.
        self.expert_scaling_factor = 0.03125 # Scales the experts layer relative to the input size.
        self.noise_scale: float = 0.4 # Noise scaling hyperparameter for noise augmented gate weights.
        self.dataset: lt.DATASET_TYPES | None = None # Optional dataset which overrides appconfig
        self.experiment_name: str | None = None # Optional experiment name will be added as a tag to tracked metrics

        self.expert_capacity_strategy: ExpertCapacityStrategy = ExpertCapacityStrategy.NONE # Default to None, can be set to "Fedus2022" or "Zhou2022"
        self.expert_capacity_strategy.__num_experts__ = self.num_experts
        self.expert_capacity_strategy.__expert_capacity_factor__ = self.expert_capacity_factor
        