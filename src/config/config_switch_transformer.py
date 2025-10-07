from src.expert_capacity_strategy import ExpertCapacityStrategy

from .config_base import ConfigBase


class Config(ConfigBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.moe_type = "switch_transformer"
        self.topk_experts = 1  # Switch Transformer uses only one expert
        self.alpha = 2 # Was 0.01 in Fedus et al. (2022)
        self.num_experts = 8
        self.expert_capacity_strategy = ExpertCapacityStrategy.FEDUS2022  # Defaults to Zhou for Switch Transformer
        self.batch_size = 2
        self.expert_scaling_factor = 0.03125
        self.noise_scale = 1.5