from .config_base import ConfigBase


class Config(ConfigBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.moe_type = "lory"
        self.num_experts = 8  # FFN is a single expert model
        self.topk_experts = 2  # FFN uses only one "expert", which is actually the entire network
        self.expert_capacity_strategy = None  # FFN does not use expert capacity strategies
        self.expert_capacity_factor = 1.0  # FFN does not scale expert
        self.expert_scaling_factor = 0.03125
        self.learning_rate = 0.0001
        self.alpha = 0.01
        self.batch_size = 64