from .config_base import ConfigBase


class Config(ConfigBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.moe_type = "noisytopk"
        self.num_experts = 8
        self.topk_experts = 2
        self.batch_size = 2
        self.expert_scaling_factor = 0.03125
        self.dataset = "cinic10"