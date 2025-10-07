from .config_base import ConfigBase


class Config(ConfigBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.moe_type = "expert_segmentation"
        self.learning_rate = 0.0001
        self.batch_size = 2
        # Expert Segmentation: 2x the number of experts, but each ones size is halfed
        self.num_experts = 8 * 2
        self.expert_scaling_factor = 0.03125 / 2
        # The total number of activated experts per sample is 3 + 1 always on expert
        self.topk_experts = 3
        self.min_activated_experts = 1
        self.alpha = 0.1
        self.noise_scale = 1.5