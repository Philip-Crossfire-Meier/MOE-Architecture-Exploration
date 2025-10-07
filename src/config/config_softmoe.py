import torch

from .config_base import ConfigBase


class Config(ConfigBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.moe_type = "softmoe"
        self.learning_rate = 0.0001
        self.batch_size = 2
        self.expert_scaling_factor = 0.03125
