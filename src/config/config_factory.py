from. config_base import ConfigBase
from .config_expert_choice import Config as ExpertChoiceConfig
from .config_expert_segmentation import Config as ExpertChoiceConfig
from .config_expert_segmentation import Config as ExpertSegmentationConfig
from .config_exploding_tree import Config as ExplodingTreeConfig
from .config_ffn import Config as FFNConfig
from .config_lory import Config as LoryConfig
from .config_noisytopk import Config as NoisyTopKConfig
from .config_roundrobin import Config as RoundRobinConfig
from .config_softmoe import Config as SoftMoEConfig
from .config_switch_transformer import Config as SwitchTransformerConfig
from .literals import MOE_TYPES


def get_config(moe_type: MOE_TYPES) -> ConfigBase:
    """
    Factory function to get the configuration based on the MoE type.
    Args:
        moe_type (MOE_TYPES): The type of MoE configuration to retrieve.
    Raises:
        ValueError: If the moe_type is not recognized.
    Returns:
        ConfigBase: The configuration object for the specified MoE type.
    """
    if moe_type == "all":
        return [NoisyTopKConfig(), ExpertSegmentationConfig(), ExpertChoiceConfig(), SoftMoEConfig(), RoundRobinConfig()]
    if moe_type == "noisytopk":
        return NoisyTopKConfig()
    elif moe_type == "exploding_tree":
        return ExplodingTreeConfig()
    elif moe_type == "expert_segmentation":
        return ExpertSegmentationConfig()
    elif moe_type == "softmoe":
        return SoftMoEConfig()
    elif moe_type == "switch_transformer":
        return SwitchTransformerConfig()
    elif moe_type == "ffn":
        return FFNConfig()
    elif moe_type == "expert_choice":
        return ExpertChoiceConfig()
    elif moe_type == "roundrobin":
        return RoundRobinConfig()
    elif moe_type == "lory":
        return LoryConfig()
    else:
        raise ValueError(f"Unknown MoE type: {moe_type}")