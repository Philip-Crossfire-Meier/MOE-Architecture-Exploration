import pytest
from src.config.config_factory import get_config
from src.config.literals import MOE_TYPES

@pytest.mark.parametrize("moe_type", [
    "softmoe", 
    "noisytopk", 
    "expert_choice", 
    "expert_segmentation", 
    "switch_transformer", 
    "roundrobin", 
    "ffn", 
    "lory"
])
def test_moe_types(moe_type):
    # Test if config for all MoEs can be instantiated
    config = get_config(moe_type)
    assert config.moe_type == moe_type
