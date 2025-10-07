import pytest
import torch
from src.moe import MoE
from src.config.config_factory import get_config

@pytest.fixture
def moe_model():
    config = get_config('switch_transformer')
    config.input_size = 10
    config.output_size = 10
    config.batch_size = 8
    config.device = 'cpu'
    config.expert_scaling_factor = 0.5
    model = MoE(config=config, input_size=config.input_size, output_size=config.output_size)
    return model

def test_check_is_tensor(moe_model):
    # Test if input tensor decorator works
    x = torch.randn(8, 10) 
    output = moe_model(x)  
    assert output is not None
    with pytest.raises(ValueError):
        moe_model(torch.tensor([1, 2, 3]))  
       
def test_switch_transformer_auxiliary_loss(moe_model):
    # Test if auxiliary loss is computed correctly
    x = torch.randn(moe_model.config.batch_size, moe_model.config.input_size)
    output = moe_model(x) 
    assert output.shape == (moe_model.config.batch_size, moe_model.config.output_size)
    aux_loss = moe_model.switch_transformer_auxiliary_loss()
    assert isinstance(aux_loss, torch.Tensor)
    assert aux_loss.dim() == 0  

def test_expert_level_load_balancing(moe_model):
    # Test if expert-level load balancing works
    x = torch.randn(moe_model.config.batch_size, moe_model.config.input_size)
    output = moe_model(x) 
    assert output.shape == (moe_model.config.batch_size, moe_model.config.output_size)
    load_balance_loss = moe_model.expert_level_load_balancing(moe_model.config.batch_size, 1)
    assert load_balance_loss.item() >= 0  

def test_l2_normalize_linalg_norm(moe_model):
    # Test L2 normalization
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    normalized_x = moe_model.l2_normalize_linalg_norm(x, dim=1)  
    expected = torch.tensor([[0.6, 0.8], [0.0, 0.0]])
    assert torch.allclose(normalized_x, expected, atol=1e-6)

def test_count_activations(moe_model):
    # Test counting of expert activations
    initial_count = moe_model.experts[0].stats["activated_training"]
    moe_model._count_activations(0)  
    new_count = moe_model.experts[0].stats["activated_training"]
    assert new_count == initial_count + 1

def test_model_size(moe_model):
    # Test model size calculation (returns size in MB)
    size_mb = moe_model.model_size()
    assert isinstance(size_mb, (int, float))
    assert size_mb > 0  
    assert size_mb < 100  