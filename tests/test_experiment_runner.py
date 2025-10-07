import pytest
import torch
import src.experiment_runner as experiment_runner
from src.config.config_factory import get_config

@pytest.fixture
def network():
    return experiment_runner.Network(
        input_size=784,
        output_size=10,
        config=get_config("expert_choice")
    )

def test_network_forward_pass(network):
    # Run a single forward pass
    input_tensor = torch.randn(1, 784)
    output = network(input_tensor)
    assert output is not None
    assert output.shape == (1, 10)

def test_network_backward_pass(network):
    # Run a single backward pass
    input_tensor = torch.randn(1, 784)
    output = network(input_tensor)
    loss = output.sum()
    loss.backward()
    assert loss.item() < 0