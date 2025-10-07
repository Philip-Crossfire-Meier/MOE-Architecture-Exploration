import torch
import torchlens
import torchviz
from torchsummary import summary

from src.app_config import AppConfig, AppConfigDict
from src.config.config_factory import get_config
from src.experiment_runner import Network

config = get_config("expert_choice")

network = Network(
    input_size=784,  # Example input size for MNIST
    output_size=10,  # Example output size for MNIST
    config=config
)

# https://github.com/szagoruyko/pytorchviz
graph = torchviz.make_dot(network(torch.randn(1, 784)), params=dict(network.named_parameters()))
graph.render("network_graph", format="png", cleanup=True)

# https://github.com/sksq96/pytorch-summary
#summary(network, (1, 784), device="cpu")

# https://github.com/johnmarktaylor91/torchlens

print(torchlens.log_forward_pass(network, torch.randn(1, 784), layers_to_save='all', vis_opt='rolled'))
