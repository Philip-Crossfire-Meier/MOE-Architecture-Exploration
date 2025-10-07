import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.app_config import AppConfig


def shannon_equiprobability(data: list) -> float:
    """
    Calculate the Shannon entropy of a set of probabilities.
    params:
        data (list): A list of probabilities.
    raises:
        ValueError: If probabilities are not non-negative or do not sum to 1.
    returns:
        float: The Shannon entropy in the range [0, log2(len(probabilities))].
    """
    if any(p < 0 for p in data) or sum(data) != 1:
        raise ValueError("Probabilities must be non-negative and sum to 1.")

    return - sum(p * np.log2(p) for p in data if p > 0)

def expert_activation_heatmap(data: torch.Tensor, output_path: str) -> None:
    """
    Generates a heatmap of expert activations in relation to output classes.
    params:
        data (torch.Tensor): A 2D tensor of expert activations per class label.
    raises:
        ValueError: If data is not a 2D tensor.
    returns:
        None
    """
    if data.dim() != 2:
        raise ValueError("Data must be a 2D tensor.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Expert Activation Heatmap")
    plt.xlabel("Classes")
    plt.ylabel("Experts")
    plt.show()
    plt.savefig(AppConfig.charts_path + "/expert_activation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def expert_activation_distribution_over_time(data: torch.Tensor, output_path: str) -> None:
    """
    Generates a line plot of expert activations over time.
    params:
        data (torch.Tensor): A 2D tensor of expert activations with time as one dimension.
    raises:
        ValueError: If data is not a 2D tensor.
    returns:
        None
    """
    if data.dim() != 2:
        raise ValueError("Data must be a 2D tensor.")

    plt.figure(figsize=(12, 6))
    for i in range(data.size(0)):
        plt.plot(data[i].cpu().numpy(), label=f"Expert {i}")
    plt.title("Expert Activation Distribution Over Time")
    plt.xlabel("Time")
    plt.ylabel("Activation")
    plt.legend()
    plt.show()
    plt.savefig(AppConfig.charts_path + "/expert_activation_distribution_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()

def expert_activation_vs_accuracy(data: torch.Tensor, accuracy: torch.Tensor, output_path: str) -> None:
    """
    Generates a scatter plot of expert activations against accuracy.
    params:
        data (torch.Tensor): A 2D tensor of expert activations.
        accuracy (torch.Tensor): A 1D tensor of accuracy values.
    raises:
        ValueError: If data is not a 2D tensor or accuracy is not a 1D tensor.
    returns:
        None
    """
    if data.dim() != 2 or accuracy.dim() != 1:
        raise ValueError("Data must be a 2D tensor and accuracy must be a 1D tensor.")

    plt.figure(figsize=(10, 6))
    for i in range(data.size(0)):
        plt.scatter(data[i].cpu().numpy(), accuracy.cpu().numpy(), label=f"Expert {i}")
    plt.title("Expert Activation vs Accuracy")
    plt.xlabel("Expert Activation")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.savefig(AppConfig.charts_path + "/expert_activation_vs_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()