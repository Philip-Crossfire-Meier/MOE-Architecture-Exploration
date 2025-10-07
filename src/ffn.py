import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .app_config import AppConfig
from .config.config_factory import get_config

Config = get_config(AppConfig.moe_type)

class FFN(nn.Module):
    ''''
    A simple Feed Forward Neural Network (FFN) with configurable input, hidden, and output sizes.
    '''

    def __init__(self, config: dict, expert_id: (int|None) = None, input_size: (int|None) = None):
        super(FFN, self).__init__()
        if input_size is None or input_size <= 1 or type(input_size) is not int:
            raise ValueError("Input size must be an integer greater than 1.")
        if expert_id is not None and (type(expert_id) is not int or expert_id < 0):
            raise ValueError("Expert ID must be a non-negative integer.")
        
        self.config = config
        self.input_size = input_size
        self.expert_id = expert_id

        self.stats = {
            "expert_id": expert_id if expert_id is not None else 0,
            "activated_training": 0,
            "activated_this_batch_training": 0,
            "activated_eval": 0,
            "activated_this_batch_eval": 0
        }
        hidden_layers = []
        hidden_layer_size = int(input_size * self.config.expert_scaling_factor)
        input_layer_size = int(input_size * self.config.expert_scaling_factor)
        input_layer = nn.Linear(input_size, hidden_layer_size, device=self.config.device)
        for i,_ in enumerate(range(self.config.hidden_layers)):
            hidden_layers.append(nn.LayerNorm(hidden_layer_size, device=self.config.device))
            hidden_layers.append(nn.Linear(hidden_layer_size, hidden_layer_size, device=self.config.device))
            hidden_layers.append(nn.Dropout(0.20))
        norm = nn.LayerNorm(hidden_layer_size, device=self.config.device)
        hidden_layers.insert(0, norm)
        head = nn.Linear(hidden_layer_size, input_size, device=self.config.device)
        self.layers = nn.ModuleList([
            input_layer,
            *hidden_layers,
            head
        ])
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FFN.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, moe_output_size).
        Raises:
            TypeError: If the input is not a torch.Tensor.
            ValueError: If the input tensor does not have the correct shape.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a torch.Tensor, got {type(x)} instead.")
        if x.dim() != 2 or x.size(1) != self.input_size:
            raise ValueError(f"Input tensor must be of shape (batch_size, {self.input_size}). Got {x.shape} instead.")

        if self.training:
            self.stats["activated_this_batch_training"] += 1
            self.stats["activated_training"] += 1
        else:
            self.stats["activated_this_batch_eval"] += 1
            self.stats["activated_eval"] += 1
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.LayerNorm):
                x = layer(x)
            else:
                x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
   