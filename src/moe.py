import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .app_config import AppConfig, AppConfigDict
from .ffn import FFN


class MoE(nn.Module):
    """
    Mixture of Experts (MoE) model that can use multiple MoE architectures to process input data. 
    This class acts as the gate/router.
    """

    def __init__(self, config: AppConfigDict, input_size: (int | None) = None, output_size: (int | None) = None):
        """
        Initialize the MoE layer.
        Args:
            config (AppConfigDict): Configuration containing MoE settings.
            input_size (int): Size of the input features.
            output_size (int): Size of the output features.
        Raises:
            ValueError: If input_size or config is not provided or is invalid.
        """
        super(MoE, self).__init__()
        if config is None:
            raise ValueError("Config is required.")
        if input_size is None or input_size <= 1 or type(input_size) is not int:
            raise ValueError("Input size must be an integer greater than 1.")

        self.config = config
        self.input_size = input_size
        self.stats = dict(
            dropped_tokens=0
        )

        if self.config.moe_type == 'exploding_tree':
            old_num_experts = self.config.num_experts
            self.config.num_experts = 1

        if self.config.moe_type == "softmoe": #softmoe specific
            self.phi = nn.Parameter(torch.empty((input_size, self.config.num_experts, self.config.slots_per_expert), device=self.config.device)) # (input_size, num_experts, slots_per_expert)
            self.phi.data.uniform_(-1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
            self.phi.requires_grad = True
            nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5)) 
            self.phi_scale = nn.Parameter(torch.tensor(0.5))
        else:
            self.noise = nn.Linear(input_size, self.config.num_experts, device=self.config.device, bias=False)
            self.gate = nn.Linear(input_size, self.config.num_experts, device=self.config.device, bias=False)
            nn.init.normal_(self.gate.weight, mean=0.0, std=0.1)
            #nn.init.constant_(self.gate.bias, 0.0)
            nn.init.normal_(self.noise.weight, mean=0.0, std=0.1)
            #nn.init.constant_(self.noise.bias, 0.0)

        if self.config.moe_type != 'exploding_tree':
            self.experts = nn.ModuleList([FFN(config=self.config, expert_id=i, input_size=input_size) for i in range(self.config.num_experts)])
        else:
            self.expert_pool = nn.ModuleList([FFN(config=self.config, expert_id=i, input_size=input_size) for i in range(old_num_experts)])
            self.experts = [self.expert_pool[random.randint(0, old_num_experts - 1)]]
            self.last_epoch_switch = [1,2]
            
        if "min_activated_experts" in self.config.__dict__:
            if self.config.min_activated_experts > 0:
                self.always_on_experts = nn.ModuleList(
                    [FFN(config=self.config, expert_id=i, input_size=input_size) for i in range(self.config.min_activated_experts)]
                )

    def check_is_tensor(func):
        """
        Decorator which checks if the first argument is a torch.Tensor and has the correct shape.
        Raises:
            TypeError: If the first argument is not a torch.Tensor.
            ValueError: If the tensor does not have the correct shape.
        """
        def wrapper(self, *args, **kw):
            x = args[0]
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Expected input to be a torch.Tensor, got {type(x)} instead.")
            if x.dim() != 2 or x.size(1) != self.input_size:
                raise ValueError(f"Input tensor must be of shape (batch_size, {self.input_size}). Got {x.shape} instead.")
            return func(self, *args, **kw)
        return wrapper
    
    def switch_transformer_auxiliary_loss(self) -> torch.Tensor:
        """
        Compute the auxiliary loss for the Switch Transformer.
        Returns:
            torch.Tensor: Auxiliary loss value.
        References:
            - Fedus et al. (2022), specifically the pseudo code appendix.
            - https://github.com/huggingface/transformers/blob/main/src/transformers/models/switch_transformers/modeling_switch_transformers.py
        """

        f = self.expert_mask.mean(dim=0)  # density_1 in paper
        p = self.last_gate_probs.mean(dim=0)  # density_2 in paper

        # Scaling
        num_experts = self.config.num_experts
        aux_loss = torch.sum(f * p) * (num_experts ** 2)
        
        return aux_loss * self.config.alpha

    @check_is_tensor
    def noisy_topk(self, x: torch.Tensor, type: (str | None) = "softplus") -> torch.Tensor:
        """
        Compute noisy top-k gate scores using softplus activation.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            type (str): Type of noise to add. Options are "softplus" or "gumbel".
        Returns:
            torch.Tensor: Noisy top-k or gumbel gate scores of shape (batch_size, num_experts).
        Raises:
            ValueError: If the type is not "softplus" or "gumbel".
            ValueError: If the input tensor does not have the correct shape.
        References:
            - Fedus et al. (2022) for softplus noise.
            - Zhou et al. (2022) for gumbel noise.
        """
        if type == "softplus":
            noise_scale = self.config.noise_scale
            p = self.gate(x) + (noise_scale * torch.randn(x.shape[0], self.config.num_experts, device=self.config.device) * F.softplus(self.noise(x)))
        elif type == "gumbel":
            temperature = 0.5
            p = self.gate(x) + (torch.randn(x.shape[0], self.config.num_experts, device=self.config.device) * F.gumbel_softmax(self.noise(x), tau=temperature, hard=True))
        else:
            raise ValueError("Invalid type for noisy top-k gate scores. Use 'softplus' or 'gumbel'.")
        return p

    @check_is_tensor
    def forward(self, x: torch.Tensor, epoch = 0, max_epochs = 0) -> torch.Tensor:
        """
        Forward pass for the Mixture of Experts (MoE) model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        Raises:
            ValueError: If the input tensor does not have the correct shape.
        """
        
        if self.config.moe_type == "softmoe":
            self.last_gate_probs = nn.Parameter(torch.zeros(self.config.num_experts, device=self.config.device)).detach()
            return self._forward_softmoe(x)
        elif self.config.moe_type == "switch_transformer":
            self.last_gate_probs = nn.Parameter(torch.zeros(self.config.num_experts, device=self.config.device)).detach()
            return self._forward_switch_transformer(x)
        elif self.config.moe_type == "expert_choice":
            self.last_gate_probs = nn.Parameter(torch.zeros(self.config.num_experts, device=self.config.device)).detach()
            return self._forward_expert_choice(x)
        elif self.config.moe_type == "roundrobin":
            self.last_gate_probs = nn.Parameter(torch.zeros(self.config.num_experts, device=self.config.device)).detach()
            return self._forward_roundrobin(x)
        elif self.config.moe_type == "ffn":
            self.last_gate_probs = nn.Parameter(torch.zeros(1, device=self.config.device)).detach()
            return self._forward_ffn(x)
        elif self.config.moe_type == "exploding_tree":
            self.last_gate_probs = nn.Parameter(torch.zeros(self.config.num_experts, device=self.config.device)).detach()
            if epoch <= len(self.experts):
                pass
            else:
                if epoch > len(self.expert_pool):
                    self.experts = self.expert_pool
                    self.config.num_experts = len(self.experts)
                elif epoch not in self.last_epoch_switch:
                    for expert in self.expert_pool:
                        if expert.stats["activated_training"] == 0:
                            self.experts = [expert]
                            self.last_epoch_switch = [epoch, epoch+1]
                            break
            return self._forward_topk(x)
        else:
            return self._forward_topk(x)
    
    @check_is_tensor
    def _forward_ffn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FFN expert.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        batch_size = x.size(0)
        expert_outputs = []

        for expert in self.experts:
            expert.stats["activated_this_batch_training"] = 0
            expert.stats["activated_this_batch_eval"] = 0

        for i in range(batch_size):
            expert_out = self.experts[0](x[i].unsqueeze(0))
            self._count_activations(0)
            expert_outputs.append(expert_out)
        weighted_output = torch.cat(expert_outputs, dim=0)
        #output = self.fc_out(weighted_output)   
        return weighted_output

    @check_is_tensor
    def _forward_switch_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for switch transformer gating mechanism.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        
        batch_size = x.size(0)
        expert_outputs = []
        expert_capacity = self.config.expert_capacity_strategy(batch_size)

        # Compute gate scores and weights
        gate_scores = self.noisy_topk(x, type="softplus")  # (batch_size, num_experts)
        gate_probs = F.softmax(gate_scores, dim=1)  # (batch_size, num_experts)
        self.last_gate_probs = gate_probs.detach() # Save so main can push this into aim
        
        expert_index = torch.argmax(gate_probs, dim=1)
        self.expert_mask = F.one_hot(expert_index, num_classes=self.config.num_experts).float()
        
        # Capacity aware expert masking (per expert, across batch)
        expert_tokens_count = torch.zeros(self.config.num_experts, device=self.config.device)
        capacity_mask = torch.ones_like(self.expert_mask)
        
        for i in range(batch_size):
            expert_idx = expert_index[i].item()
            if expert_tokens_count[expert_idx] >= expert_capacity:
                capacity_mask[i, expert_idx] = 0  # Drop this token
                self.stats["dropped_tokens"] += 1
            else:
                expert_tokens_count[expert_idx] += 1
        
        self.expert_mask *= capacity_mask

        for i in range(batch_size):
            expert_idx = expert_index[i].item()

            # Check if this token was dropped due to capacity
            if self.expert_mask[i, expert_idx] == 0:
                # Use residual connection for dropped tokens
                expert_outputs.append(torch.zeros((1, self.input_size), device=self.config.device))
                continue
                
            # Process the expert with gate probability as weight
            expert_out = self.experts[expert_idx](x[i].unsqueeze(0))
            self._count_activations(expert_idx)
            gate_weight = gate_probs[i, expert_idx]
            expert_outputs.append(gate_weight * expert_out)
        
        weighted_output = torch.cat(expert_outputs, dim=0)  # (batch_size, hidden_size)
        #output = self.fc_out(weighted_output)

        return weighted_output

    @check_is_tensor
    def _forward_roundrobin(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Round Robin gating mechanism.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        
        batch_size = x.size(0)
        expert_outputs = []
        
        for expert in self.experts:
            expert.stats["activated_this_batch_training"] = 0
            expert.stats["activated_this_batch_eval"] = 0

        # Round-robin assignment of experts
        for i in range(batch_size):
            if self.training:
                expert_idx = min(range(self.config.num_experts), key=lambda idx: self.experts[idx].stats["activated_training"])
            else:
                expert_idx = min(range(self.config.num_experts), key=lambda idx: self.experts[idx].stats["activated_eval"])
            expert_out = self.experts[expert_idx](x[i].unsqueeze(0))
            self._count_activations(expert_idx)
            expert_outputs.append(expert_out)
        #output = self.fc_out(torch.cat(expert_outputs, dim=0))
        return torch.cat(expert_outputs, dim=0)

    @check_is_tensor
    def _forward_expert_choice(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert Choice gating mechanism.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        Notes (from/adapted from Zhou et al. (2022)):
            k = (n x c) / e
            n is the total number of tokens in the input batch (such as batch size × sequence length)
            c is the capacity factor
            e is the number of experts
            d is the model hidden dimension
        References: Zhou et al. (2022)
        """
        
        batch_size = x.size(0)
        c = 1
        e = self.config.num_experts
        n = x.size(0) * x.size(1)
        k = min(int((n * c) / e), batch_size)  # Ensure k doesn't exceed available tokens
        k = max(k, 1) 
        G = torch.zeros((e, k), device=self.config.device)
        P = torch.zeros((e, k, n), device=self.config.device)
        WGx = self.gate(x)  # (total number of tokens, num_experts)

        # Compute gate scores and weights
        S = nn.functional.softmax(WGx, dim=-1)  # (batch_size, num_experts)
        G, I = torch.topk(S.T, k)  # (num_experts, k), (num_experts, k)
        expert_outputs = [torch.zeros(1, self.input_size, device=self.config.device) for _ in range(batch_size)]

        for expert in self.experts:
            expert.stats["activated_this_batch_training"] = 0
            expert.stats["activated_this_batch_eval"] = 0

        for i, expert in enumerate(self.experts):
            # Process each token this expert chooses (up to k tokens)
            for j in range(k):
                if j < I.size(1): 
                    idx = I[i, j].item()
                    if idx < batch_size:  
                        output = expert(x[idx].unsqueeze(0))  # (1, input_size)
                        expert_outputs[idx] += G[i, j] * output
                        self._count_activations(i)

        output = torch.stack(expert_outputs).squeeze(1)  # (batch_size, input_size)
        return output

    @check_is_tensor
    def _forward_topk(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for models using top-k/top1 (switch transformer) gating mechanism.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        
        batch_size = x.size(0)
        expert_outputs = []
        expert_capacity = self.config.expert_capacity_strategy(batch_size)

        # Compute gate scores and weights
        gate_scores = self.noisy_topk(x, type="softplus")  # (batch_size, num_experts)
        gate_probs = F.softmax(gate_scores, dim=1)  # (batch_size, num_experts)
        self.last_gate_probs = gate_probs.detach() # Save so main can push this into aim
    
        # Get top-k indices and values for each sample
        if len(self.experts) < self.config.topk_experts:
            topk_values, topk_indices = torch.topk(gate_probs, k=len(self.experts), dim=1)  # (batch_size, topk)
        else:
            topk_values, topk_indices = torch.topk(gate_probs, k=self.config.topk_experts, dim=1)  # (batch_size, topk)

        for expert in self.experts:
            expert.stats["activated_this_batch_training"] = 0
            expert.stats["activated_this_batch_eval"] = 0
            expert.stats["capacity_counter"] = 0

        for i in range(batch_size):
            # For each sample, gather top-k expert outputs
            sample_outputs = []
            for j, expert_idx in enumerate(topk_indices[i]):
                
                if self.experts[expert_idx].stats["capacity_counter"] <= self.config.expert_capacity_strategy(self.input_size):
                    self.experts[expert_idx].stats["capacity_counter"] += 1
                    self._count_activations(expert_idx)
                    expert_out = self.experts[expert_idx](x[i].unsqueeze(0))  # (1, hidden_size)
                else:
                    # Skipping the expert if it exceeds capacity is currently the only supported strategy. This might be an interesting area of research in the future.
                    self.stats["dropped_tokens"] += 1
                    expert_out = torch.zeros((1, self.experts[expert_idx].hidden_size), device=self.config.device)
                    continue
                
                sample_outputs.append(topk_values[i, j] * expert_out)
            
            # Sum weighted outputs for this sample
            sample_sum = torch.stack(sample_outputs, dim=0).sum(dim=0)  # (1, hidden_size)
            if "min_activated_experts" in self.config.__dict__ and self.config.min_activated_experts > 0:
                # Add always-on experts
                for idx, expert in enumerate(self.always_on_experts):
                    sample_sum += expert(x[i].unsqueeze(0)) * (1.0 / self.config.min_activated_experts)
                    self._count_activations(idx)

            # Append to expert outputs
            expert_outputs.append(sample_sum)
        
        weighted_output = torch.cat(expert_outputs, dim=0)  # (batch_size, hidden_size)
        #output = self.fc_out(weighted_output)
        
        return weighted_output

    def expert_level_load_balancing(self, batch_size: int, seq_length: int) -> torch.Tensor:
        gate = self.last_gate_probs
        affinity = torch.zeros((self.config.num_experts, self.config.num_experts))
        T = batch_size+seq_length
        f = torch.count_nonzero(gate, dim=0).float() * self.config.num_experts / (self.config.topk_experts * T)
        P = affinity.sum(dim=0) / T
        expert_loss = self.config.alpha * torch.matmul(f,P)
        return expert_loss

    def l2_normalize_linalg_norm(self, x: torch.Tensor, dim: int, eps: (float | None) = 1e-6) -> torch.Tensor:
        """
        L2 normalize the input tensor along the specified dimension.
        Args:
            x (torch.Tensor): Input tensor to normalize.
            dim (int): Dimension along which to normalize.
            eps (float): Some small value to avoid division by zero.
        Returns:
            torch.Tensor: Normalized tensor.
        References:
            - Puigcerver et al. (2024) for numerical stability in MoE models
        """
        norm = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
        return x / (norm + eps)

    @check_is_tensor
    def _forward_softmoe(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Forward pass for the SoftMoE model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        References:
            - https://arxiv.org/abs/2101.03961
            - https://github.com/fkodom/soft-mixture-of-experts/blob/main/soft_mixture_of_experts/soft_moe.py
        """
        batch_size = x.size(0)
        input_size = x.size(1)
        num_experts = self.phi.size(1)
        slots_per_expert = self.phi.size(2)

        # We follow Puigcerver et al. (2024) and do a l2 normalization of phi and x to avoid numerical instability when scaling expert dimensions
        norm_phi = self.l2_normalize_linalg_norm(self.phi, dim=0) * self.phi_scale # we can't change nn.Parameter in place, so we assign it to a new variable norm_phi
        x = self.l2_normalize_linalg_norm(x, dim=1)

        # For each batch, input_size * input_size -> (num_experts, slots_per_expert)
        phi_flat = norm_phi.view(input_size, -1)  # (input_size, num_experts * slots_per_expert)
        logits = torch.matmul(x, phi_flat)  # (batch_size, num_experts * slots_per_expert)
        logits = logits.view(batch_size, num_experts, slots_per_expert)  # (batch_size, num_experts, slots_per_expert)

        D_dispatch_weights = torch.softmax(logits, dim=1) # (batch_size, num_experts, slots_per_expert)

        # Combine weights: flatten last two dims, softmax, then reshape
        logits_flat = logits.view(batch_size, -1)  # (batch_size, num_experts * slots_per_expert)
        C_combined_weights = torch.softmax(logits_flat, dim=-1).view(batch_size, num_experts, slots_per_expert)

        # Prepare input for experts: (batch_size, input_size) -> (batch_size, num_experts, slots_per_expert, input_size)
        x_exp = x.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, input_size)
        D_exp = D_dispatch_weights.unsqueeze(-1)  # (batch_size, num_experts, slots_per_expert, 1)
        expert_inputs = x_exp * D_exp  # (batch_size, num_experts, slots_per_expert, input_size)

        # Merge batch and expert dims for processing through experts
        expert_inputs_reshaped = expert_inputs.view(-1, input_size)  # (batch_size * num_experts * slots_per_expert, input_size)

        # We'll round-robin inputs/slots to experts
        expert_outputs = []
        for expert_idx, expert in enumerate(self.experts):
            for slot in range(slots_per_expert):
                idxs = torch.arange(batch_size) * num_experts * slots_per_expert + expert_idx * slots_per_expert + slot
                expert_in = expert_inputs_reshaped[idxs]
                expert_outputs.append(expert(expert_in))  # (batch_size, hidden_size)

        # Stack and reshape outputs to (batch_size, num_experts, slots_per_expert, hidden_size)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_outputs = expert_outputs.view(batch_size, num_experts, slots_per_expert, -1)

        # Weighted sum over experts and slots
        C_combined_weights_exp = C_combined_weights.unsqueeze(-1)  # (batch_size, num_experts, slots_per_expert, 1)
        weighted_output = (expert_outputs * C_combined_weights_exp).sum(dim=(1, 2))  # (batch_size, hidden_size)

        #output = self.fc_out(weighted_output)  # (batch_size, output_size)

        return weighted_output

    def _count_activations(self, expert_idx: int) -> None:
        """
        Count the number of activations for a specific expert.
        Args:
            expert_idx (int): The index of the expert to count activations for.
        """
        expert = self.experts[expert_idx]
        if self.training:
            expert.stats["activated_this_batch_training"] += 1
            expert.stats["activated_training"] += 1
        else:
            expert.stats["activated_this_batch_eval"] += 1
            expert.stats["activated_eval"] += 1

    def model_size(self) -> float:
        """
        Compute the size of the model in MB.
        Returns:
            float: Size of the model in MB.
        References: 
            https://discuss.pytorch.org/t/finding-model-size/130275
        """
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in self.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / 1024**2
       
        