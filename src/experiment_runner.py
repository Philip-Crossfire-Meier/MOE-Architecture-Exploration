import math
import os
import random
import time
from datetime import datetime

import numpy as np
import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
from aim import Distribution
from muon import MuonWithAuxAdam, SingleDeviceMuon
from torch.optim.lr_scheduler import OneCycleLR
from torch.profiler import ProfilerActivity, profile, record_function

from src import analyze, moe
from src.app_config import AppConfig, AppConfigDict
from src.app_logger import CSVLogger
from src.cls_ui import progress, update_ui
from src.config.config_base import ConfigBase
from src.datasets import load_dataset, transform_test_dataset


class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: ConfigBase):
        """
        Initialize the neural network.
        Args:
            input_size (int): Size of the input layer.
            output_size (int): Size of the output layer.
            config (ConfigBase): Experiment configuration dict.
        Returns:
            None
        """
        super(Network, self).__init__()
        
        self.config = config
        self.input = nn.Linear(input_size, input_size, device=self.config.device)
        self.moe = moe.MoE(input_size=input_size, output_size=input_size, config=config)
        self.head = nn.Linear(input_size, output_size, device=self.config.device, bias=False)
        nn.init.kaiming_uniform_(self.head.weight, a=math.sqrt(5))
        
        self.input.requires_grad = False

    def forward(self, x, epoch=0, max_epochs=0):
        """
        Forward pass through the network.
        Args:
            x (Tensor): The input tensor.
        Returns:
            Tensor: The output tensor from the last network layer.
        """
        x = self.input(x)
        x = self.moe(x, epoch, max_epochs)
        x = self.head(x)
        return x

class ER():
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode

    def _init_run(self):
        """
        Initialize the experiment run.
        Raises:
            ValueError: If the run cannot be initialized.
        Returns:
            None
        """
        # Set random seed for reproducibility
        np.random.seed(self.config.seed)
        if self.config.device == "cpu":
            torch.manual_seed(self.config.seed)

            self.start_time = None
            self.end_time = None
        elif self.config.device == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)

            # Prepare timing
            _ = torch.randn(1000, 1000).cuda() @ torch.randn(1000, 1000).cuda()
            torch.cuda.synchronize()
            
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        elif self.config.device == "mps":
            torch.mps.manual_seed(self.config.seed)

        self.main_text = ""
        self.expert_utilization = ""

        self.step = 1

        params = self.config.__dict__.copy()
        params["device"] = str(self.config.device)
        del params["expert_capacity_strategy"] # aim can't handle enums

        if not self.test_mode:
            self.run["hparams"] = {**params}
            self.run.add_tag(self.config.moe_type)
            self.run.add_tag(hash(self.config))
            self.run.add_tag(AppConfig.dataset)
            if self.config.experiment_name:
                self.run.add_tag(self.config.experiment_name)

        self.epoch_task = progress.add_task(f"Epoch 1/{self.config.num_epochs}", total=self.config.num_epochs)
        self.batch_task = progress.add_task(f"Batch 1/{self.num_batches}", total=(self.num_batches))

        exp_path = AppConfig.results_path + '/' + self.run_id
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        self.exp_path = exp_path

    def run_experiment(self, live: bool, run, run_id, config: ConfigBase, test_mode: bool = False):
        """
        Initialize experiment
        Args:
            live: The live object for reporting to the dashboard.
            run: The run object
            run_id: The run ID
            config (ConfigBase): The experiment config dict.
            test_mode (bool): Whether to run in test mode. This disables all tracking to aim and log files.
        Returns:
            None
        Raises:
            ValueError: If the optimizer selected in the config is not supported.
        """
        self.config = config
        self.run = run
        self.run_id = run_id
        self.test_mode = test_mode

        (X_tr, y_tr), (X_te, y_te) = load_dataset(AppConfig.data_path, dataset=self.config.dataset) # lines x features
        sequence_length = X_tr.size(1) 
        self.num_batches = X_tr.size(0) // self.config.batch_size
        self.num_classes = torch.unique(y_tr).size(0)
        
        self._init_run() 

        # Create the network, optimizer and scheduler
        self.network = Network(input_size=sequence_length, output_size=self.num_classes, config=self.config)
        self.scheduler = None
        #torch.compile(self.network, mode='default', fullgraph=True, dynamic=True)

        # we use L2 regularization/weight decay to prevent overfitting, weights decay values follow Child et al. (2019) https://arxiv.org/pdf/1904.10509
        if AppConfig.optimizer == "muon":
            # Muon optimizer with auxiliary Adam optimizer. Muon only works on 2D weights and should only be used on hidden layers.
            # Reference: https://github.com/KellerJordan/cifar10-airbench/blob/28bff5f5b31e95aa45b5b20e1f48baf1ed98d5f6/airbench94_muon.py#L362
            # Separate parameters by dimension
            hidden_weights_2d = [p for p in self.network.moe.parameters() if p.ndim >= 2]
            hidden_biases_1d = [p for p in self.network.moe.parameters() if p.ndim == 1]
            nonhidden_params = [*self.network.head.parameters(), *self.network.input.parameters()]

            # Create separate optimizers
            param_groups_muon = [
                dict(params=hidden_weights_2d, use_muon=True, lr=0.02, weight_decay=0.01)
            ]
            param_groups_adam = [
                dict(params=hidden_biases_1d + nonhidden_params, lr=self.config.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
            ]
            self.optimizer_muon = SingleDeviceMuon(param_groups_muon)
            self.optimizer_adam = torch.optim.Adam(param_groups_adam, lr=self.config.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
            self.optimizers = [self.optimizer_muon, self.optimizer_adam]
        elif AppConfig.optimizer == "adam":
            self.optimizer_adam = torch.optim.AdamW(self.network.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
            self.optimizers = [self.optimizer_adam]
            self.scheduler = OneCycleLR(self.optimizer_adam, max_lr=self.config.learning_rate, epochs=self.config.num_epochs, steps_per_epoch=self.num_batches) # default: cosine, warmup: 0.3 * total steps
        elif AppConfig.optimizer == "sgd":
            self.optimizer_adam = torch.optim.SGD(self.network.parameters(), lr=self.config.learning_rate, momentum=0.9, weight_decay=1e-2)
            self.optimizers = [self.optimizer_adam]
            self.scheduler = OneCycleLR(self.optimizer_adam, max_lr=self.config.learning_rate, epochs=self.config.num_epochs, steps_per_epoch=self.num_batches) # default: cosine, warmup: 0.3 * total steps
        else:
            raise ValueError(f"Unsupported optimizer or optimizer not set: {AppConfig.optimizer}")
        
        self.network.to(self.config.device)
        torch.autograd.set_detect_anomaly(False) # only needed for debugging, otherwise this can slow down training
        activation_distribution_list = []

        # Profile the model to get MACs and parameters
        macs, params = thop.profile(self.network, inputs=(torch.randn(1, self.network.input.in_features).to(self.config.device),))
        
        if not self.test_mode:
            self.run.track(macs, name='macs', step=self.step, context={"subset":"train"}, epoch=1)
            self.run.track(params, name='params', step=self.step, context={"subset":"train"}, epoch=1)

        # Main loop
        if self.config.device == "cuda":
            self.start_event.record()
        else:
            self.start_time = time.time()
        for idx_epoch in range(1, self.config.num_epochs+1):
            (tloss, terr, expert_utilization, system_usage, loss_error_text) = self._train(idx_epoch, X_tr, y_tr, live)

            for i in range(self.config.num_experts):
                activation_distribution_list.append(self.network.moe.experts[i].stats["activated_training"])

            # Reference for the custom softmax: https://dlsyscourse.org/, specifically https://github.com/dlsyscourse/hw1/blob/main/hw1.ipynb
            e_x = np.exp(activation_distribution_list - np.max(activation_distribution_list))
            shannon_equi = analyze.shannon_equiprobability(e_x / e_x.sum(axis=0))
            if not self.test_mode:
                self.run.track(shannon_equi, name='shannon_equiprobability', step=self.step, context={"subset":"train"}, epoch=idx_epoch)
                self.run.track(np.mean(tloss), name='loss', step=self.step, context={"subset":"train"}, epoch=idx_epoch)
                self.run.track(np.mean(terr), name='error', step=self.step, context={"subset":"train"}, epoch=idx_epoch)
                self.run.track(np.mean(1 - np.array(terr)), name='accuracy', step=self.step, context={"subset":"train"}, epoch=idx_epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            progress.reset(self.batch_task, start=True, total=self.num_batches)
            update_ui(live, expert_utilization, system_usage, loss_error_text)

            self._eval(idx_epoch, X_te, y_te, live)

        if self.config.device == "cuda":
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.end_time = time.time()
            elapsed_time_ms = (self.end_time - self.start_time) * 1000

        if not self.test_mode:
            self.run.track(elapsed_time_ms, name='elapsed_time', step=self.step, context={"subset":"train"}, epoch=1)

    def _train(self, idx_epoch: int, X_tr: torch.Tensor, y_tr: torch.Tensor, live):
        """
        Run the model training.
        Args:
            idx_epoch (int): The current epoch.
            X_tr (torch.Tensor): The input tensor.
            y_tr (torch.Tensor): The true labels.
            live: The live object for reporting to the dashboard.
        Returns:
            None
        """
        self.network.train()
        progress.update(self.epoch_task, advance=1, refresh=True, description=f"Epoch {idx_epoch}/{self.config.num_epochs}")

        # Augment and shuffle training data so the network doesn't learn sequences
        augmented_training_dataset = transform_test_dataset(normalize_only=False) if idx_epoch > 1 else transform_test_dataset(normalize_only=True)
        batch_tuples = list(enumerate(range(0, augmented_training_dataset.size(0), self.config.batch_size)))
        random.shuffle(batch_tuples)

        batch_count = 1
        tloss = []
        terr = []

        for index, _ in batch_tuples:
            batch_input = augmented_training_dataset[index:index+self.config.batch_size].to(dtype=torch.float, device=self.config.device)
            batch_true_output = y_tr[index:index+self.config.batch_size]
            
            output = self.network(batch_input, idx_epoch, self.config.num_epochs)
            probabilities = F.softmax(output, dim=1).detach()
            loss = F.cross_entropy(output, batch_true_output, label_smoothing=0.2, reduction='sum')

            if self.config.moe_type == "switch_transformer":
                aux_loss = self.network.moe.switch_transformer_auxiliary_loss()
                loss += aux_loss
            elif self.config.moe_type == "expert_segmentation":
                aux_loss = self.network.moe.expert_level_load_balancing(batch_size=self.config.batch_size, seq_length=batch_input.size(1))
                loss += aux_loss
            tloss.append(loss.item())
            error = (output.argmax(dim=1) != batch_true_output).float().mean()
            terr.append(error.item())
        
            loss.backward()
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # Clear gradients for the next step

            loss_error_text = f"Epoch [{idx_epoch}/{self.config.num_epochs}], Loss: {loss.item():.4f}, Error: {error.item():.4f}, Batch: {batch_count}/{self.num_batches}, Step: {self.step}"
            system_usage = f"Model Mem. Usage: { self.network.moe.model_size():.2f} MB\nDevice: {self.config.device}\nConfig: {AppConfig.moe_type}\nDataset: {AppConfig.dataset}\nLR: {self.config.learning_rate:.10f}\nDropped Tokens: {self.network.moe.stats['dropped_tokens']}\nBatch Size: {self.config.batch_size}"
            
            if hasattr(self.network.moe, 'always_on_experts'):
                expert_utilization = "Expert Activations\n------------------\n" + "\n".join([f"Expert {i}: {self.network.moe.experts[i].stats['activated_training'] }" for i in range(self.config.num_experts)]) + "\n".join([f"\nShared Expert {i}: {self.network.moe.always_on_experts[i].stats['activated_training']}\n" for i,_ in enumerate(self.network.moe.always_on_experts)])
            else:
                expert_utilization = "Expert Activations\n------------------\n" + "\n".join([f"Expert {i}: {self.network.moe.experts[i].stats['activated_training'] }" for i in range(self.config.num_experts)])
            
            progress.update(self.batch_task, advance=1, refresh=True, description=f"Batch {batch_count}/{self.num_batches}")
            progress.refresh()
            update_ui(live, expert_utilization, system_usage, loss_error_text)
            
            batch_count += 1
            self.step += 1
        
        if not self.test_mode:
            self.run.track(self.network.moe.model_size(), name='model_size', step=self.step, context={ "subset":"train" }, epoch=idx_epoch)
            self.run.track(self.network.moe.stats['dropped_tokens'], name='dropped_tokens', step=self.step, context={ "subset":"train" }, epoch=idx_epoch)
            for i in range(self.config.num_experts):
                self.run.track(self.network.moe.experts[i].stats["activated_training"], name=f'expert_{i}_activated_training', step=self.step, context={"subset":"train"}, epoch=idx_epoch)

        return tloss, terr, expert_utilization, system_usage, loss_error_text

    def _eval(self, idx_epoch: int, X_te: torch.Tensor, y_te: torch.Tensor, live):
        """
        Evaluate the model on the test set.
        Args:
            idx_epoch (int): The current epoch.
            X_te (torch.Tensor): The input features for the test set.
            y_te (torch.Tensor): The true labels for the test set.
            live: The live object for reporting to the dashboard.
        Returns:
            None
        """
        self.network.eval()
        if not self.test_mode:
            self.run.track(self.network.moe.model_size(), name='model_size', step=self.step, context={"subset":"val"}, epoch=idx_epoch)

        with torch.no_grad():
            hit = 0
            loss = []
            for i in range(0, X_te.shape[0]):
                output = self.network(X_te[i].unsqueeze(0))
                loss.append(F.cross_entropy(output, y_te[i].unsqueeze(0)).item())
                smax = F.softmax(output, dim=1)
                predicted = smax.argmax(dim=1).item()
                if predicted == y_te[i]:
                    hit += 1
                else:
                    gate_values = self.network.moe.last_gate_probs.cpu().numpy()
                    loss_error_text = f"Predicted: {predicted}, Actual: {y_te[i]}"
                    update_ui(live, loss_error_text=loss_error_text)
                    if not self.test_mode:
                        self.run.track(predicted, name='predicted', step=self.step, context={"subset":"val"}, epoch=idx_epoch)
                        self.run.track(y_te[i].item(), name='actual', step=self.step, context={"subset":"val"}, epoch=idx_epoch)
                        self.run.track(Distribution(smax.cpu().numpy()), name='predicted_failed', step=self.step, context={"subset":"train"}, epoch=idx_epoch)
                        self.run.track(Distribution(gate_values), name='gate_values', step=self.step, context={"subset":"val"}, epoch=idx_epoch)
                    
                    if not self.test_mode:
                        with CSVLogger(f"failed_predictions_{self.config.moe_type}", self.run_id) as logger:
                            logger.log(
                                f"{self.step},"
                                f"{idx_epoch},"
                                f"{datetime.now()},"
                                f"{i},"
                                f"{predicted},"
                                f"{y_te[i]},"
                                f"{','.join([f'{v:.4f}' for v in gate_values.flatten()])},"
                                f"{','.join([f'{v:.4f}' for v in smax.cpu().numpy().flatten()])}"
                            )

            # We'll do a single profiling run to measure the performance of the model
            if self.config.device == "cuda":
                # CUDA profiling
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=True) as prof:
                    with record_function("model_training"):
                        self.network(X_te[self.config.batch_size-1:self.config.batch_size])  # Use a single batch for profiling
            else:
                # CPU profiling
                with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
                    with record_function("model_training"):
                        self.network(X_te[self.config.batch_size-1:self.config.batch_size])

            key_averages = prof.key_averages()
            profiling_data = {
                'total_cpu_time': sum([item.cpu_time_total for item in key_averages]),
                'memory_usage': sum([item.cpu_memory_usage for item in key_averages])
            }
            if self.config.device == "cuda":
                profiling_data['total_cuda_time'] = sum([item.cuda_time_total for item in key_averages])

            loss_error_text = loss_error_text + f" Total: {i} Accuracy: " + str(hit / X_te.shape[0])
            update_ui(live, loss_error_text=loss_error_text)
            
            if not self.test_mode:
                for key, value in profiling_data.items():
                    self.run.track(value, name=f'profiling_{key}', step=self.step, context={"subset":"val"}, epoch=idx_epoch)
                self.run.track(np.mean(loss), name='loss', step=self.step, context={"subset":"val"}, epoch=idx_epoch)
                self.run.track(hit / X_te.shape[0], name='accuracy', step=self.step, context={"subset":"val"}, epoch=idx_epoch)
                self.run.log_info(loss_error_text)
                for i in range(self.config.num_experts):
                    self.run.track(self.network.moe.experts[i].stats["activated_eval"], name=f'expert_{i}_activated_eval', step=self.step, context={"subset":"val"}, epoch=idx_epoch)
            
        