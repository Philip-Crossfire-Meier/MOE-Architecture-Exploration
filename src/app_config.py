import os
from typing import Literal, TypedDict

import src.config.literals as lt

class AppConfigDict():
    def __init__(self):
        dir = os.path.dirname(__file__)

        self.moe_type: lt.MOE_TYPES = "noisytopk" # Type of MoE to use, e.g., "noisytopk", "topk", "gated". This parameter is used only if the config is not provided via the command line.
        self.dataset: lt.DATASET_TYPES = "mnist" # Dataset to use for training and evaluation
        self.results_path: str = dir + "/../results" # Folder where the results are stored. This is the parent folder for logs, charts and saved_models.
        self.log_path_folder: str = "logs" # Folder where the logs are stored
        self.data_path: str = dir + "/../data" # Folder where the data is stored
        self.charts_path_folder: str = "charts" # Folder where the charts are stored
        self.config_path: str = dir + "/config" # Path to the configuration files
        self.saved_models_path_folder: str = "saved_models" # Folder where the saved models are stored
        self.warmup_steps: int = 1000 # Unused
        self.optimizer: lt.OPTIMIZER_TYPES = "adam" # Which optimizer to use. Adam is AdamW

    def setup_paths(self, run_id: str) -> None:
        """
        Set up the directory structure for a new experiment run.
        Args:
            run_id (str): Unique identifier for the run (same as aimStack run_id), used to create a subdirectory.
        Returns:
            None
        Raises:
            Exception: If there is an error creating the directories.
        """
        try:
            existential_paths = [
                self.results_path,
                self.data_path,
            ]

            for path in existential_paths:
                if os.path.exists(path) is False:
                    print(f"Creating directory: {path}")
                    os.makedirs(path)

            run_results_path = os.path.join(self.results_path, run_id)
            if not os.path.exists(run_results_path):
                os.makedirs(run_results_path)

            folders = [
                self.log_path_folder,
                self.charts_path_folder,
                self.saved_models_path_folder
            ]

            for folder in folders:
                os.makedirs(os.path.join(run_results_path, folder), exist_ok=True)
                print(f"Created directory: {os.path.join(run_results_path, folder)}")

            self.log_path = os.path.join(run_results_path, self.log_path_folder)
            self.charts_path = os.path.join(run_results_path, self.charts_path_folder)
            self.saved_models_path = os.path.join(run_results_path, self.saved_models_path_folder)
        except Exception as e:
            print(f"Error setting up paths: {e}")
            raise

AppConfig = AppConfigDict()