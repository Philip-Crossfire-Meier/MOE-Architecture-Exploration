import argparse
import os
from uuid import uuid4 as UUID

from aim import Run
from rich.live import Live

import src.config.literals as lt
from src.app_config import AppConfig
from src.cls_ui import generate_layout, progress
from src.config.config_factory import get_config
from src.experiment_runner import ER

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    if os.getenv('MOE_TYPE') is not None:
        AppConfig.moe_type = os.getenv('MOE_TYPE')
    else:
        AppConfig.moe_type = args.config

    configs = get_config(args.config)

    er = ER()
    run_id = str(UUID())
    AppConfig.setup_paths(run_id) # Ensure directories are set up -> Don't log anything before this
    if configs.dataset:
        AppConfig.dataset = configs.dataset
    aimrepo = os.getenv('AIMREPO')
    
    if aimrepo == "useworkspace":
        run = Run(experiment=run_id, repo='/workspace')
    #run = Run(experiment=run_id)
    else:
        run = Run(experiment=run_id) 
    
    with Live(generate_layout(progress=progress), refresh_per_second=10) as live:
        if type (configs) is list:
            for config in configs:
                er.run_experiment(live, run, run_id, config)
        else:
            er.run_experiment(live, run, run_id, configs)
