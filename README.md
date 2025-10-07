# MOE Architecture Exploration Research Project
This is the source code for the MOE Architecture Exploration Research Project.

## Requirements
Python with the libraries listed in requirements.txt

## Usage
Call the main module with the desired experiment config like so:
```
python -m src.main --config <config>
```
Some experiment configs can be found in the src/config folder. When first launching the experiment runner, datasets will be downloaded from the PyTorch data repository into the /data folder.

To run multiple training runs, modify and use the run_iterations.sh script.
