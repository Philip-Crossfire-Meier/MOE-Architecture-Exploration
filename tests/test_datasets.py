import os
import pytest
import torch
import src.datasets as datasets
import src.config.literals as lt
from typing import get_args

@pytest.mark.parametrize("dataset_name", get_args(lt.DATASET_TYPES))
def test_load_dataset(dataset_name):
    # Test if dataset can be loaded
    dir = os.path.join(os.path.dirname(__file__), "../data", dataset_name)
    dataset = datasets.load_dataset(dir, dataset=dataset_name)
    assert dataset is not None
    assert isinstance(dataset, tuple)
    assert len(dataset) == 2
    assert isinstance(dataset[0], tuple) and isinstance(dataset[1], tuple)
    assert len(dataset[0]) == 2 and len(dataset[1]) == 2
    assert isinstance(dataset[0][0], torch.Tensor) and isinstance(dataset[0][1], torch.Tensor)
    assert isinstance(dataset[1][0], torch.Tensor) and isinstance(dataset[1][1], torch.Tensor)

def test_transform_test_dataset():
    # Test if images are transformed
    dir = os.path.join(os.path.dirname(__file__), "../data", 'mnist')
    dataset = datasets.load_dataset(dir, dataset='mnist')
    X_train, _ = dataset[0]
    transformed_data = datasets.transform_test_dataset()
    assert transformed_data is not None
    assert isinstance(transformed_data, torch.Tensor)
    assert transformed_data.shape[1] == X_train.shape[1]  # Same flattened dimension
    assert transformed_data.size(0) == X_train.size(0)  # Same number of samples
    assert transformed_data.dtype == torch.float  # Ensure the data type is float
   