import random
from typing import Tuple

import torch
import torchvision.transforms.v2 as T
from torchvision import datasets, transforms

from src.app_config import AppConfig
from src.config.config_factory import get_config

Config = get_config(AppConfig.moe_type)

X_train_cache = None

def transform_test_dataset(normalize_only: bool = True) -> torch.Tensor:
    """
    Return a copy of the training dataset with augmentations applied.
    Args:
        normalize_only (bool): If True, only normalization is applied. If False, a series of augmentations are applied.
    Returns:
        torch.Tensor: Transformed training dataset.
    Raises:
        ValueError: If the training dataset has not been loaded yet.
    """
    global X_train_cache

    if X_train_cache is None:
        raise ValueError("Training dataset has not been loaded yet.")

    X_local = X_train_cache.detach().clone()  # Clone the cached training data to avoid modifying the original
    # Reference: https://www.kaggle.com/code/samuelcortinhas/mnist-cnn-data-augmentation-99-6-accuracy

    if normalize_only:
       # For the first epoch we apply normalization only
        transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        # Reference for augmentations other than RandomRotation and RandomZoomOut: https://github.com/ido-nasi/MaskDetectionML/blob/c0c3519ec6c6be53707b35fac26c3e007307f4cc/code/ResNet18.py
        transform = T.Compose([
            T.RandomRotation(degrees=10, expand=False),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            T.RandomAffine(degrees=0, shear=5),
            T.ElasticTransform(alpha=50.0, sigma=5.0),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.RandomZoomOut(side_range=(1.0, 1.2), fill=0),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    for i in range(X_local.size(0)):
        image = X_local[i]  # (C, H, W)
        try:
            aug_img = transform(image)  # Apply the transformation
            # Ensure the augmented image has the same shape as the original -> some transformations change the image size
            if aug_img.shape == image.shape:
                X_local[i] = aug_img
        except Exception:
            continue    
        
    sequence_length = X_local.size(1) * X_local.size(2) * X_local.size(3)
    X_local = X_local.view(-1, sequence_length)

    return X_local

def load_dataset(dir: str, dataset: str | None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load the dataset based on the configuration in AppConfig.
    Args:
        dir (str): Directory where the dataset will be stored or loaded from.
    Returns:
        Tuple containing training and test datasets, each as a tuple of (features, labels).
    Raises:
        ValueError: If the dataset specified in AppConfig is not supported.
    """
    if dataset is None:
        dataset = AppConfig.dataset
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # Convert grayscale to RGB
        ])
        train_dataset = datasets.MNIST(root=dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=dir, train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = datasets.CIFAR10(root=dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=dir, train=False, download=True, transform=transform)
    elif dataset == "cinic10":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = datasets.ImageFolder(root=dir + '/cinic-10/train', transform=transform)
        val_dataset = datasets.ImageFolder(root=dir + '/cinic-10/valid', transform=transform)
        test_dataset = datasets.ImageFolder(root=dir + '/cinic-10/test', transform=transform)

    X_tr = torch.stack([x[0] for x in train_dataset], dim=0).to(Config.device) # type is floatXX for data and long for labels
    global X_train_cache
    X_train_cache = X_tr.detach().clone().to(Config.device)  # Cache the training data for augmentation in future epochs
    y_tr = torch.tensor([x[1] for x in train_dataset], dtype=torch.long).to(Config.device)

    X_te = torch.stack([x[0] for x in test_dataset], dim=0).to(Config.device)
    y_te = torch.tensor([x[1] for x in test_dataset], dtype=torch.long).to(Config.device)

    # Reshape (actually flatten to be precise) to (num_batches, sequence_length) -> Future improvement could be to use Con2D to keep the spatial structure, potentially add positional encodings
    sequence_length = X_tr.size(1) * X_tr.size(2) * X_tr.size(3)  # For CIFAR-10, this is 32x32x3 = 3072, for MNIST, this is 28x28x3 = 2352
    X_tr = X_tr.view(-1, sequence_length)  
    X_te = X_te.view(-1, sequence_length)

    # Ensure that the training set is a multiple of the batch size
    if X_tr.size(0) % Config.batch_size != 0:
        usable_len = (X_tr.size(0) // Config.batch_size) * Config.batch_size
        X_tr = X_tr[:usable_len]
        y_tr = y_tr[:usable_len]
    
    # Ensure that the test set is a multiple of the batch size
    if X_te.size(0) % Config.batch_size != 0:
        usable_len = (X_te.size(0) // Config.batch_size) * Config.batch_size
        X_te = X_te[:usable_len]
        y_te = y_te[:usable_len]

    return (X_tr, y_tr), (X_te, y_te)

    