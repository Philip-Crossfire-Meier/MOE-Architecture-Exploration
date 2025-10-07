from src.analyze import shannon_equiprobability
import pytest
import numpy as np

def test_shannon_equiprobability_valid():
    # Test with valid probabilities
    data = [0.2, 0.3, 0.5]
    result = shannon_equiprobability(data)
    assert result >= 0 and result <= np.log2(len(data))

def test_shannon_equiprobability_zero():
    # Test with zero probabilities
    data = [0.0, 1.0]
    result = shannon_equiprobability(data)
    assert result == 0.0

def test_shannon_equiprobability_invalid():
    # Test with invalid probabilities (negative)
    data = [-0.1, 1.1]
    with pytest.raises(ValueError):
        shannon_equiprobability(data)

def test_shannon_equiprobability_empty():
    # Test with invalid probabilities (do not sum to 1)
    data = [0.5, 0.6]
    with pytest.raises(ValueError):
        shannon_equiprobability(data)

def test_shannon_equiprobability_with_softmax():
    # Test with softmax probabilities
    data = [2.0, 1.0, 0.1]
    softmax_data = np.exp(data) / np.sum(np.exp(data))
    result = shannon_equiprobability(softmax_data)
    assert result >= 0 and result <= np.log2(len(softmax_data))