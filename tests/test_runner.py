# tests/test_runner.py
import torch
import pytest
from model import LinearModel

def test_linear_forward():
    model = LinearModel()
    model.eval()  # ensure BatchNorm works with batch size = 1
    input_tensor = torch.rand(1, 17 * 2)  # 17 2D joints as input
    output = model(input_tensor)

    assert output.shape == (1, 17 * 3)  # 17 3D joints as output
