#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn as nn
import math
from typing import Dict, Callable, Any

# Registry for initialization functions
INIT_REGISTRY: Dict[str, Callable[[torch.Tensor, Dict[str, Any]], None]] = {}

def register_init(name: str):
    """Decorator to register initialization functions"""
    def decorator(func):
        INIT_REGISTRY[name] = func
        return func
    return decorator

# Weight initialization functions
@register_init("xavier_uniform")
def xavier_uniform_init(tensor: torch.Tensor, fan_in, fan_out) -> None:
    """Xavier uniform initialization"""
    std = math.sqrt(6 / (fan_in + fan_out))
    tensor.uniform_(-std, std)

@register_init("kaiming_uniform")
def kaiming_uniform_init(tensor: torch.Tensor, fan_in, fan_out) -> None:
    """Kaiming uniform initialization (originally named xavier_uniform_small)"""
    std = math.sqrt(3 / (fan_in))
    tensor.uniform_(-std, std)

@register_init("xavier_gaussian")
def xavier_gaussian_init(tensor: torch.Tensor, fan_in, fan_out) -> None:
    """Xavier Gaussian initialization using truncated normal"""
    stdv = 1 / math.sqrt(fan_in)
    torch.nn.init.trunc_normal_(tensor, mean=0, std=stdv, a=-0.9, b=0.9)

@register_init("xlstm")
def xlstm_init(tensor: torch.Tensor, fan_in, fan_out) -> None:
    """xLSTM-style initialization"""
    stdv = 2 / math.sqrt(5 * fan_in)
    tensor.normal_(mean=0, std=stdv)

# Matrix A initialization functions
@register_init("small_gaussian")
def small_gaussian_init(tensor: torch.Tensor, fan_in, fan_out) -> None:
    """Small Gaussian initialization"""
    std = 1 / math.sqrt(fan_in + fan_out)
    torch.nn.init.trunc_normal_(tensor, mean=0, std=0.01)

@register_init("negative_exponential_mamba")
def negative_exponential_mamba_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Negative exponential initialization (Mamba-style)"""
    tensor.uniform_(1, 16)
    tensor.data = torch.exp(-tensor.data)

@register_init("negative_exponential")
def negative_exponential_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Negative exponential initialization"""
    tensor.uniform_(0, 8)
    tensor.data = torch.exp(-tensor.data)

@register_init("zero")
def zero_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Zero initialization"""
    tensor.data[:] = 0

@register_init("uniform")
def uniform_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Uniform initialization in [-0.9, 0.9]"""
    tensor.uniform_(-0.9, 0.9)

@register_init("gazillion")
def gazillion_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Extreme uniform initialization (gazillion range)"""
    tensor.uniform_(-90000, 900000)

# Bias initialization functions
@register_init("constant_zero")
def constant_zero_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Constant zero initialization"""
    torch.nn.init.constant_(tensor, 0.0)

@register_init("bias_uniform")
def bias_uniform_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Uniform bias initialization for first row"""
    torch.nn.init.constant_(tensor, 0.0)
    tensor[0].data.uniform_(-0.9, 0.9)

@register_init("bias_linspace")
def bias_linspace_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Linspace bias initialization for first row"""
    torch.nn.init.constant_(tensor, 0.0)
    tensor[0, :] = torch.linspace(3, 6, tensor.shape[1])

@register_init("bias_minus_linspace")
def bias_minus_linspace_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Negative linspace bias initialization for first row"""
    torch.nn.init.constant_(tensor, 0.0)
    tensor[0, :] = -1 * torch.linspace(0, 1, tensor.shape[1])

@register_init("bias_minus_linspace_small")
def bias_minus_linspace_small_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Negative linspace bias initialization for first row"""
    torch.nn.init.constant_(tensor, 0.0)
    tensor[0, :] = -1 * torch.linspace(0, 1, tensor.shape[1])

@register_init("bias_constant_1")
def bias_constant_1_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Constant 1 bias initialization for first row"""
    torch.nn.init.constant_(tensor, 0.0)
    tensor[0, :] = 1

@register_init("bias_constant_minus_1")
def bias_constant_minus_1_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Constant -1 bias initialization for first row"""
    torch.nn.init.constant_(tensor, 0.0)
    tensor[0, :] = -1

@register_init("bias_constant_2")
def bias_constant_2_init(tensor: torch.Tensor, *args, **kwargs) -> None:
    """Constant 2 bias initialization for first row"""
    torch.nn.init.constant_(tensor, 0.0)
    tensor[0, :] = 2

