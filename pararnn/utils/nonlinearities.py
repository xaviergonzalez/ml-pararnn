#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from functools import partial

import torch
from torch import nn
import torch.nn.functional as F


NONLIN_REGISTRY = {
    'identity': nn.Identity(),
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'silu': nn.SiLU(),
    'gelu': nn.GELU(approximate='tanh'),
    'selu': nn.SELU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'half_sigmoid': lambda x: F.sigmoid(x) * 0.5,
    'exp': torch.exp,
    'softplus': nn.Softplus(),
    'elu': nn.ELU()
}

NONLIN_AND_DERIVATIVE_REGISTRY = {
    'identity': ( lambda x: x, lambda x: torch.ones_like(x) ),
    'relu': ( F.relu, lambda x: torch.where( x > 0, 1, 0) ),
    'leaky_relu': ( partial(F.leaky_relu,negative_slope=0.01), lambda x: torch.where( x > 0, 1, 0.01) ),  # default -slope is 0.01
    'silu': ( F.silu, lambda x: F.sigmoid(x) * (1 + x * (1-F.sigmoid(x))) ),
    'gelu': ( nn.GELU(approximate='tanh'), lambda x: torch.vmap(torch.func.grad(torch.nn.GELU(approximate='tanh')))(torch.ravel(x)).view_as(x) ),
    'selu': ( F.selu, lambda x: torch.vmap(torch.func.grad(F.selu))(torch.ravel(x)).view_as(x) ),
    'tanh': ( F.tanh, lambda x: 1 - F.tanh(x)*F.tanh(x) ),
    'sigmoid': ( F.sigmoid, lambda x: F.sigmoid(x) * (1-F.sigmoid(x)) ),
    'half_sigmoid': ( lambda x: 0.5 * F.sigmoid(x), lambda x: 0.5 * F.sigmoid(x) * (1-F.sigmoid(x)) ),
    'exp': ( torch.exp, torch.exp ),
    'softplus': ( F.softplus, F.sigmoid ),                                                # default beta is 1.0
    'elu': ( partial(F.elu,alpha=1.0), lambda x: torch.where( x > 0, 1, torch.exp(x))  )  # default alpha is 1.0
}

