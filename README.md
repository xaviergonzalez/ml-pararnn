# ParaRNN

ParaRNN is a high-performance package for automating parallel application of RNNs along sequence-length, dramatically speed up RNN applications compared to traditional sequential approaches.

The code has been developed as part of the publication:
[ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models](https://arxiv.org/abs/2510.21450)

---

## Overview

Traditional RNN processing requires updating the RNN hidden state as the input sequence gets analyzed: a procedure inherently sequential, which makes its application to long sequences time-consuming.
ParaRNN overcomes this issue by implementing a combination of Newton method and parallel reduction algorithms which can effectively evaluate the RNN application in parallel along the sequence length.
The package supports various RNN architectures, and provides reference implementations for diagonalized GRU and LSTM variants.

### Features

**Automated Parallel RNN Framework**: ParaRNN provides full support for parallel application of custom RNN cells through automatic assembly of the Newton linearized system. The framework leverages PyTorch's autograd for Jacobian computations, requiring only the definition of the RNN cell's recurrent step and its system parameters. This allows researchers to focus on RNN cell design while the package handles the complex parallel processing automatically.

**High-Performance CUDA Kernels**: The package includes ready-to-use, efficient CUDA implementations of parallel reduction algorithms specifically optimized for structured Jacobians. These kernels are designed to handle the most common diagonal and block-diagonal Jacobian structures, providing significant performance improvements over the PyTorch implementation

**Modular and Extensible Architecture**: ParaRNN's modular structure simplifies development work by providing numerous possibilities for expansion and adaptations. The architecture supports the users from prototyping new RNN cells using the PyTorch backend to implementing fully-fused CUDA kernels for maximum performance. This flexibility enables both rapid experimentation and production-ready deployment.

### RNN Application Modes

ParaRNN supports four distinct application modes for the considered RNN cells, trading off ease-of-application against performance
1. **Sequential**: classical sequential application, mostly used for testing and debugging, or at inference time.
2. **Parallel**: reference implementation of Newton + Parallel Reduction, which only relies on native PyTorch operations. Mainly thought for prototyping, debugging, and exploring new RNN cell definitions.
3. **Parallel_CUDA**: performant implementation where Jacobian assembly and Newton iterations are performed in PyTorch, but the parallel reduction solver is implemented in a custom CUDA kernel, specialized for Jacobians with diagonal or NxN-block-diagonal structure.
4. **Parallel_FUSED**: top-performance implementation featuring a fully-fused custom CUDA kernel for the whole Newton routine. Requires prescribing the RNN cell action and Jacobians assembly in CUDA.

Modes can be swapped by setting
```python
model.mode = 'parallel_CUDA'
```
---
## Installation

ParaRNN requires Python 3.9+ with PyTorch and CUDA support. The package includes custom CUDA kernels that are compiled during installation, so a compatible C++ compiler and CUDA toolkit must be available on your system.

Install in development mode:

```bash
git clone https://github.com/apple/ml-pararnn
cd ml-pararnn

# Setup your virtual environment if you need to
# virtualenv venv
# source venv/bin/activate

# Install torch if you haven't already
# pip install torch

pip install -e . --no-build-isolation
```

The installation process will automatically build the required CUDA extensions.
> [!WARNING]
> Make sure your environment has CUDA available, as the package will not function properly without GPU support for the accelerated modes.

---

## Quick Start

Here's a complete example demonstrating how to compare performance between sequential and parallel RNN processing modes. This code uses ParaRNN's built-in testing function with a diagonalized GRU model (ParaGRU), showing the significant speedup achieved through parallel processing:

```python
import torch
from pararnn.rnn_cell.test import sequential_vs_parallel
from pararnn.rnn_cell.gru_diag_mh import GRUDiagMH, GRUDiagMHConfig

# Run the sequential vs parallel comparison
sequential_vs_parallel(
    model_type=GRUDiagMH,
    model_config_type=GRUDiagMHConfig,
    seq_length=256,
    device='cuda'
)
```

This test creates a ParaGRU cell and runs the same computation using different application modes (sequential, parallel, and parallel_cuda), comparing outputs and performance.
You'll see (rough) timing comparisons, showing the speedup achieved by parallel processing methods, and a printout of the errors of the output (and gradients) computed by every application mode.  
> [!WARNING]
> Expect errors to increase as `machine_precision * seq_length`, due to the sequence-wise accumulation of numerical approximations.

---

## Creating Custom RNN Cells

ParaRNN's modular architecture makes it straightforward to implement custom RNN cells. You need to define three main components: the system parameters, the cell class, and the implementation class.

### Step 1: Define System Parameters

First, create a dataclass that inherits from `SystemParameters` to define your RNN's learnable parameters, activation functions, and configuration. This example shows a minimal custom RNN structure with a single weight matrix and nonlinear activation:

```python
from dataclasses import dataclass
from pararnn.rnn_cell.rnn_cell_utils import SystemParameters, Config
import torch
import typing as typ

# Type variable for generic type hints
T = typ.TypeVar("T")

# Trait class serves as a marker/tag for this specific RNN type
# Used by the generic type system to ensure type consistency
@dataclass(frozen=True)
class MyRNNTrait:
    pass

# Configuration dataclass defines hyperparameters and settings
# Inherits from Config with the trait as a type parameter
@dataclass
class MyRNNConfig(Config[MyRNNTrait]):
    my_nonlin_type: str = "tanh"  # Activation function type

# System parameters encapsulate all learnable parameters and functions
# This is what gets passed to the recurrence step during computation
@dataclass
class MyRNNSystemParameters(SystemParameters[MyRNNTrait]):
    # Learnable parameters
    my_weight: torch.Tensor      # Weight matrix for state transformation
    my_nonlin: typ.Callable      # Activation function (e.g., tanh, relu)

    # Unpacks parameters into a tuple for easy manipulation
    # Used internally by the framework for parameter extraction
    def unpack(self) -> typ.Tuple[
        torch.Tensor, typ.Callable
    ]:
        return (
            self.my_weight, self.my_nonlin
        )

    # Repacks parameters from a tuple back into the dataclass
    # Used after parameter updates or transformations
    @classmethod
    def repack(
            cls: T,
            pars: typ.Tuple[
                torch.Tensor,
                typ.Callable,
            ]
    ) -> T:
        return MyRNNSystemParameters(
            my_weight=pars[0],
            my_nonlin=pars[1],
        )

```

### Step 2: Choose Implementation Structure

Next, choose the appropriate base implementation class based on your RNN's Jacobian structure for optimal performance. The choice affects whether CUDA-accelerated parallel reduction algorithms are available. Implement the core `recurrence_step` method that defines your RNN's state transition:

```python
from pararnn.rnn_cell.rnn_cell_impl import RNNCellDenseImpl, RNNCellDiagImpl, RNNCellBlockDiagImpl

# For dense Jacobians (no optimization)
class MyRNNImpl(RNNCellDenseImpl[MyRNNSystemParameters]):
    ...
# For diagonal Jacobians (CUDA-accelerated parallel reduction available)
class MyRNNImpl(RNNCellDiagImpl[MyRNNSystemParameters]):
    ...
# For block-diagonal Jacobians (CUDA-accelerated parallel reduction available)
class MyRNNImpl(RNNCellBlockDiagImpl[MyRNNSystemParameters]):
    ... 
    
    @classmethod
    def recurrence_step(
            cls,
            x,                  # (B), T, D_in
            h,                  # (B), T, D
            system_parameters
    ):
        """Core recurrence: h_t = f(h_{t-1}, x_t; params)"""
        ...    
```

### Step 3: Define the Cell Class

Finally, create the main RNN cell class that manages parameters and state. This class inherits from `BaseRNNCell` and handles parameter initialization, device management, and provides the interface for forward passes:

```python
from pararnn.rnn_cell.rnn_cell import BaseRNNCell


class MyRNNCell(BaseRNNCell[MyRNNConfig, MyRNNSystemParameters, MyRNNImpl]):
    
    def __init__(self, config):
        super().__init__(config)
    
    def _specific_init(self, config):
        # Initialize learnable parameters
        self.my_weight = torch.nn.Parameter( ... )
        self.my_nonlin = ...
        ...
    
    @property
    def _system_parameters(self):
        return MyRNNSystemParameters(
            my_weight=self.my_weight,
            my_nonlin=self.my_nonlin,
            ...
        )
```

### Usage

Your custom RNN cell automatically supports multiple ParaRNN application modes: `sequential` and `parallel` are always available. Here's how to instantiate a cell and apply it in parallel:

```python
# Create and use your custom cell
config = MyRNNConfig(state_dim=64, input_dim=32, mode='parallel')
model = MyRNNCell(config)

x = torch.randn(batch_size, seq_length, input_dim, device='cuda')
output = model(x)  # Automatically uses parallel processing!
```

The framework handles Jacobian computation via `autograd` and automatically assembles the Newton linearized system, allowing you to focus purely on experimenting with different recurrence relationship definitions.

Moreover, if a (block-)diagonal structure for the cell Jacobians was flagged (ie, if `MyRNNImpl` inherits from the `-Diag` or `-BlockDiag` specializations), the automatic Jacobians computation becomes more efficient, and the `parallel_CUDA` mode also becomes available out-of-the-box. This leverages specialized CUDA kernels to perform parallel reduction, while still relying on PyTorch for system assembly.

Finally, unlocking the `parallel_FUSED` application mode requires providing a CUDA implementation of the cell's recurrence step and Jacobians. This combines the whole Newton routine (including system assembly and parallel reduction) into a single CUDA kernel, and is by far the fastest mode.  

> [!WARNING]
> Make sure to verify Newton's stability when defining your own cells! Particularly, unbounded Jacobians would likely result in hidden state explosion and poor convergence.

---

## Package Structure

The codebase is organized into the following key components:

```
pararnn/
    rnn_cell/                   # Core RNN cell implementations and configurations
        rnn_cell.py                 # Stateful base class for RNN cell definition
        rnn_cell_impl.py            # Impl class containing cell-specific methods implementation
        rnn_cell_application.py     # Collection of static methods defining the cell application modes 
    parallel_reduction/         # Parallel reduction algorithms and Newton solvers - PyTorch
    csrc/                       # CUDA C++ implementations for GPU acceleration of parallel reduction
    utils/                      # Utility functions (initialization, timing, ...)
```

---

## How to cite
If you find ParaRNN useful in your research, please consider citing:

### [ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models](https://arxiv.org/abs/2510.21450)

```bibtex
@misc{danieli2025pararnnunlockingparalleltraining,
      title={{ParaRNN}: Unlocking Parallel Training of Nonlinear {RNNs} for Large Language Models}, 
      author={Federico Danieli and Pau Rodr\'iguez and Miguel Sarabia and Xavier Suau and Luca Zappella},
      year={2025},
      eprint={2510.21450},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.21450}, 
}
```
[![arXiv](https://img.shields.io/badge/arXiv-2510.21450-b31b1b.svg)](https://arxiv.org/abs/2510.21450)
