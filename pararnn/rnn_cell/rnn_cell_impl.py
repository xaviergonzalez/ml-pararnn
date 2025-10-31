#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as typ
import typing_inspect
from functools import partial
import abc

import torch
import math

from pararnn.rnn_cell.rnn_cell_utils import SystemParametersT, TraitT
from pararnn.parallel_reduction.parallel_reduction import ParallelSolve


class RNNCellImpl(abc.ABC, typ.Generic[SystemParametersT]):
    """
    Generic Implementation of a Recurrent Model
    """
    last_state = None  # Class variable to store h
    
    @classmethod
    @typ.final
    def trait(cls) -> typ.Type[TraitT]:
        my_system_params = typing_inspect.get_args(typing_inspect.get_generic_bases(cls)[0])[0]
        return my_system_params.trait()
    
    @classmethod
    @typ.final
    def system_parameters_type(cls) -> typ.Type[SystemParametersT]:
        generic_base_type, = typing_inspect.get_generic_bases(cls)
        system_parameters_type, = typing_inspect.get_args(generic_base_type)
        return system_parameters_type
    
    @classmethod
    @abc.abstractmethod
    def parallel_reduce(cls) -> typ.Callable:
        return partial(ParallelSolve.parallel_reduce, reduction_step=ParallelSolve._reduction_step_dense)
        
    
    @classmethod
    def parallel_reduce_cuda(cls) -> typ.Callable:
        raise NotImplementedError

    @classmethod
    def fused_parallel_forward(
            cls,
            x: torch.Tensor,  # Must work both for B,T,N and B,N
            system_parameters: SystemParametersT
    ) -> torch.Tensor:
        raise NotImplementedError
    
    @classmethod
    def _roll_state(cls, h: torch.Tensor) -> torch.Tensor:
        h_prev = torch.roll(h, shifts=1, dims=-2)
        h_prev[..., 0, :] = 0.
        return h_prev
    
    # Model Application ================================================================================================
    
    @classmethod
    def post_process(
            cls,
            h: torch.Tensor,  # should work both with B,T,N and T,B,N
            x: torch.Tensor,
            system_parameters: SystemParametersT
    ) -> torch.Tensor:
        """
        Defaults to identity
        """
        return h
    
    @staticmethod
    @abc.abstractmethod
    def recurrence_step(
            x: torch.Tensor,  # Must work both for B,T,N and B,N
            h: torch.Tensor,  # Must work both for B,T,N and B,N
            system_parameters: SystemParametersT
    ) -> torch.Tensor:
        """
        The core of the implementation: definition of the recurrence relationship ht = f( ht-1, xt; params)
        """
        ...
    
    # Parallel forward pass ============================================================================================
    
    @classmethod
    def assemble_initial_guess(
            cls,
            x: torch.Tensor,
            state_dim: int,
            system_parameters: SystemParametersT
    ) -> torch.Tensor:
        return torch.zeros([*x.shape[:-1], state_dim], device=x.device, dtype=x.dtype)
    
    @classmethod
    def _transpose_jacobians(cls, jac: torch.Tensor) -> torch.Tensor:
        return jac
    
    @classmethod
    @abc.abstractmethod
    def compute_jacobians(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: SystemParametersT,
    ) -> torch.Tensor:
        """
        Computation of Jacobians of recurrence relationship df/dht
        This too can be computed using autograd, but the required operations depend on the actual Jacobian structure
        """
        ...

    @classmethod
    @abc.abstractmethod
    def compute_jacobians_bwd(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: SystemParametersT,
    ) -> torch.Tensor:
        """
        Computation of Jacobians of recurrence relationship df/dht - version for assembly of the backward system
        This too can be computed using autograd, but the required operations depend on the actual Jacobian structure
        """
        ...

    
    @classmethod
    def compute_negative_residuals(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: SystemParametersT
    ) -> torch.Tensor:
        """
        Residual of recurrence relationship are automatically computed starting from recurrence step definition
        """
        
        sol_prev = cls._roll_state(sol)
        
        h = cls.recurrence_step(x, sol_prev, system_parameters)
        
        return - (sol - h)
    
    # Parallel backward pass ===========================================================================================
    @classmethod
    def backprop_to_system_parameters(
            cls,
            dl_dht: torch.Tensor,
            x: torch.Tensor,
            h: torch.Tensor,
            system_parameters: SystemParametersT
    ) -> typ.Tuple[typ.Union[torch.Tensor, None], ...]:
        """
        By default, grads wrt system parameters are automatically computed using autograd, starting from grads wrt state
        """
        with torch.set_grad_enabled(True):
            h_prev = cls._roll_state(h)
            inputs = [x.requires_grad_(True)]
            for param in system_parameters:
                if torch.is_tensor(param) and param.requires_grad:
                    inputs.append(param)
            
            grads = list(
                torch.autograd.grad(
                    outputs=cls.recurrence_step(x, h_prev, system_parameters),
                    inputs=inputs,
                    grad_outputs=dl_dht,
                    allow_unused=True,
                    materialize_grads=True
                )
            )
        
        for (i, param) in enumerate(system_parameters):
            if not torch.is_tensor(param) or not param.requires_grad:
                grads.insert(i + 1, None)
        
        return tuple(grads)
    
    @classmethod
    def backprop_post_processing(
            cls,
            grad_y: torch.Tensor,
            x: torch.Tensor,
            h: torch.Tensor,
            system_parameters: SystemParametersT
    ) -> typ.Tuple[typ.Union[torch.Tensor, None], ...]:
        """
        By default, grads wrt post-processing operations are automatically computed using autograd
        """

        with torch.set_grad_enabled(True):
            inputs = [h.requires_grad_(True), x.requires_grad_(True)]
            for param in system_parameters:
                if torch.is_tensor(param) and param.requires_grad:
                    inputs.append(param)
            
            grads = list(
                torch.autograd.grad(
                    outputs=cls.post_process(h, x, system_parameters),
                    inputs=inputs,
                    grad_outputs=grad_y,
                    allow_unused=True,
                    materialize_grads=True
                )
            )
        
        for (i, param) in enumerate(system_parameters):
            if not torch.is_tensor(param) or not param.requires_grad:
                grads.insert(i + 2, None)
        
        return tuple(grads)


# TODO: maybe better define this inside each class that employs it, rather than here once and for all?
RNNCellImplT = typ.TypeVar("RNNCellImplT", bound=RNNCellImpl)


class RNNCellDenseImpl(RNNCellImpl[SystemParametersT], typ.Generic[SystemParametersT]):
    """
    Specialisation for Implementation of a Recurrent Model with Dense Jacobians
    """
    
    @classmethod
    def parallel_reduce(cls) -> typ.Callable:
        return partial(ParallelSolve.parallel_reduce, reduction_step=ParallelSolve._reduction_step_dense)
    
    @classmethod
    def parallel_reduce_cuda(cls) -> typ.Callable:
        raise NotImplementedError("CUDA parallel reduction solver not Implemented for dense Jacobians")
    
    @classmethod
    def _transpose_jacobians(cls, jac: torch.Tensor) -> torch.Tensor:
        return jac.transpose(-1, -2)
    
    @classmethod
    def _compute_jacobians_inner(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: SystemParametersT,
    ) -> torch.Tensor:
        with torch.set_grad_enabled(True):
            h_prev = cls._roll_state(sol)
            h_prev.requires_grad_(True)
            jac_func = torch.func.vmap(
                torch.func.jacrev(cls.recurrence_step, argnums=1),
                in_dims=(0, 0, None)
            )
            jacobians = jac_func(
                x.view(-1,x.shape[-1]),
                h_prev.view(-1,h_prev.shape[-1]),
                system_parameters
            ).reshape([*h_prev.shape,h_prev.shape[-1]])
        return - jacobians
    
    @classmethod
    def compute_jacobians(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: SystemParametersT,
    ) -> torch.Tensor:
        return cls._compute_jacobians_inner(sol, x, system_parameters)
    
    @classmethod
    def compute_jacobians_bwd(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: SystemParametersT,
    ) -> torch.Tensor:
        jacobians = cls._compute_jacobians_inner(sol, x, system_parameters)
        
        # TODO: rather than flipping jacobians, flipping rhs before computing jacs should be faster!
        jacobians = cls._transpose_jacobians(
            torch.roll(
                torch.flip(
                    jacobians,
                    dims=[-3]
                ),
                shifts=1,
                dims=-3
            )
        )
        jacobians[..., 0,:,:] = 0
        
        return jacobians
    

class RNNCellDiagImpl(RNNCellImpl[SystemParametersT], typ.Generic[SystemParametersT]):
    """
    Specialisation for Implementation of a Recurrent Model with Diagonal Jacobians
    """
    
    @classmethod
    def parallel_reduce(cls) -> typ.Callable:
        return partial(ParallelSolve.parallel_reduce, reduction_step=ParallelSolve._reduction_step_diag)
    
    @classmethod
    def _parallel_reduce_cuda_wrapper(
            cls,
            jac: torch.Tensor,
            rhs: torch.Tensor,
    ) -> torch.Tensor:
        """
        TODO: this is a terrible hack: parallel_reduce_cuda expects [B,N,T], while all other funcs expect [B,T,N]
              so I'm transposing and re-transposing at every iteration!
        """
        return ParallelSolve.parallel_reduce_diag_cuda(jac.transpose(-1, -2), rhs.transpose(-1, -2)).transpose(-1, -2)
    
    @classmethod
    def parallel_reduce_cuda(cls) -> typ.Callable:
        return cls._parallel_reduce_cuda_wrapper
    
    @classmethod
    def _transpose_jacobians(cls, jac: torch.Tensor) -> torch.Tensor:
        # Transpose of diagonal matrix is itself
        return jac
    
    @classmethod
    def _compute_jacobians_inner(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: SystemParametersT,
    ) -> torch.Tensor:
        with torch.set_grad_enabled(True):
            h_prev = cls._roll_state(sol)
            h_prev.requires_grad_(True)
            jacobians = - torch.autograd.grad(
                outputs=cls.recurrence_step(x, h_prev, system_parameters).sum(),
                inputs=h_prev,
                retain_graph=False,
                allow_unused=True
            )[0]
        return jacobians
    
    @classmethod
    def compute_jacobians(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: SystemParametersT,
    ) -> torch.Tensor:
        return cls._compute_jacobians_inner(sol, x, system_parameters )
    
    @classmethod
    def compute_jacobians_bwd(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: SystemParametersT,
    ) -> torch.Tensor:
        jacobians = cls._compute_jacobians_inner(sol, x, system_parameters)
        
        # TODO: rather than flipping jacobians, flipping rhs before computing jacs should be faster!
        jacobians = cls._transpose_jacobians(
            torch.roll(
                torch.flip(
                    jacobians,
                    dims=[-2]
                ),
                shifts=1,
                dims=-2
            )
        )
        jacobians[..., 0, :] = 0
        
        return jacobians


class RNNCellBlockDiagImpl(RNNCellImpl[SystemParametersT], typ.Generic[SystemParametersT]):
    """
    Specialisation for Implementation of a Recurrent Model with Block Diagonal Jacobians
    NB: all blocks must necessarily be *of the same size* - otherwise the structure wouldn't be block-*diagonal*...
    """
    
    @classmethod
    @abc.abstractmethod
    def _num_blocks(cls) -> int:
        """
        Util function to define number of (same-size) blocks the hidden state is split into
        """
        ...
    
    @classmethod
    def parallel_reduce(cls) -> typ.Callable:
        return partial(ParallelSolve.parallel_reduce, reduction_step=ParallelSolve._reduction_step_block_diag)
    
    @classmethod
    def _parallel_reduce_cuda_wrapper(
            cls,
            jac: torch.Tensor,  # [(B), T, N*K, K]
            rhs: torch.Tensor,  # [(B), T, N*K]
    ) -> torch.Tensor:
        # TODO: this is a terrible hack: _parallel_reduce_cuda expects [B,N,T,K (,K)], while all other funcs expect [B,T,N*K (,K)]
        #       so I'm chunking, transposing, and recombining at every iteration!
        
        block_size = math.ceil(rhs.shape[-1] / cls._num_blocks())
        my_rhs = torch.stack(rhs.split(dim=-1, split_size=block_size), dim=-1).transpose(-3, -2)
        my_jac = torch.stack(jac.split(dim=-2, split_size=block_size), dim=-2).transpose(-3, -4)
        
        num_blocks = cls._num_blocks()
        if num_blocks == 2:
            sol = ParallelSolve.parallel_reduce_block_diag_2x2_cuda(my_jac, my_rhs)
        elif num_blocks == 3:
            sol = ParallelSolve.parallel_reduce_block_diag_3x3_cuda(my_jac, my_rhs)
        else:
            raise NotImplementedError(
                f"CUDA block-diagonal solver is not compiled by default for {num_blocks}x{num_blocks} Jacobians. "
                "The required specialisation can however be added to the torch bindings, and interfaced in parallel_reduction.py"
            )
        
        # return torch.einsum('...ntk->...tkn', sol).flatten(-2)
        return sol.permute(*range(sol.ndim - 3), -2, -1, -3).flatten(-2)


    @classmethod
    def parallel_reduce_cuda(cls) -> typ.Callable:
        return cls._parallel_reduce_cuda_wrapper
    
    @classmethod
    def _transpose_jacobians(
            cls,
            jac: torch.Tensor  # [(B,T),N*K,K] with K=cls._num_blocks()
    ) -> torch.Tensor:
        num_blocks = cls._num_blocks()
        block_size = math.ceil(jac.shape[-2] / jac.shape[-1])
        for i in range(num_blocks):
            for j in range(i + 1, num_blocks):
                temp = jac[..., i * block_size:(i + 1) * block_size,
                       j].clone()  # must clone otherwise torch returns a view!
                jac[..., i * block_size:(i + 1) * block_size, j] = jac[..., j * block_size:(j + 1) * block_size, i]
                jac[..., j * block_size:(j + 1) * block_size, i] = temp
        
        return jac
    
    @classmethod
    def _compute_jacobians_inner(
        cls,
        sol: torch.Tensor,
        x: torch.Tensor,
        system_parameters: SystemParametersT,
    ) -> torch.Tensor:
    
        with torch.set_grad_enabled(True):
            h_prev = cls._roll_state(sol)
            h_prev.requires_grad_(True)
            # Need to independently compute Jacobians wrt variables in each block -> vmap it!
            num_blocks = cls._num_blocks()
            block_size = math.ceil(h_prev.shape[-1] / num_blocks)
            tangents = torch.zeros([*h_prev.shape, num_blocks], device=h_prev.device)
            for block in range(num_blocks):
                tangents[..., block * block_size:(block + 1) * block_size, block] = 1
            
            def block_partial_derivative(tangent: torch.Tensor) -> torch.Tensor:
                return - torch.autograd.grad(
                    outputs=cls.recurrence_step(x, h_prev, system_parameters),
                    grad_outputs=tangent,
                    inputs=h_prev,
                    retain_graph=False,
                    allow_unused=True
                )[0]
            
            # this actually returns the transpose of the Jacobian!
            jacobians = torch.vmap(block_partial_derivative, in_dims=-1, out_dims=-1)(tangents)
            jacobians = cls._transpose_jacobians(jacobians)
            
            return jacobians
    
    @classmethod
    def compute_jacobians(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: SystemParametersT,
    ) -> torch.Tensor:
        return cls._compute_jacobians_inner(sol, x, system_parameters )
        
    @classmethod
    def compute_jacobians_bwd(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: SystemParametersT,
    ) -> torch.Tensor:
        jacobians = cls._compute_jacobians_inner( sol, x, system_parameters )
        jacobians = cls._transpose_jacobians(jacobians)    # TODO: annoying that I need to transpose and re-transpose...
        # TODO: rather than flipping jacobians, flipping sol before computing jacs might be faster!
        jacobians = torch.roll(
            torch.flip(
                jacobians,
                dims=[-3]
            ),
            shifts=1,
            dims=-3
        )
        jacobians[..., 0, :, :] = 0
        
        return jacobians


