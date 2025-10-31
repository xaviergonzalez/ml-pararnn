#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as typ
import typing_extensions as typx
from functools import partial
from enum import Enum

import torch

from pararnn.rnn_cell.rnn_cell_utils import SystemParametersT
from pararnn.rnn_cell.rnn_cell_impl import RNNCellImplT
from pararnn.parallel_reduction.parallel_reduction import ParallelSolve, NewtonConfig


class RNNCellApplicationMode(str, Enum):
    """
    Different ways of applying the Recurrent Model
    """
    SEQUENTIAL = "sequential"           # Classical, sequential application
    PARALLEL = "parallel"               # Parallel application in PyTorch
    PARALLEL_CUDA = "parallel_CUDA"     # Parallel application with custom CUDA kernels for reduction operations
    PARALLEL_FUSED = "parallel_FUSED"   # Parallel application with fully-fused CUDA kernels for whole Newton routine


class RNNCellSequentialApplication(typ.Generic[RNNCellImplT, SystemParametersT]):
    
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(
            x: torch.Tensor,  # (B), T, N
            state_dim: int,
            impl: typ.Type[RNNCellImplT],
            system_parameters: SystemParametersT
    ) -> torch.Tensor:
        h = torch.empty([*x.shape[:-1], state_dim], dtype=x.dtype, device=x.device)  # (B), T, N
        h_prev = torch.zeros_like(h[...,0,:])                                        # (B), N
        
        for t in range(x.shape[-2]):
            h_prev = impl.recurrence_step(
                x[...,t,:],
                h_prev,
                system_parameters
            )
            h[...,t,:] = h_prev

        return impl.post_process(h, x, system_parameters)
        

class RNNCellParallelApplication(torch.autograd.Function, typ.Generic[RNNCellImplT, SystemParametersT]):
    
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(
            ctx: typ.Any,
            x: torch.Tensor,  # (B), T, N
            state_dim: int,
            impl: typ.Type[RNNCellImplT],
            system_parameters_type: typ.Type[SystemParametersT],
            newton_config: NewtonConfig,
            *system_parameters_tuple: typ.Type[typx.Unpack[SystemParametersT]]
    ) -> torch.Tensor:
        with torch.no_grad():

            system_parameters = system_parameters_type.repack(
                system_parameters_tuple
            )
            ctx.system_parameters = system_parameters  # shallow-copy system_parameters
            ctx.impl = impl
            
            h0 = impl.assemble_initial_guess(x, state_dim, system_parameters)
            
            solve = RNNCellParallelApplication._setup_forward_parallel_solver(x, system_parameters, impl, newton_config)
            
            h, its, flag = solve(h0)
            h = h.detach()
            
            y = impl.post_process(h, x, system_parameters)
            
            ctx.save_for_backward(x, h)
        
            return y.detach()

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(
            ctx: typ.Any,
            grad_y: typ.Any,
    ) -> typ.Any:
        system_parameters = ctx.system_parameters
        impl = ctx.impl
        
        x, h = ctx.saved_tensors

        grad_h, grad_x_post_proc, *grad_params_post_proc = impl.backprop_post_processing(
            grad_y=grad_y,
            x=x,
            h=h,
            system_parameters=system_parameters
        )

        dl_dht = RNNCellParallelApplication._backprop_recursion(
            gradient=grad_h,
            x=x,
            h=h,
            system_parameters=system_parameters,
            impl=impl
        )
        
        grad_x_rec, *grad_params_recursion = impl.backprop_to_system_parameters(
            dl_dht=dl_dht,
            x=x,
            h=h,
            system_parameters=system_parameters
        )
        
        grad_x = grad_x_post_proc + grad_x_rec
        grad_system_params = [
            None if grad_pp is None or grad_rec is None else grad_pp + grad_rec
            for (grad_pp, grad_rec) in zip(grad_params_post_proc, grad_params_recursion)
        ]
        
        return grad_x, None, None, None, None, *grad_system_params
    
    @staticmethod
    def _backprop_recursion(
            gradient: torch.Tensor,  # (B), T, N
            h: torch.Tensor,  # (B), T, N
            x: torch.Tensor,  # (B), T, N
            system_parameters: SystemParametersT,
            impl: typ.Type[RNNCellImplT]
    ) -> torch.Tensor:
        rhs = torch.flip(gradient, dims=[-2])
        
        jacobians = impl.compute_jacobians_bwd(h, x, system_parameters)
        
        dl_dht = torch.flip(RNNCellParallelApplication._get_parallel_reduction(impl)(jacobians, rhs), dims=[-2])
        
        return dl_dht
    
    @staticmethod
    def _setup_forward_parallel_solver(
            x: torch.Tensor,
            system_parameters: SystemParametersT,
            impl: typ.Type[RNNCellImplT],
            newton_config: NewtonConfig
    ) -> typ.Callable:
        compute_negative_residuals = partial(
            impl.compute_negative_residuals,
            x=x,
            system_parameters=system_parameters
        )
        compute_jacobians = partial(
            impl.compute_jacobians,
            x=x,
            system_parameters=system_parameters
        )
        solve = partial(
            ParallelSolve.newton_solve,
            compute_negative_residuals=compute_negative_residuals,
            compute_jacobians=compute_jacobians,
            linear_solve=RNNCellParallelApplication._get_parallel_reduction(impl),
            newton_config=newton_config
        )
        return solve
    
    @staticmethod
    def _get_parallel_reduction(impl: typ.Type[RNNCellImplT]) -> typ.Callable:
        return impl.parallel_reduce()


class RNNCellParallelCUDAApplication(torch.autograd.Function, typ.Generic[RNNCellImplT, SystemParametersT]):
    
    # TODO: This could actually just inherit from RNNCellParallelApplication and overwrite _get_parallel_reduction(),
    #       but torch.autograd.Function expects staticmethods, not classmethods, so let's just copy-paste...
    
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(
            ctx: typ.Any,
            x: torch.Tensor,  # (B), T, N
            state_dim: int,
            impl: typ.Type[RNNCellImplT],
            system_parameters_type: typ.Type[SystemParametersT],
            newton_config: NewtonConfig,
            *system_parameters_tuple: typ.Type[typx.Unpack[SystemParametersT]]
    ) -> torch.Tensor:
        with torch.no_grad():
            system_parameters = system_parameters_type.repack(
                system_parameters_tuple
            )
            ctx.system_parameters = system_parameters  # shallow-copy system_parameters
            ctx.impl = impl
            
            h0 = impl.assemble_initial_guess(x, state_dim, system_parameters)
            
            solve = RNNCellParallelCUDAApplication._setup_forward_parallel_solver(x, system_parameters, impl, newton_config)
            
            h, its, flag = solve(h0)
            h = h.detach()
            
            y = impl.post_process(h, x, system_parameters)
            
            ctx.save_for_backward(x, h)
    
            return y.detach()
    
    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(
            ctx: typ.Any,
            grad_y: typ.Any,
    ) -> typ.Any:
        system_parameters = ctx.system_parameters
        impl = ctx.impl
        
        x, h = ctx.saved_tensors
        
        grad_h, grad_x_post_proc, *grad_params_post_proc = impl.backprop_post_processing(
            grad_y=grad_y,
            x=x,
            h=h,
            system_parameters=system_parameters
        )

        dl_dht = RNNCellParallelCUDAApplication._backprop_recursion(
            gradient=grad_h,
            x=x,
            h=h,
            system_parameters=system_parameters,
            impl=impl
        )

        grad_x_rec, *grad_params_recursion = impl.backprop_to_system_parameters(
            dl_dht=dl_dht,
            x=x,
            h=h,
            system_parameters=system_parameters
        )

        grad_x = grad_x_post_proc + grad_x_rec
        
        grad_system_params = [
            None if grad_pp is None or grad_rec is None else grad_pp + grad_rec
            for (grad_pp, grad_rec) in zip(grad_params_post_proc, grad_params_recursion)
        ]

        return grad_x, None, None, None, None, *grad_system_params
    
    @staticmethod
    def _backprop_recursion(
            gradient: torch.Tensor,  # (B), T, N
            h: torch.Tensor,  # (B), T, N
            x: torch.Tensor,  # (B), T, N
            system_parameters: SystemParametersT,
            impl: typ.Type[RNNCellImplT]
    ) -> torch.Tensor:
        rhs = torch.flip(gradient, dims=[-2])
        
        jacobians = impl.compute_jacobians_bwd(h, x, system_parameters)
        
        dl_dht = torch.flip(RNNCellParallelCUDAApplication._get_parallel_reduction(impl)(jacobians, rhs), dims=[-2])
        
        return dl_dht
    
    @staticmethod
    def _setup_forward_parallel_solver(
            x: torch.Tensor,
            system_parameters: SystemParametersT,
            impl: typ.Type[RNNCellImplT],
            newton_config: NewtonConfig
    ) -> typ.Callable:
        compute_negative_residuals = partial(
            impl.compute_negative_residuals,
            x=x,
            system_parameters=system_parameters
        )
        compute_jacobians = partial(
            impl.compute_jacobians,
            x=x,
            system_parameters=system_parameters
        )
        solve = partial(
            ParallelSolve.newton_solve,
            compute_negative_residuals=compute_negative_residuals,
            compute_jacobians=compute_jacobians,
            linear_solve=RNNCellParallelCUDAApplication._get_parallel_reduction(impl),
            newton_config=newton_config
        )
        return solve
  
    @staticmethod
    def _get_parallel_reduction(impl: typ.Type[RNNCellImplT]) -> typ.Callable:
        return impl.parallel_reduce_cuda()


class RNNCellParallelFusedApplication(torch.autograd.Function, typ.Generic[RNNCellImplT, SystemParametersT]):
    """
    Bypass any fancy modularity, and just brute-force implement the whole parallel solver in pure CUDA
    """
    
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(
            ctx: typ.Any,
            x: torch.Tensor,  # (B), T, N
            state_dim: int,
            impl: typ.Type[RNNCellImplT],
            system_parameters_type: typ.Type[SystemParametersT],
            newton_config: NewtonConfig,        # ignored here
            *system_parameters_tuple: typ.Type[typx.Unpack[SystemParametersT]]
    ) -> torch.Tensor:
        with torch.no_grad():
            system_parameters = system_parameters_type.repack(
                system_parameters_tuple
            )
            ctx.system_parameters = system_parameters  # shallow-copy system_parameters
            ctx.impl = impl
            
            h = impl.fused_parallel_forward(x, system_parameters)
            h = h.detach()
            
            y = impl.post_process(h, x, system_parameters)
            
            ctx.save_for_backward(x, h)
        
        return y.detach()
    
    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(
            ctx: typ.Any,
            grad_y: typ.Any,
    ) -> typ.Any:
        system_parameters = ctx.system_parameters
        impl = ctx.impl
        
        x, h = ctx.saved_tensors
        
        grad_h, grad_x_post_proc, *grad_params_post_proc = impl.backprop_post_processing(
            grad_y=grad_y,
            x=x,
            h=h,
            system_parameters=system_parameters
        )
        
        dl_dht = RNNCellParallelFusedApplication._backprop_recursion(
            gradient=grad_h,
            x=x,
            h=h,
            system_parameters=system_parameters,
            impl=impl
        )
        
        grad_x_rec, *grad_params_recursion = impl.backprop_to_system_parameters(
            dl_dht=dl_dht,
            x=x,
            h=h,
            system_parameters=system_parameters
        )
        
        grad_x = grad_x_post_proc + grad_x_rec
        
        grad_system_params = [
            None if grad_pp is None or grad_rec is None else grad_pp + grad_rec
            for (grad_pp, grad_rec) in zip(grad_params_post_proc, grad_params_recursion)
        ]
        
        return grad_x, None, None, None, None, *grad_system_params
    
    @staticmethod
    def _backprop_recursion(
            gradient: torch.Tensor,  # (B), T, N
            h: torch.Tensor,  # (B), T, N
            x: torch.Tensor,  # (B), T, N
            system_parameters: SystemParametersT,
            impl: typ.Type[RNNCellImplT]
    ) -> torch.Tensor:
        dl_dht = impl.fused_parallel_backward(gradient, h, x, system_parameters)

        return dl_dht
    






