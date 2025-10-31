#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from enum import IntEnum
import typing as typ
from dataclasses import dataclass

from pathlib import Path
import math
import torch

from pararnn.parallel_reduction.utils import get_diag_chunk_size, get_block_diag_2x2_chunk_size
from pararnn.parallel_reduction.utils import get_threads_per_block, get_threads_per_warp


# torch.ops.load_library("../../parallel_reduce_cuda.cpython-310-x86_64-linux-gnu.so")
LIB_PATH = str(next(iter(Path(__file__).resolve().parent.parent.parent.glob("*parallel_reduce_cuda.cpython*so"))))
torch.ops.load_library(LIB_PATH)


@torch.library.register_fake("parallel_reduce_cuda::parallel_reduce_diag_cuda")
def _(jac, rhs):
    torch._check(jac.shape == rhs.shape)
    torch._check(jac.device == rhs.device)
    return torch.empty_like(rhs)


@torch.library.register_fake("parallel_reduce_cuda::parallel_reduce_block_diag_2x2_cuda")
def _(jac, rhs):
    torch._check(jac.shape[:-1] == rhs.shape)
    torch._check(jac.shape[-1] == rhs.shape[-1])
    torch._check(jac.shape[-1] == 2)
    torch._check(jac.device == rhs.device)
    return torch.empty_like(rhs)
@torch.library.register_fake("parallel_reduce_cuda::parallel_reduce_block_diag_3x3_cuda")
def _(jac, rhs):
    torch._check(jac.shape[:-1] == rhs.shape)
    torch._check(jac.shape[-1] == rhs.shape[-1])
    torch._check(jac.shape[-1] == 3)
    torch._check(jac.device == rhs.device)
    return torch.empty_like(rhs)


class NewtonTermination(IntEnum):
    MAX_ITERATIONS_REACHED = 0
    ABS_NORM_CONVERGED = 1
    REL_NORM_CONVERGED = 2


@dataclass
class NewtonConfig:
    max_its: int = 3         # stopping criterion 1: maximum number of its
    abs_tol: float = 0       # stopping criterion 2: achieved abs_tol ||f(xk)|| < abs_tol
    rel_tol: float = 0       # stopping criterion 3: achieved rel_tol ||f(xk)||/||f(x0)|| < rel_tol
    norm: str = 'inf'        # type of norm ||*|| considered for convergence
    omega_sor: float = 1.    # Successive-Over-Relaxation parameter (<1 stabilising; =1 vanilla Newton; >1 accelerating)
    verbose: int = 0


class ParallelSolve:
    @staticmethod
    def newton_solve(
            sol: torch.Tensor,
            compute_negative_residuals: typ.Callable,
            compute_jacobians: typ.Callable,
            linear_solve: typ.Callable,
            newton_config: NewtonConfig = NewtonConfig()
    ) -> typ.Tuple[torch.Tensor, int, NewtonTermination]:
        # def _get_residual_norm(residual: torch.Tensor) -> torch.Tensor:
        #     return torch.linalg.vector_norm(residual, dim=(1, 2), ord=float(newton_config.norm))
            
        # Initialise residuals
        # resnorms = torch.zeros(newton_config.max_its + 1, sol.shape[0], device=sol.device)
        
        # if newton_config.verbose > 4:
        #     print("Starting Newton solver: initial residual ", torch.max(resnorms[0]))
        
        # Main Newton its
        it = 0
        flag = NewtonTermination.MAX_ITERATIONS_REACHED
        for it in range(newton_config.max_its):
            
            rhs = compute_negative_residuals(sol)

            # Tolerance convergence check
            # resnorms[it] = _get_residual_norm(rhs)
            # if torch.all(resnorms[it] <= newton_config.abs_tol):
            #     flag = NewtonTermination.ABS_NORM_CONVERGED
            #     break
            # if it>0 and torch.all(resnorms[it] / resnorms[0] <= newton_config.rel_tol):
            #     flag = NewtonTermination.REL_NORM_CONVERGED
            #     break
            
            jacobians = compute_jacobians(sol)
            
            # Solve and update
            dsol = linear_solve(jacobians, rhs)
            sol = sol + newton_config.omega_sor * dsol
            
        # if newton_config.verbose > 3:
        #     print("All done in", it, "iterations, terminated with flag ", flag)
        
        return sol, it, flag
    
    @staticmethod
    def parallel_reduce(
            jacobians: torch.Tensor,    # (B),T,N?,N?
            rhs: torch.Tensor,          # (B),T,N?
            reduction_step: typ.Callable
    ) -> torch.Tensor:
        # num_steps = math.ceil(math.log2(rhs.shape[-2]))     # T must always be second to last for rhs!
        num_steps = (rhs.shape[-2]-1).bit_length()            # = ceil(log2(rhs.shape[-2]))
        for step in range(num_steps):
            jacobians, rhs = reduction_step(jacobians, rhs, step)
        return rhs
    
    @staticmethod
    def _reduction_step_dense(
            jacobians: torch.Tensor,    # (B),T,N,N
            rhs: torch.Tensor,          # (B),T,N
            step: int
    ) -> typ.Tuple[torch.Tensor, torch.Tensor]:
        """Generic parallel solve step with dense Jacobians"""
        # this defines the idx-distance between the eqs to be used for reduction
        idx = 1 << step     # = 2**step
        # reduction step
        rhs[..., idx:,:] -= torch.einsum('...tij,...tj->...ti', (jacobians[..., idx:,:,:], rhs[..., :-idx,:]))
        jacobians[..., idx:,:,:] = torch.einsum(
            '...tij,...tjk->...tik',
            -jacobians[..., idx:,:,:], jacobians[..., :-idx,:,:]
        )
        # set subdiagonals of first 2^i eqs to 0, just for safety
        # -> actually, this is necessary, because jac[0] can be \neq 0
        jacobians[..., :idx,:,:] = 0
        
        return jacobians, rhs
    
    @staticmethod
    def _reduction_step_diag(
            jacobians: torch.Tensor,    # (B),T,N
            rhs: torch.Tensor,          # (B),T,N
            step: int
    ) -> typ.Tuple[torch.Tensor, torch.Tensor]:
        """Generic parallel solve step with diagonal Jacobians"""
        # this defines the idx-distance between the eqs to be used for reduction
        idx = 1 << step     # = 2**step
        # reduction step
        rhs[..., idx:,:] -= jacobians[..., idx:,:] * rhs[..., :-idx,:]
        jacobians[..., idx:,:] = -jacobians[..., idx:,:] * jacobians[..., :-idx,:]
        
        # set subdiagonals of first 2^i eqs to 0, just for safety
        # -> actually, this is necessary, because jac[0] can be \neq 0
        jacobians[..., :idx,:] = 0
        
        return jacobians, rhs
    
    @staticmethod
    def _reduction_step_block_diag(
            jacobians: torch.Tensor,    # (B),T,m*N,m
            rhs: torch.Tensor,          # (B),T,m*N
            step: int
    ) -> typ.Tuple[torch.Tensor, torch.Tensor]:
        """Generic parallel solve step with Jacobians in a mxm block structure, each block being diagonal of same size"""
        # this defines the idx-distance between the eqs to be used for reduction
        idx = 1 << step     # = 2**step
        # reduction step
        # - rhs is
        # |00 01 02| |0|               |00 01 02|   |0 1 2|
        # |10 11 12| |1| = row_reduce( |10 11 12|(*)|0 1 2|  )
        # |20 21 22| |2|               |20 21 22|   |0 1 2|
        num_blocks = jacobians.shape[-1]
        block_size = math.ceil(jacobians.shape[-2]/num_blocks)

        rhs[..., idx:,:] -= torch.sum(
            jacobians[..., idx:,:,:] *
            rhs[..., :-idx,:].view(
                *rhs.shape[:-2], -1, num_blocks, block_size
            ).transpose(-1, -2).repeat(*([1]*len(rhs.shape[:-2])), 1, num_blocks, 1),
            dim=-1
        )
        # - A is similar, but with an extra dim to play with to perform the reduce
        jacobians[..., idx:,:,:] = -torch.sum(
            jacobians[..., idx:,:,:].unsqueeze(-3).expand(
                *([-1]*len(rhs.shape[:-2])), -1, jacobians.shape[-1],-1,-1
            ) *
            jacobians[..., :-idx,:,:].view(
                *jacobians.shape[:-3],-1,num_blocks,block_size,num_blocks
            ).transpose(-1,-3).repeat(*([1]*len(rhs.shape[:-2])),1,1,num_blocks,1),
            dim=-1
        ).transpose(-1,-2)
        
        # set subdiagonals of first 2^i eqs to 0, just for safety
        # -> actually, this is necessary, because jac[0] can be \neq 0
        jacobians[..., :idx,:,:] = 0

        return jacobians, rhs
    
    
    
    @staticmethod
    def parallel_reduce_diag_pseudo_cuda(
            jac: torch.Tensor,  # (B),T,N
            rhs: torch.Tensor,  # (B),T,N
    ) -> torch.Tensor:
        
        #TODO debug func that mimics behaviour of CUDA kernel in torch. Not all edge cases are yet considered in the implementation
        chunk_size = get_diag_chunk_size(jac.dtype)
        warp_size = get_threads_per_warp()
        block_size = get_threads_per_block()
        seq_length = rhs.shape[-2]

        # - Thomas reduction (as many steps as I can while staying inside the vec)
        chunk_spill = ((seq_length-1) % chunk_size) + 1
        for i in range(1, chunk_spill):
            rhs[:, :, i::chunk_size] -= jac[:, :, i::chunk_size] * rhs[:, :, i-1:-1:chunk_size]
            jac[:, :, i::chunk_size] *= - jac[:, :, i-1:-1:chunk_size]
        # - - trailing Thomas reduction (excluding last chunk)
        for i in range(chunk_spill, chunk_size):
            rhs[:, :, i::chunk_size] -= jac[:, :, i::chunk_size] * rhs[:, :, i-1:1-chunk_spill:chunk_size]
            jac[:, :, i::chunk_size] *= - jac[:, :, i-1:1-chunk_spill:chunk_size]

        # - parallel reduction of last eqs in chunk:
        # - - Within warp
        for i in range(0, math.ceil(math.log2(warp_size))):
            po2 = 2 ** i
            in_mask = (torch.arange(min(warp_size, block_size), device='cuda') >= po2).repeat(
                [math.ceil(block_size / warp_size)]).unsqueeze(0).unsqueeze(0)
            rhs[:, :, chunk_size - 1::chunk_size] -= (
                    jac[:, :, chunk_size - 1::chunk_size] *
                    rhs[:, :, chunk_size - 1::chunk_size].roll(po2, dims=2) * in_mask
            )
            jac[:, :, chunk_size - 1::chunk_size] *= (
                    - jac[:, :, chunk_size - 1::chunk_size].roll(po2, dims=2) * in_mask +
                    (~in_mask)
            )
        # - parallel reduction of last eqs in chunk:
        #   - within block
        for i in range(0, math.ceil(math.log2(block_size / warp_size))):
            po2 = 2 ** i
            in_mask = (torch.arange(math.ceil(block_size / warp_size), device='cuda') >= po2).unsqueeze(0).unsqueeze(0)
            rhs[:, :, warp_size * chunk_size - 1::warp_size * chunk_size] -= (
                    jac[:, :, warp_size * chunk_size - 1::warp_size * chunk_size] *
                    rhs[:, :, warp_size * chunk_size - 1::warp_size * chunk_size].roll(po2, dims=2) * in_mask
            )
            jac[:, :, warp_size * chunk_size - 1::warp_size * chunk_size] *= (
                    - jac[:, :, warp_size * chunk_size - 1::warp_size * chunk_size].roll(po2, dims=2) * in_mask +
                    (~in_mask)
            )
        # - substituting sol of last eq of prev warp in last eqs of each chunk within each warp
        for i in range(warp_size - 1):
            rhs[:, :, chunk_size * warp_size + chunk_size * (i + 1) - 1::chunk_size * warp_size] -= (
                    jac[:, :, chunk_size * warp_size + chunk_size * (i + 1) - 1::chunk_size * warp_size] *
                    rhs[:, :, chunk_size * warp_size - 1:-1:chunk_size * warp_size]
            )
            jac[:, :, chunk_size * warp_size + chunk_size * (i + 1) - 1::chunk_size * warp_size] *= - jac[:, :,
                                                                                                         chunk_size * warp_size - 1:-1:chunk_size * warp_size]
        # - substitution within chunk
        for i in range(chunk_size - 1):
            rhs[:, :, chunk_size + i::chunk_size] -= (
                    jac[:, :, chunk_size + i::chunk_size] *
                    rhs[:, :, chunk_size - 1:-1:chunk_size]
            )
            jac[:, :, chunk_size + i::chunk_size] *= - jac[:, :, chunk_size - 1:-1:chunk_size]
        
        return rhs
    
    @staticmethod
    def expand_jacobians(jacobians: torch.Tensor, jac_type: str = 'dense') -> torch.Tensor:
        """
        Expand a simplified structure jacobian to match its dense version (mostly for debugging)
        """
        
        if jac_type == 'dense':
            return jacobians
        
        elif jac_type == 'diag':
            identity = torch.eye(jacobians.shape[-1], device=jacobians.device)
            return torch.einsum( '...i,ij->...ij', (jacobians, identity) )
        
        elif jac_type == 'block_diag':
            num_blocks = jacobians.shape[-1]
            block_size = math.ceil(jacobians.shape[-2] / num_blocks)
            
            piled_identities = torch.cat([torch.eye(block_size, device=jacobians.device)]*num_blocks, dim=0)
            expanded_jacobians = torch.einsum(
                '...ik,ij->...kji', (jacobians, piled_identities)
            ).reshape([*jacobians.shape[:-2],jacobians.shape[-2],jacobians.shape[-2]]).transpose(-1,-2)
            return expanded_jacobians
        
        else:
            raise ValueError(f"Unrecognized option '{jac_type}'. Available are: 'dense', 'diag', 'block_diag'.")
    
    @staticmethod
    def contract_jacobians(jacobians: torch.Tensor, jac_type: str = 'dense') -> torch.Tensor:
        """
        Contract a dense jacobian to its simplified structure version (mostly for debugging)
        (dual of expand_jacobians)
        """
        
        if jac_type == 'dense':
            return jacobians
        
        elif jac_type == 'diag':
            return torch.einsum('...ii->...i', jacobians)
        
        elif jac_type == 'block_diag':
            # You're better off just expanding the other, if you want to compare against it
            raise NotImplementedError
            
        else:
            raise ValueError(f"Unrecognized option '{jac_type}'.")
        
    @staticmethod
    @torch._dynamo.allow_in_graph
    def parallel_reduce_diag_cuda(jac: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        with torch.autocast('cuda', enabled=False):
            dtype = jac.dtype
            jac = jac.to(torch.float32)
            rhs = rhs.to(torch.float32)
            return torch.ops.parallel_reduce_cuda.parallel_reduce_diag_cuda(jac, rhs).to(dtype)

    @staticmethod
    @torch._dynamo.allow_in_graph
    def parallel_reduce_block_diag_2x2_cuda(jac: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        with torch.autocast('cuda', enabled=False):
            dtype = jac.dtype
            jac = jac.to(torch.float32)
            rhs = rhs.to(torch.float32)
            return torch.ops.parallel_reduce_cuda.parallel_reduce_block_diag_2x2_cuda(jac, rhs).to(dtype)
    
    @staticmethod
    @torch._dynamo.allow_in_graph
    def parallel_reduce_block_diag_3x3_cuda(jac: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        with torch.autocast('cuda', enabled=False):
            dtype = jac.dtype
            jac = jac.to(torch.float32)
            rhs = rhs.to(torch.float32)
            return torch.ops.parallel_reduce_cuda.parallel_reduce_block_diag_3x3_cuda(jac, rhs).to(dtype)
