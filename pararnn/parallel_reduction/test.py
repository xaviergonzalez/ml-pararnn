#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import torch
import math
import time

import typing as typ

from pararnn.parallel_reduction.parallel_reduction import ParallelSolve
import pararnn.parallel_reduction.utils as pr_utils


def parallel_reduction_torch_vs_cuda_diag(
        seq_length: typ.Optional[int] = None,
        batch_size: int = 3,
        hidden_dim: int = 2,
        dtype: torch.dtype = torch.float32,
        verbose: bool = True
) -> bool:
    chunk_size = pr_utils.get_diag_chunk_size(dtype)
    warp_size = pr_utils.get_threads_per_warp()
    if seq_length is None:
        seq_length = chunk_size * pr_utils.get_threads_per_block()
    block_size = math.ceil(seq_length / chunk_size)
    
    # pick jacobians randomly btw {-1,1}: this is to ensure that the hidden state doesn't explode/collapse for long seq
    jac = torch.randint( 0, 2, [batch_size,hidden_dim,seq_length], device='cuda', dtype=dtype ) * 2. - 1.
    # jac = -torch.ones( [batch_size,hidden_dim,seq_length], device='cuda' )
    rhs = torch.zeros_like( jac )
    rhs[:,:,0] = 1.
    # jac = torch.rand( [batch_size,hidden_dim,seq_length], device='cuda' )
    # rhs = torch.rand_like( jac )
    jac[:,:,0] = 0.
    
    jac_cuda = jac.clone()
    rhs_cuda = rhs.clone()
    
    jac_torch = jac.clone().transpose(-1,-2).contiguous()
    rhs_torch = rhs.clone().transpose(-1,-2).contiguous()
    jac_seq = jac.clone()
    rhs_seq = rhs.clone()

    my_jac = jac.clone()
    my_rhs = rhs.clone()
    
    if verbose:
        print('Computing parallel reduction in CUDA')
    t = time.time()
    sol_cuda = ParallelSolve.parallel_reduce_diag_cuda(jac_cuda, rhs_cuda)
    elapsed_cuda = time.time() - t
    if verbose:
        print('CUDA parallel reduction computed in ', elapsed_cuda)
        print('Computing parallel reduction in torch')
    t = time.time()
    sol_torch = ParallelSolve.parallel_reduce(jac_torch, rhs_torch, ParallelSolve._reduction_step_diag)
    elapsed_torch = time.time() - t
    sol_torch = sol_torch.transpose(-1, -2)
    if verbose:
        print('Torch parallel reduction computed in ', elapsed_torch)
        print('Computing sequential in torch')
    sol_seq = torch.zeros_like(rhs)
    t = time.time()
    sol_seq[:, :, 0] = rhs_seq[:,:,0]
    for i in range(1, sol_seq.shape[-1]):
        sol_seq[:,:,i] = rhs_seq[:,:,i] - jac_seq[:,:,i] * sol_seq[:,:,i-1]
    elapsed_seq = time.time() - t
    if verbose:
        print('Torch sequential computed in ', elapsed_seq)
        print('Speedup CUDA/torch: ', elapsed_torch / elapsed_cuda, 'x')
        print('Speedup par/seq: ',    elapsed_seq   / elapsed_torch, 'x')

        # mimic action of cuda kernel here:
        if seq_length % chunk_size == 0:
            t = time.time()
            # - thomas reduction
            for i in range(1, chunk_size):
                my_rhs[:,:,i::chunk_size] -= my_jac[:,:,i::chunk_size] * my_rhs[:,:,i-1:-1:chunk_size]
                my_jac[:,:,i::chunk_size] *= - my_jac[:,:,i-1:-1:chunk_size]
            # - parallel reduction of last eqs in chunk:
            #   - within warp
            for i in range(0, math.ceil(math.log2(warp_size))):
                po2 = 2**i
                in_mask = (torch.arange(min(warp_size,block_size), device='cuda') >= po2).repeat(
                    [math.ceil(block_size / warp_size)]).unsqueeze(0).unsqueeze(0)
                my_rhs[:,:,chunk_size-1::chunk_size] -= (
                        my_jac[:,:,chunk_size-1::chunk_size] *
                        my_rhs[:,:,chunk_size-1::chunk_size].roll(po2,dims=2) * in_mask
                )
                my_jac[:,:,chunk_size-1::chunk_size] *= (
                        - my_jac[:,:,chunk_size-1::chunk_size].roll(po2,dims=2) * in_mask +
                        (~in_mask)
                )
            # - parallel reduction of last eqs in chunk:
            #   - within block
            for i in range(0, math.ceil(math.log2(block_size / warp_size))):
                po2 = 2**i
                in_mask = (torch.arange(math.ceil(block_size / warp_size), device='cuda') >= po2).unsqueeze(0).unsqueeze(0)
                my_rhs[:,:,warp_size*chunk_size-1::warp_size*chunk_size] -= (
                        my_jac[:,:,warp_size*chunk_size-1::warp_size*chunk_size] *
                        my_rhs[:,:,warp_size*chunk_size-1::warp_size*chunk_size].roll(po2,dims=2) * in_mask
                )
                my_jac[:,:,warp_size*chunk_size-1::warp_size*chunk_size] *= (
                    - my_jac[:,:,warp_size*chunk_size-1::warp_size*chunk_size].roll(po2,dims=2) * in_mask +
                    (~in_mask)
                )
            # - substituting sol of last eq of prev warp in last eqs of each chunk within each warp
            for i in range(warp_size-1):
                my_rhs[:,:,chunk_size*warp_size+chunk_size*(i+1)-1::chunk_size*warp_size] -= (
                        my_jac[:,:,chunk_size*warp_size+chunk_size*(i+1)-1::chunk_size*warp_size] *
                        my_rhs[:,:,chunk_size*warp_size-1:-1:chunk_size*warp_size]
                )
                my_jac[:,:,chunk_size*warp_size+chunk_size*(i+1)-1::chunk_size*warp_size] *= - my_jac[:,:,chunk_size*warp_size-1:-1:chunk_size*warp_size]
            # - substitution within chunk
            for i in range(chunk_size-1):
                my_rhs[:,:,chunk_size+i::chunk_size] -= (
                        my_jac[:,:,chunk_size+i::chunk_size] *
                        my_rhs[:,:,chunk_size-1:-1:chunk_size]
                )
                my_jac[:,:,chunk_size+i::chunk_size] *= - my_jac[:,:,chunk_size-1:-1:chunk_size]
            elapsed_pseudo_CUDA = time.time() - t
            
            print('Pseudo_CUDA computed in ', elapsed_pseudo_CUDA)
            print('Speedup pseudo_CUDA/CUDA: ', elapsed_pseudo_CUDA / elapsed_cuda, 'x')
            
            print('Error pseudo_CUDA-CUDA: ', torch.max(torch.abs(my_rhs-sol_cuda)).item())
        else:
            print('seq_length', seq_length, 'must be divisible by chunk_size', chunk_size,'to compare against pseudo-CUDA')

        print('Error CUDA-torch (parallel reduction): ', torch.max(torch.abs(sol_cuda-sol_torch)).item())
        print('Error par-seq: ', torch.max(torch.abs(sol_torch-sol_seq)).item())
    
    return torch.max(torch.abs(sol_cuda-sol_torch)).item() == 0
    

def parallel_reduction_torch_vs_cuda_block_diag(
        seq_length: typ.Optional[int] = None,
        batch_size: int = 3,
        hidden_dim: int = 2,
        dtype: torch.dtype = torch.float32,
        jac_randomisation_type: str = 'rotation',  # 'rotation', 'sign_flip', or 'components_swap'
        verbose: bool = True
) -> bool:
    num_hidden_vars = 2     # Just test for 2x2 blocks
    chunk_size = pr_utils.get_block_diag_2x2_chunk_size(dtype)
    warp_size = pr_utils.get_threads_per_warp()
    if seq_length is None:
        seq_length = chunk_size * pr_utils.get_threads_per_block()
    block_size = math.ceil(seq_length / chunk_size)

    
    # we must be careful how we pick the jacobians: if their norm is ><1, then the hidden state explodes/collapses,
    # and for long sequences the comparison is meaningless
    if jac_randomisation_type == 'rotation':
        # Jacobians are random rotation matrices: [cos(theta), -sin(theta); sin(theta), cos(theta)]
        theta = 2. * torch.pi * torch.rand([batch_size,hidden_dim,seq_length], device='cuda', dtype=torch.float64 )
        jac = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)], dim=-1 ),
                           torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)], dim=-2).to(dtype)
        # jac_tot = torch.stack([torch.stack([torch.cos(torch.sum(theta,dim=-1)), -torch.sin(torch.sum(theta,dim=-1))], dim=-1 ),
        #                        torch.stack([torch.sin(torch.sum(theta,dim=-1)), torch.cos(torch.sum(theta,dim=-1))], dim=-1)], dim=-2)
        rhs = torch.zeros(jac.shape[:-1], device='cuda', dtype=dtype)
        rhs[...,0,:] = 1.
    elif jac_randomisation_type == 'sign_flip':
        # Jacobians do sign flip: [+-1, 0; 0, +-1]
        diags = torch.randint( 0, 2, [batch_size, hidden_dim, seq_length, num_hidden_vars], device='cuda', dtype=dtype ) * 2. - 1.
        jac = torch.zeros([batch_size, hidden_dim, seq_length, num_hidden_vars, num_hidden_vars], device='cuda', dtype=dtype)
        for i in range(num_hidden_vars):
            jac[...,i,i] = diags[...,i]
        rhs = torch.randint(0, 3, jac.shape[:-1], device='cuda', dtype=dtype) - 1.
    elif jac_randomisation_type == 'components_swap':
        # Jacobians do components swap and sign flip: [0, +-1; +-1, 0]
        diags = torch.randint( 0, 2, [batch_size, hidden_dim, seq_length, num_hidden_vars], device='cuda', dtype=dtype ) * 2. - 1.
        jac = torch.zeros([batch_size, hidden_dim, seq_length, num_hidden_vars, num_hidden_vars], device='cuda', dtype=dtype)
        jac[...,0,1] = diags[...,0]
        jac[...,1,0] = diags[...,1]
        rhs = torch.randint(0, 3, jac.shape[:-1], device='cuda', dtype=dtype) - 1.
    else:
        raise ValueError("jac_randomisation_type not recognised")
    
    jac[...,0,:,:] = 0.

    jac_cuda = jac.clone()
    rhs_cuda = rhs.clone()
    
    jac_torch = torch.cat(jac.clone().transpose(-3,-4).unbind(dim=-2),dim=-2).contiguous()
    rhs_torch = rhs.clone().permute(*range(rhs.ndim - 3), -2, -1, -3).flatten(-2).contiguous()

    jac_seq = jac.clone()
    rhs_seq = rhs.clone()
    
    my_jac = jac.clone()
    my_rhs = rhs.clone()
    
    if verbose:
        print('Computing parallel reduction in CUDA')
    t = time.time()
    sol_cuda = ParallelSolve.parallel_reduce_block_diag_2x2_cuda(jac_cuda, rhs_cuda)
    elapsed_cuda = time.time() - t
    if verbose:
        print('CUDA parallel reduction computed in ', elapsed_cuda)
        print('Computing parallel reduction in torch')
    t = time.time()
    sol_torch = ParallelSolve.parallel_reduce(jac_torch, rhs_torch, ParallelSolve._reduction_step_block_diag)
    elapsed_torch = time.time() - t
    sol_torch = sol_torch.reshape([batch_size,seq_length,num_hidden_vars,hidden_dim]).transpose(-1,-2).transpose(1,2)
    if verbose:
        print('Torch parallel reduction computed in ', elapsed_torch)
        print('Computing sequential in torch')
    sol_seq = torch.zeros_like(rhs)
    t = time.time()
    sol_seq[:, :, 0] = rhs_seq[:, :, 0]
    for i in range(1, sol_seq.shape[-2]):
        sol_seq[:, :, i] = rhs_seq[:, :, i] - torch.einsum('bnij,bnj->bni', (jac_seq[:,:,i], sol_seq[:,:,i-1]))
    elapsed_seq = time.time() - t
    if verbose:
        print('Torch sequential computed in ', elapsed_seq)
        print('Speedup CUDA/torch: ', elapsed_torch / elapsed_cuda, 'x')
        print('Speedup par/seq: ', elapsed_seq / elapsed_torch, 'x')
    
        # mimic action of cuda kernel here:
        if seq_length % chunk_size == 0:
            t = time.time()
            # - thomas reduction
            for i in range(1, chunk_size):
                my_rhs[:,:,i::chunk_size] -= torch.einsum('bntij,bntj->bnti', (my_jac[:,:,i::chunk_size], my_rhs[:,:,i-1:-1:chunk_size]))
                my_jac[:, :, i::chunk_size] = -torch.einsum('bntik,bntkj->bntij', (my_jac[:,:,i::chunk_size], my_jac[:,:,i-1:-1:chunk_size]))
            # - parallel_reduction reduction of last eqs in chunk:
            #   - within warp
            for i in range(0, math.ceil(math.log2(warp_size))):
                po2 = 2**i
                in_mask = (torch.arange(min(warp_size,block_size), device='cuda') >= po2).repeat(
                    [math.ceil(block_size / warp_size)]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                my_rhs[:,:,chunk_size-1::chunk_size] -= torch.einsum(
                    'bntij,bntj->bnti',
                    (
                        my_jac[:,:,chunk_size-1::chunk_size],
                        my_rhs[:,:,chunk_size-1::chunk_size].roll(po2,dims=2) * in_mask
                    )
                )
                my_jac[:,:,chunk_size-1::chunk_size] = - torch.einsum(
                    'bntik,bntkj->bntij',
                    (
                        my_jac[:,:,chunk_size-1::chunk_size],
                        my_jac[:,:,chunk_size-1::chunk_size].roll(po2,dims=2) * in_mask.unsqueeze(-1)
                    )
                ) + my_jac[:,:,chunk_size-1::chunk_size] * (~in_mask.unsqueeze(-1))
        
            # - parallel reduction of last eqs in chunk:
            #   - within block
            for i in range(0, math.ceil(math.log2(block_size / warp_size))):
                po2 = 2**i
                in_mask = (torch.arange(math.ceil(block_size / warp_size), device='cuda') >= po2).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                my_rhs[:,:,warp_size*chunk_size-1::warp_size*chunk_size] -= torch.einsum(
                    'bntij,bntj->bnti',
                    (
                        my_jac[:,:,warp_size*chunk_size-1::warp_size*chunk_size],
                        my_rhs[:,:,warp_size*chunk_size-1::warp_size*chunk_size].roll(po2,dims=2) * in_mask
                    )
                )
                my_jac[:,:,warp_size*chunk_size-1::warp_size*chunk_size] = - torch.einsum(
                    'bntik,bntkj->bntij',
                    (
                        my_jac[:,:,warp_size*chunk_size-1::warp_size*chunk_size],
                        my_jac[:,:,warp_size*chunk_size-1::warp_size*chunk_size].roll(po2,dims=2) * in_mask.unsqueeze(-1)
                    )
                ) + my_jac[:,:,warp_size*chunk_size-1::warp_size*chunk_size] * (~in_mask.unsqueeze(-1))
        
            # - substituting sol of last eq of prev warp in last eqs of each chunk within each warp
            for i in range(warp_size-1):
                my_rhs[:,:,chunk_size*warp_size+chunk_size*(i+1)-1::chunk_size*warp_size] -= torch.einsum(
                    'bntij,bntj->bnti', (
                        my_jac[:,:,chunk_size*warp_size+chunk_size*(i+1)-1::chunk_size*warp_size],
                        my_rhs[:,:,chunk_size*warp_size-1:-1:chunk_size*warp_size]
                    )
                )
                my_jac[:,:,chunk_size*warp_size+chunk_size*(i+1)-1::chunk_size*warp_size] = -torch.einsum(
                    'bntik,bntkj->bntij',
                    (
                        my_jac[:,:,chunk_size*warp_size+chunk_size*(i+1)-1::chunk_size*warp_size],
                        my_jac[:,:,chunk_size*warp_size-1:-1:chunk_size*warp_size]
                    )
                )
            # - substitution within chunk
            for i in range(chunk_size-1):
                my_rhs[:,:,chunk_size+i::chunk_size] -= torch.einsum(
                    'bntij,bntj->bnti', (
                        my_jac[:,:,chunk_size+i::chunk_size],
                        my_rhs[:,:,chunk_size-1:-1:chunk_size]
                    )
                )
                my_jac[:,:,chunk_size+i::chunk_size] = -torch.einsum(
                    'bntik,bntkj->bntij',
                    (
                        my_jac[:,:,chunk_size+i::chunk_size],
                        my_jac[:,:,chunk_size-1:-1:chunk_size]
                    )
                )
            elapsed_pseudo_CUDA = time.time() - t
            print('Pseudo_CUDA computed in ', elapsed_pseudo_CUDA)
            print('Speedup pseudo_CUDA/CUDA: ', elapsed_pseudo_CUDA / elapsed_cuda, 'x')
            print('Error pseudo_CUDA-CUDA: ', torch.max(torch.abs(my_rhs-sol_cuda)).item())
            print('Error pseudo_CUDA-torch: ', torch.max(torch.abs(my_rhs-sol_torch)).item())
        else:
            print('seq_length', seq_length, 'must be divisible by chunk_size', chunk_size,'to compare against pseudo-CUDA')

        print('Error CUDA-torch (parallel reduction): ', torch.max(torch.abs(sol_cuda-sol_torch)).item())
        print('Error par-seq: ', torch.max(torch.abs(sol_torch-sol_seq)).item())
    
    return torch.max(torch.abs(sol_cuda-sol_torch)).item() == 0


def parallel_reduction_torch_vs_cuda_diag_comparison_overkill(
        min_seq_length: int = 2,
        max_seq_length: int = 2**22,
        batch_size: int = 3,
        hidden_dim: int = 2,
        dtype: torch.dtype = torch.float32,
):
    for seq_length in range(min_seq_length, max_seq_length):
        correct = parallel_reduction_torch_vs_cuda_diag(seq_length=seq_length, batch_size=batch_size,
                                                        hidden_dim=hidden_dim, dtype=dtype, verbose=False)
        if not correct:
            print(f"Found an error for seq_length={seq_length}. Abort")
            break
        else:
            print(f"All good for seq_length={seq_length}")


def parallel_reduction_torch_vs_cuda_block_diag_comparison_overkill(
        min_seq_length: int = 2,
        max_seq_length: int = 2**22,
        batch_size: int = 3,
        hidden_dim: int = 2,
        dtype: torch.dtype = torch.float32,
        jac_randomisation_type: str = 'components_swap'
):
    for seq_length in range(min_seq_length, max_seq_length):
        correct = parallel_reduction_torch_vs_cuda_block_diag(seq_length=seq_length, batch_size=batch_size,
                                                              hidden_dim=hidden_dim, dtype=dtype,
                                                              jac_randomisation_type=jac_randomisation_type,
                                                              verbose=False)
        if not correct:
            print(f"Found an error for seq_length={seq_length}. Abort")
            break
        else:
            print(f"All good for seq_length={seq_length}")
