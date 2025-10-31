#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import time
from functools import partial
import typing as typ

import torch

from pararnn.rnn_cell.rnn_cell import BaseRNNCell
from pararnn.rnn_cell.rnn_cell_utils import Config
from pararnn.rnn_cell.rnn_cell_application import RNNCellApplicationMode
from pararnn.parallel_reduction.parallel_reduction import ParallelSolve, NewtonConfig
from pararnn.rnn_cell.gru_diag_mh import GRUDiagMHConfig, GRUDiagMH

torch.manual_seed(42)


def sequential_vs_parallel(
        model_type: BaseRNNCell = typ.Type[GRUDiagMH],
        model_config_type: Config = typ.Type[GRUDiagMHConfig],
        seq_length: int = 256,
        device: str = 'cpu'
):
    """
    Check that parallel and sequential fwd/bwd application of RNN match
    NB: expect errors growing as eps * seq_length, due to numerical approx
    """

    print('Computing parallel')
    newton_config = NewtonConfig()
    config = model_config_type(state_dim=14, input_dim=6, device=device, newton_config=newton_config, mode='parallel')
    x = torch.randn([3, seq_length, config.input_dim], device=device, requires_grad=True)
    model = model_type(config)
    model.zero_grad()
    t = time.time()
    y_par = model(x)
    l_par = torch.sum(y_par ** 2)
    elapsed_fwd_par = time.time() - t
    print('Parallel fwd pass computed in ', elapsed_fwd_par)
    
    t = time.time()
    l_par.backward()
    elapsed_bwd_par = time.time() - t
    print('Parallel bwd pass computed in ', elapsed_bwd_par)
    model_gradients_par = [None if param.grad is None else param.grad.detach().clone() for param in model.parameters()]
    names = [name for name,_ in model.named_parameters()]
    x_grad_par = x.grad.detach().clone()
    x.grad = None
    model.zero_grad()

    
    print('Computing sequential')
    model.mode = RNNCellApplicationMode.SEQUENTIAL
    t = time.time()
    y_seq = model(x)
    l_seq = torch.sum(y_seq ** 2)
    elapsed_fwd_seq = time.time() - t
    print('Sequential fwd pass computed in ', elapsed_fwd_seq)
    print('Speedup ', elapsed_fwd_seq / elapsed_fwd_par, 'x')
    
    t = time.time()
    l_seq.backward()
    elapsed_bwd_seq = time.time() - t
    print('Sequential bwd pass computed in ', elapsed_bwd_seq)
    print('Speedup ', elapsed_bwd_seq / elapsed_bwd_par, 'x')
    
    model_gradients_seq = [None if param.grad is None else param.grad.detach().clone() for param in model.parameters()]
    x_grad_seq = x.grad.detach().clone()
    x.grad = None
    model.zero_grad()
    
    y_par_cuda = None
    model_gradients_par_cuda = None
    x_grad_par_cuda = None
    
    if device == 'cuda':
        print('Computing parallel CUDA')
        model.mode = RNNCellApplicationMode.PARALLEL_CUDA
        t = time.time()
        y_par_cuda = model(x)
        l_par_cuda = torch.sum(y_par_cuda ** 2)
        elapsed_fwd_par_cuda = time.time() - t
        print('Parallel CUDA fwd pass computed in ', elapsed_fwd_par_cuda)
        print('Speedup (wrt seq)', elapsed_fwd_seq / elapsed_fwd_par_cuda, 'x')
        print('Speedup (wrt torch parallel_reduction)', elapsed_fwd_par / elapsed_fwd_par_cuda, 'x')

        t = time.time()
        l_par_cuda.backward()
        elapsed_bwd_par_cuda = time.time() - t
        print('Parallel CUDA bwd pass computed in ', elapsed_bwd_par_cuda)
        print('Speedup (wrt seq)', elapsed_bwd_seq / elapsed_bwd_par_cuda, 'x')
        print('Speedup (wrt torch parallel_reduction)', elapsed_bwd_par / elapsed_bwd_par_cuda, 'x')

        model_gradients_par_cuda = [None if param.grad is None else param.grad.detach().clone() for param in model.parameters()]
        x_grad_par_cuda = x.grad.detach().clone()
        x.grad = None
        

    print('Output comparison:\n||y_par - y_seq||_inf = ', torch.max(torch.abs(y_par - y_seq)).item())
    if device == 'cuda':
        print('Output comparison:\n||y_par_cuda - y_seq||_inf = ', torch.max(torch.abs(y_par_cuda - y_seq)).item())
    
    print('Gradients comparison:\n||gradW_par - gradW_seq||_inf =')
    for i in range(len(model_gradients_par)):
        if model_gradients_par[i] is not None:
            print(names[i], ': ', torch.max(torch.abs(model_gradients_par[i] - model_gradients_seq[i])).item())
        else:
            print(names[i], ': requires no grad tracking')
    if device == 'cuda':
        print('Gradients comparison:\n||gradW_par_cuda - gradW_seq||_inf =')
        for i in range(len(model_gradients_par_cuda)):
            if model_gradients_par[i] is not None:
                print(names[i], ': ', torch.max(torch.abs(model_gradients_par_cuda[i] - model_gradients_seq[i])).item())
            else:
                print(names[i], ': requires no grad tracking')

    print('||gradX - gradX_seq||_inf =', torch.max(torch.abs(x_grad_par - x_grad_seq)))
    if device == 'cuda':
        print('||gradX_cuda - gradX_seq||_inf =', torch.max(torch.abs(x_grad_par_cuda - x_grad_seq)))


def check_jacobians(
        model_type: BaseRNNCell = typ.Type[GRUDiagMH],
        model_config_type: Config = typ.Type[GRUDiagMHConfig],
        structure_type: str = 'dense'
):
    """
    Check that autograd and own implementation of Jacobians match
    """

    config = model_config_type(state_dim=2, input_dim=4, mode='parallel')
    x = torch.randn([2, 10, config.input_dim])
    model = model_type(config)
    h = [torch.zeros(x.shape[0], model.state_dim)]
    for t in range(x.shape[1]):
        h.append(model._get_impl_type().recurrence_step(
                x[:,t,:],
                h[-1],
                model.system_parameters
            )
        )
    h = torch.cat([state.unsqueeze(1) for state in h], dim=1)
    
    tangents = torch.eye(h.shape[-1])
    jacobians_autograd = torch.empty(
        [x.shape[0], x.shape[1], h.shape[-1], h.shape[-1]],
    )
    
    my_recurrence_step = partial( model._get_impl_type().recurrence_step, system_parameters=model.system_parameters )
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            recursive_step_vjps = (
                torch.func.vjp(
                    my_recurrence_step, x[None, i, j], h[None, i, j]
                )[1]
            )
            recursive_step_jacobian_funcs = torch.func.vmap(
                recursive_step_vjps, in_dims=0
            )
            with torch.no_grad():
                jacobians_autograd[i, j] = - recursive_step_jacobian_funcs(tangents[:, None, ...])[1][:, 0, :]
    
    jacobians_own = ParallelSolve.expand_jacobians(
        model._get_impl_type().compute_jacobians(sol=h[:, 1:], x=x, system_parameters=model.system_parameters),
        jac_type=structure_type
    )
    
    print('||J_aut - J_own||_inf = ', torch.max(torch.abs(jacobians_autograd - jacobians_own)))


    