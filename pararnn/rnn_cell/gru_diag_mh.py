#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from dataclasses import dataclass
import typing as typ
from pathlib import Path

import torch
import torch.nn as nn
import math


from pararnn.rnn_cell.rnn_cell import BaseRNNCell
from pararnn.rnn_cell.rnn_cell_impl import RNNCellDiagImpl
from pararnn.rnn_cell.rnn_cell_utils import SystemParameters, Config
from pararnn.utils.init import INIT_REGISTRY

'''
Implementation of diagonal version of Fully Gated Recurrent Unit, see https://en.wikipedia.org/wiki/Gated_recurrent_unit
splitting x into multiple heads
Main recursion step, starting from h=0:

z =     sigma_z(diag(Az) h + Bz x + bz)
r =     sigma_r(diag(Ar) h + Br x + br)
h_new = sigma_h(diag(Ah) (h*r) + Bh x + bh)

h = (1-z) h + z h_new

But with the various B's considered block-diag, with num_head blocks of dims head_dim_state x head_dim_input
'''

LIB_PATH = str(next(iter(Path(__file__).resolve().parent.parent.parent.glob("*parallel_reduce_cuda.cpython*so"))))
torch.ops.load_library(LIB_PATH)


@torch.library.register_fake("parallel_reduce_cuda::fused_fwd_gru_diag_mh")
def _(AT, BxpbT):
    torch._check(AT.shape[-1] == BxpbT.shape[-1])
    torch._check(AT.shape[0] == BxpbT.shape[-3])
    return torch.empty_like(BxpbT[...,0])

@torch.library.register_fake("parallel_reduce_cuda::fused_bwd_gru_diag_mh")
def _(gradientT, hT, AT, BxpbT):
    torch._check(AT.shape[-1] == BxpbT.shape[-1])
    torch._check(AT.shape[0] == BxpbT.shape[-3])
    torch._check(hT.shape[-1] == BxpbT.shape[-2])
    torch._check(gradientT.shape == hT.shape)
    return torch.empty_like(hT)


T = typ.TypeVar("T")


@dataclass(frozen=True)
class GRUDiagMHTrait:
    pass


@dataclass
class GRUDiagMHConfig(Config[GRUDiagMHTrait]):
    nonlin_update: str = 'sigmoid'
    nonlin_reset: str = 'sigmoid'
    nonlin_state: str = 'tanh'
    num_heads: int = 2
    a_init_fn: str = "xlstm"
    b_init_fn: str = "bias_minus_linspace"
    w_init_fn: str = "xavier_uniform"



@dataclass
class GRUDiagMHSystemParameters(SystemParameters[GRUDiagMHTrait]):
    A: torch.Tensor
    B: torch.Tensor
    b: torch.Tensor
    nonlin_update: typ.Callable
    nonlin_reset: typ.Callable
    nonlin_state: typ.Callable
    derivative_nonlin_update: typ.Callable
    derivative_nonlin_reset: typ.Callable
    derivative_nonlin_state: typ.Callable
    
    def unpack(self) -> typ.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        typ.Callable, typ.Callable, typ.Callable,
        typ.Callable, typ.Callable, typ.Callable
    ]:
        return (
            self.A, self.B, self.b,
            self.nonlin_update, self.nonlin_reset, self.nonlin_state,
            self.derivative_nonlin_update, self.derivative_nonlin_reset, self.derivative_nonlin_state
        )
    
    @classmethod
    def repack(
            cls: T,
            pars: typ.Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor,
                typ.Callable, typ.Callable, typ.Callable,
                typ.Callable, typ.Callable, typ.Callable
            ]
    ) -> T:
        return GRUDiagMHSystemParameters(
            A=pars[0],
            B=pars[1],
            b=pars[2],
            nonlin_update=pars[3],
            nonlin_reset=pars[4],
            nonlin_state=pars[5],
            derivative_nonlin_update=pars[6],
            derivative_nonlin_reset=pars[7],
            derivative_nonlin_state=pars[8]
        )


class GRUDiagMHImpl(RNNCellDiagImpl[GRUDiagMHSystemParameters]):
    
    @staticmethod
    def recurrence_step(
            x: torch.Tensor,
            h: torch.Tensor,
            system_parameters: GRUDiagMHSystemParameters
    ) -> torch.Tensor:
        num_heads = system_parameters.B.shape[0]
        Bxpb = torch.einsum(
            '...hi,hivj->...vhj',
            (x.view([*x.shape[:-1], num_heads, -1]), system_parameters.B)
        ).flatten(-2) + system_parameters.b
        z, r = torch.unbind(
            torch.einsum('vj,...j->...vj', (system_parameters.A[:2, :], h)) + Bxpb[..., :2, :],
            dim=-2
        )
        z = system_parameters.nonlin_update(z)
        r = system_parameters.nonlin_reset(r)
        h_new = system_parameters.nonlin_state(
            system_parameters.A[2, :] * h * r + Bxpb[..., 2, :]
        )
        
        return z * h_new + (1 - z) * h
    
    @classmethod
    def assemble_initial_guess(
            cls,
            x: torch.Tensor,
            state_dim: int,
            system_parameters: GRUDiagMHSystemParameters
    ) -> torch.Tensor:
        # improve initial guess: what you'd get if state was 0 everywhere
        h = torch.zeros([*x.shape[:-1], state_dim], device=x.device, dtype=x.dtype)
        return cls.recurrence_step(x, h, system_parameters)
    
    @classmethod
    def _compute_jacobians_inner(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: GRUDiagMHSystemParameters,
    ) -> torch.Tensor:
        h_prev = cls._roll_state(sol)
        num_heads = system_parameters.B.shape[0]

        Bxpb = torch.einsum(
            '...hi,hivj->...vhj',
            (x.view([*x.shape[:-1], num_heads, -1]), system_parameters.B)
        ).flatten(-2) + system_parameters.b
        pre_nl_z, pre_nl_r = torch.unbind(
            torch.einsum('vj,...j->...vj', (system_parameters.A[:2, :], h_prev)) + Bxpb[..., :2, :],
            dim=-2
        )
        z = system_parameters.nonlin_update(pre_nl_z)
        r = system_parameters.nonlin_reset(pre_nl_r)
        pre_nl_h = system_parameters.A[2, :] * h_prev * r + Bxpb[..., 2, :]
        h = system_parameters.nonlin_state(pre_nl_h)
        
        grad_z = system_parameters.derivative_nonlin_update(pre_nl_z)
        grad_r = system_parameters.derivative_nonlin_reset(pre_nl_r)
        grad_h = system_parameters.derivative_nonlin_state(pre_nl_h)
        
        J_z, J_r, J_h = torch.unbind(
            system_parameters.A * torch.stack([grad_z, grad_r, grad_h], dim=-2),
            dim=-2
        )
        J_h = J_h * (r + h_prev * J_r)
        
        jac = (1 - z) + (h - h_prev) * J_z + z * J_h
        
        return - jac
    
    @classmethod
    def compute_jacobians(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: GRUDiagMHSystemParameters,
    ) -> torch.Tensor:
        return cls._compute_jacobians_inner(sol, x, system_parameters)
    
    @classmethod
    def compute_jacobians_bwd(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: GRUDiagMHSystemParameters,
    ) -> torch.Tensor:
        jac = cls._compute_jacobians_inner(sol, x, system_parameters)
        jac = torch.roll(
            torch.flip(
                jac,
                dims=[-2]
            ),
            shifts=1,
            dims=-2
        )
        jac[..., 0, :] = 0
        
        return jac
    
    @classmethod
    def backprop_post_processing(
            cls,
            grad_y: torch.Tensor,
            x: torch.Tensor,
            h: torch.Tensor,
            system_parameters: GRUDiagMHSystemParameters
    ) -> typ.Tuple[typ.Optional[torch.Tensor], ...]:
        return (
            grad_y,
            torch.zeros_like(x),
            torch.zeros_like(system_parameters.A),
            torch.zeros_like(system_parameters.B),
            torch.zeros_like(system_parameters.b),
            None, None, None,
            None, None, None
        )
    
    @classmethod
    def backprop_to_system_parameters(
            cls,
            dl_dht: torch.Tensor,
            x: torch.Tensor,
            h: torch.Tensor,
            system_parameters: GRUDiagMHSystemParameters
    ) -> typ.Tuple[typ.Union[torch.Tensor, None], ...]:
        h_prev = cls._roll_state(h)
        num_heads = system_parameters.B.shape[0]

        Bxpb = torch.einsum(
            '...hi,hivj->...vhj',
            (x.view([*x.shape[:-1], num_heads, -1]), system_parameters.B)
        ).flatten(-2) + system_parameters.b
        pre_nl_z, pre_nl_r = torch.unbind(
            torch.einsum('vj,...j->...vj', (system_parameters.A[:2, :], h_prev)) + Bxpb[..., :2, :],
            dim=-2
        )
        z = system_parameters.nonlin_update(pre_nl_z)
        r = system_parameters.nonlin_reset(pre_nl_r)
        pre_nl_h = system_parameters.A[2, :] * h_prev * r + Bxpb[..., 2, :]
        h_new = system_parameters.nonlin_state(pre_nl_h)
        
        grad_h = dl_dht * z * system_parameters.derivative_nonlin_state(pre_nl_h)
        grad_z = dl_dht * (h_new - h_prev) * system_parameters.derivative_nonlin_update(pre_nl_z)
        grad_r = grad_h * system_parameters.A[2, :] * h_prev * system_parameters.derivative_nonlin_reset(pre_nl_r)
        
        grad_zrh = torch.stack([grad_z, grad_r, grad_h], dim=-2)
        
        grad_b = torch.sum(grad_zrh, dim=tuple(range(grad_zrh.ndim - 2)))
        grad_B = torch.einsum(
            '...vhi,...hj->hjvi',
            (grad_zrh.view([*grad_zrh.shape[:-1],num_heads,-1]), x.view([*x.shape[:-1],num_heads,-1]))
        )
        grad_x = torch.einsum(
            '...vhi,hjvi->...hj',
            (grad_zrh.view([*grad_zrh.shape[:-1],num_heads,-1]), system_parameters.B)
        ).flatten(-2)
        
        grad_zrh[..., 2, :] = grad_zrh[..., 2, :] * r
        grad_A = torch.einsum('...vi,...i->vi', (grad_zrh, h_prev))
        
        return (
            grad_x,
            grad_A, grad_B, grad_b,
            None, None, None,
            None, None, None
        )
    
    @classmethod
    def fused_parallel_forward(
            cls,
            x: torch.Tensor,
            system_parameters: GRUDiagMHSystemParameters,
    ) -> torch.Tensor:
        num_heads = system_parameters.B.shape[0]
        
        Bxpb = torch.einsum(
            '...hi,hivj->...vhj',
            (x.view([*x.shape[:-1], num_heads, -1]), system_parameters.B)
        ).flatten(-2) + system_parameters.b
        
        h = cls._fused_parallel_fwd(
            system_parameters.A.transpose(-1, -2).contiguous(),
            torch.einsum('...tvj->...jtv', Bxpb).contiguous()
        ).transpose(-1, -2)
        
        return h
    
    @classmethod
    def fused_parallel_backward(
            cls,
            gradient: torch.Tensor,
            h: torch.Tensor,
            x: torch.Tensor,
            system_parameters: GRUDiagMHSystemParameters,
    ) -> torch.Tensor:
        num_heads = system_parameters.B.shape[0]
        
        Bxpb = torch.einsum(
            '...hi,hivj->...vhj',
            (x.view([*x.shape[:-1], num_heads, -1]), system_parameters.B)
        ).flatten(-2) + system_parameters.b
        
        dl_dht = cls._fused_parallel_bwd(
            gradient.transpose(-1, -2).contiguous(),
            h.transpose(-1, -2).contiguous(),
            system_parameters.A.transpose(-1, -2).contiguous(),
            torch.einsum('...tvj->...jtv', Bxpb).contiguous()
        ).transpose(-1, -2)
        
        return dl_dht
    
    @staticmethod
    @torch._dynamo.allow_in_graph
    def _fused_parallel_fwd(AT: torch.Tensor, BxpbT: torch.Tensor) -> torch.Tensor:
        with torch.autocast('cuda', enabled=False):
            dtype = AT.dtype
            return torch.ops.parallel_reduce_cuda.fused_fwd_gru_diag_mh(AT.to(torch.float32),
                                                                        BxpbT.to(torch.float32)).to(dtype)
    
    @staticmethod
    @torch._dynamo.allow_in_graph
    def _fused_parallel_bwd(
            gradientT: torch.Tensor,
            hT: torch.Tensor,
            AT: torch.Tensor,
            BxpbT: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast('cuda', enabled=False):
            dtype = AT.dtype
            return torch.ops.parallel_reduce_cuda.fused_bwd_gru_diag_mh(
                gradientT.to(torch.float32),
                hT.to(torch.float32),
                AT.to(torch.float32),
                BxpbT.to(torch.float32)
            ).to(dtype)
    

class GRUDiagMH(BaseRNNCell[GRUDiagMHConfig, GRUDiagMHSystemParameters, GRUDiagMHImpl]):
    
    def __init__(self, config: GRUDiagMHConfig):
        super().__init__(config)
    
    def _specific_init(self, config: GRUDiagMHConfig):

        assert self.input_dim % config.num_heads == 0, "Number of heads must exactly divide input dimension"
        assert self.state_dim % config.num_heads == 0, "Number of heads must exactly divide state dimension"

        self.num_heads = config.num_heads
        self.head_input_dim = math.ceil(self.input_dim / config.num_heads)
        self.head_state_dim = math.ceil(self.state_dim / config.num_heads)

        # System parameters
        # - collated z,r,h
        self.A = nn.Parameter(torch.empty([3, self.state_dim], device=self.device, dtype=self.dtype))
        self.B = nn.Parameter(torch.empty([self.num_heads, self.head_input_dim, 3, self.head_state_dim], device=self.device, dtype=self.dtype))
        self.b = nn.Parameter(torch.empty([3, self.state_dim], device=self.device, dtype=self.dtype))
        self.nonlin_update, self.derivative_nonlin_update = self._set_nonlinearity_and_derivative(config.nonlin_update)
        self.nonlin_reset, self.derivative_nonlin_reset = self._set_nonlinearity_and_derivative(config.nonlin_reset)
        self.nonlin_state, self.derivative_nonlin_state = self._set_nonlinearity_and_derivative(config.nonlin_state)
        
        self.reset_parameters()

    @property
    def _system_parameters(self):
        # - handy class to collect them all
        return GRUDiagMHSystemParameters(
            A=self.A,
            B=self.B,
            b=self.b,
            nonlin_update=self.nonlin_update,
            nonlin_reset=self.nonlin_reset,
            nonlin_state=self.nonlin_state,
            derivative_nonlin_update=self.derivative_nonlin_update,
            derivative_nonlin_reset=self.derivative_nonlin_reset,
            derivative_nonlin_state=self.derivative_nonlin_state,
        )

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        INIT_REGISTRY[self._config.a_init_fn](self.A.data, fan_in=self._config.state_dim, fan_out=None)
        INIT_REGISTRY[self._config.w_init_fn](self.B.data, fan_in=self.head_input_dim, fan_out=self.state_dim)
        INIT_REGISTRY[self._config.b_init_fn](self.b.data, fan_in=None, fan_out=self.b.numel())