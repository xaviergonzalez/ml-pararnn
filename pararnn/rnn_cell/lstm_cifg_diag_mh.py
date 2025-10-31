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
from pararnn.rnn_cell.rnn_cell_impl import RNNCellBlockDiagImpl
from pararnn.rnn_cell.rnn_cell_utils import SystemParameters, Config
from pararnn.utils.init import INIT_REGISTRY

'''
Efficient implementation of Long Short-Term Memory Unit, with Coupled Input and Forget Gate,
see https://arxiv.org/pdf/1503.04069
But considering only diagonal state matrix A, and splitting x into multiple heads
Main recursion step, starting from h=0:

f =     sigma_f(diag(Af) h + Bf x + Cf c + bf)
c_new = sigma_c(diag(Ac) h + Bc x + bc)
c =     f c + (1-f) c_new
o =     sigma_o(diag(Ao) h + Bo x + Co c + bo)

h =     o sigma_h(c)

All the matrices above are collated for efficiency, so that we actually use
A = [Af, Ai, Ac, Ao]
B = [Bf, Bi, Bc, Bo]
b = [bf, bi, bc, bo]
C = [Cf, Bi, Co]
And the various B's considered block-diag, with num_head blocks of dims head_dim_state x head_dim_input

NB: with some abuse of notation, we're considering as "hidden state" the collation of [c,h]

'''

LIB_PATH = str(next(iter(Path(__file__).resolve().parent.parent.parent.glob("*parallel_reduce_cuda.cpython*so"))))
torch.ops.load_library(LIB_PATH)


@torch.library.register_fake("parallel_reduce_cuda::fused_fwd_lstm_cifg_diag_mh")
def _(AT, BxpbT, CT):
    torch._check(AT.shape[-1] == BxpbT.shape[-1])
    torch._check(AT.shape[0] == BxpbT.shape[-3])
    torch._check(AT.shape[0] == CT.shape[0])
    return torch.empty_like(BxpbT[...,0:2])

@torch.library.register_fake("parallel_reduce_cuda::fused_bwd_lstm_cifg_diag_mh")
def _(gradientT, hT, AT, BxpbT, CT):
    torch._check(AT.shape[-1] == BxpbT.shape[-1])
    torch._check(AT.shape[0] == BxpbT.shape[-3])
    torch._check(AT.shape[0] == CT.shape[0])
    torch._check(hT.shape[-2] == BxpbT.shape[-2])
    torch._check(gradientT.shape == hT.shape)
    return torch.empty_like(hT)


T = typ.TypeVar("T")


@dataclass(frozen=True)
class LSTMCIFGDiagMHTrait:
    pass


@dataclass
class LSTMCIFGDiagMHConfig(Config[LSTMCIFGDiagMHTrait]):
    nonlin_f: str = 'sigmoid'
    nonlin_o: str = 'sigmoid'
    nonlin_c: str = 'tanh'
    nonlin_state: str = 'tanh'
    num_heads: int = 2
    a_init_fn: str = 'xlstm'
    w_init_fn: str = 'xavier_uniform'
    b_init_fn: str = 'bias_minus_linspace'


@dataclass
class LSTMCIFGDiagMHSystemParameters(SystemParameters[LSTMCIFGDiagMHTrait]):
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    b: torch.Tensor
    nonlin_f: typ.Callable
    nonlin_o: typ.Callable
    nonlin_c: typ.Callable
    nonlin_state: typ.Callable
    derivative_nonlin_f: typ.Callable
    derivative_nonlin_o: typ.Callable
    derivative_nonlin_c: typ.Callable
    derivative_nonlin_state: typ.Callable
    
    def unpack(self) -> typ.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        typ.Callable, typ.Callable, typ.Callable, typ.Callable,
        typ.Callable, typ.Callable, typ.Callable, typ.Callable
    ]:
        return (
            self.A, self.B, self.C, self.b,
            self.nonlin_f, self.nonlin_o, self.nonlin_c, self.nonlin_state,
            self.derivative_nonlin_f, self.derivative_nonlin_o,
            self.derivative_nonlin_c, self.derivative_nonlin_state,
        )
    
    @classmethod
    def repack(
            cls: T,
            pars: typ.Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                typ.Callable, typ.Callable, typ.Callable, typ.Callable,
                typ.Callable, typ.Callable, typ.Callable, typ.Callable
            ]
    ) -> T:
        return LSTMCIFGDiagMHSystemParameters(
            A=pars[0],
            B=pars[1],
            C=pars[2],
            b=pars[3],
            nonlin_f=pars[4],
            nonlin_o=pars[5],
            nonlin_c=pars[6],
            nonlin_state=pars[7],
            derivative_nonlin_f=pars[8],
            derivative_nonlin_o=pars[9],
            derivative_nonlin_c=pars[10],
            derivative_nonlin_state=pars[11],
        )


class LSTMCIFGDiagMHImpl(RNNCellBlockDiagImpl[LSTMCIFGDiagMHSystemParameters]):
    
    @classmethod
    def _num_blocks(cls) -> int:
        return 2
    
    @staticmethod
    def _split_hidden_state(h: torch.Tensor) -> typ.Tuple[torch.Tensor, torch.Tensor]:
        cc, hh = torch.tensor_split(h, 2, dim=-1)
        return cc, hh
    
    @staticmethod
    def _recombine_hidden_state(c: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([c, h], dim=-1)
    
    @classmethod
    def recurrence_step(
            cls,
            x: torch.Tensor,
            h: torch.Tensor,
            system_parameters: LSTMCIFGDiagMHSystemParameters
    ) -> torch.Tensor:
        cc, hh = cls._split_hidden_state(h)
        num_heads = system_parameters.B.shape[0]
        
        f, o, c = torch.unbind(
            torch.einsum('vj,...j->...vj', (system_parameters.A, hh)) +
            torch.einsum(
                'hivj,...hi->...vhj',
                (system_parameters.B, x.view([*x.shape[:-1],num_heads,-1]))
            ).flatten(-2) +
            system_parameters.b,
            dim=-2
        )
        f = system_parameters.nonlin_f(f + system_parameters.C[0, :] * cc)
        c = system_parameters.nonlin_c(c)
        
        cc = f * cc + (1 - f) * c
        o = system_parameters.nonlin_o(o + system_parameters.C[1, :] * cc)
        hh = o * system_parameters.nonlin_state(cc)
        
        return cls._recombine_hidden_state(cc, hh)
    
    @classmethod
    def post_process(
            cls,
            h: torch.Tensor,
            x: torch.Tensor,
            system_parameters: LSTMCIFGDiagMHSystemParameters
    ) -> torch.Tensor:
        _, hh = cls._split_hidden_state(h)
        return hh
    
    @classmethod
    def assemble_initial_guess(
            cls,
            x: torch.Tensor,
            state_dim: int,
            system_parameters: LSTMCIFGDiagMHSystemParameters
    ) -> torch.Tensor:
        # improve initial guess: what you'd get if state was 0 everywhere
        h = torch.zeros([*x.shape[:-1], state_dim], device=x.device, dtype=x.dtype)
        return cls.recurrence_step(x, h, system_parameters)
    
    @classmethod
    def _compute_jacobians_inner(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: LSTMCIFGDiagMHSystemParameters,
    ) -> typ.Tuple[torch.Tensor, ...]:
        cc, _ = cls._split_hidden_state(sol)
        
        sol_prev = cls._roll_state(sol)
        c_prev, h_prev = cls._split_hidden_state(sol_prev)
        num_heads = system_parameters.B.shape[0]

        pre_nl_f, pre_nl_o, pre_nl_c = torch.unbind(
            torch.einsum('vj,...j->...vj', (system_parameters.A, h_prev)) +
            torch.einsum(
                'hivj,...hi->...vhj',
                (system_parameters.B, x.view([*x.shape[:-1],num_heads,-1]))
            ).flatten(-2) +
            system_parameters.b,
            dim=-2
        )
        pre_nl_f = pre_nl_f + system_parameters.C[0, :] * c_prev
        pre_nl_o = pre_nl_o + system_parameters.C[1, :] * cc
        f = system_parameters.nonlin_f(pre_nl_f)
        o = system_parameters.nonlin_o(pre_nl_o)
        c = system_parameters.nonlin_c(pre_nl_c)
        
        grad_f = system_parameters.derivative_nonlin_f(pre_nl_f)
        grad_o = system_parameters.derivative_nonlin_o(pre_nl_o)
        grad_c = system_parameters.derivative_nonlin_c(pre_nl_c)
        
        Jh_f, Jh_o, Jh_c = torch.unbind(
            system_parameters.A * torch.stack([grad_f, grad_o, grad_c], dim=-2),
            dim=-2
        )
        Jc_f, Jc_o = torch.unbind(
            system_parameters.C * torch.stack([grad_f, grad_o], dim=-2),
            dim=-2
        )
        
        o_sdercc = o * system_parameters.derivative_nonlin_state(cc)
        scc = system_parameters.nonlin_state(cc)
        
        Jcc = - (Jc_f * (c_prev - c) + f)
        Jch = - (Jh_f * (c_prev - c) + (1 - f) * Jh_c)
        Jhc = Jcc * (Jc_o * scc + o_sdercc)
        Jhh = - (scc * (Jh_o - Jc_o * Jch) - o_sdercc * Jch)
        
        # jacobians are stored as [B,T,[Jcc,Jch;Jhc,Jhh]], where Jij are all diagonals
        return Jcc, Jch, Jhc, Jhh
    
    @classmethod
    def compute_jacobians(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: LSTMCIFGDiagMHSystemParameters,
    ) -> torch.Tensor:
        Jcc, Jch, Jhc, Jhh = cls._compute_jacobians_inner(sol, x, system_parameters)
        
        jac = torch.cat(
            [
                torch.cat([Jcc.unsqueeze(-1), Jch.unsqueeze(-1)], dim=-1),
                torch.cat([Jhc.unsqueeze(-1), Jhh.unsqueeze(-1)], dim=-1)
            ],
            dim=-2
        )
        
        return jac
    
    @classmethod
    def compute_jacobians_bwd(
            cls,
            sol: torch.Tensor,
            x: torch.Tensor,
            system_parameters: LSTMCIFGDiagMHSystemParameters,
    ) -> torch.Tensor:
        Jcc, Jch, Jhc, Jhh = cls._compute_jacobians_inner(sol, x, system_parameters)
        
        jacobians = torch.cat(
            [
                torch.cat([Jcc.unsqueeze(-1), Jhc.unsqueeze(-1)], dim=-1),
                torch.cat([Jch.unsqueeze(-1), Jhh.unsqueeze(-1)], dim=-1)
            ],
            dim=-2
        )  # this does the "transpose"
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
    
    @classmethod
    def backprop_post_processing(
            cls,
            grad_y: torch.Tensor,
            x: torch.Tensor,
            h: torch.Tensor,
            system_parameters: LSTMCIFGDiagMHSystemParameters
    ) -> typ.Tuple[typ.Optional[torch.Tensor], ...]:
        grad_h = cls._recombine_hidden_state(torch.zeros_like(grad_y), grad_y)
        return (
            grad_h,
            torch.zeros_like(x),
            torch.zeros_like(system_parameters.A),
            torch.zeros_like(system_parameters.B),
            torch.zeros_like(system_parameters.C),
            torch.zeros_like(system_parameters.b),
            None, None, None, None,
            None, None, None, None
        )
    
    @classmethod
    def backprop_to_system_parameters(
            cls,
            dl_dht: torch.Tensor,
            x: torch.Tensor,
            h: torch.Tensor,
            system_parameters: LSTMCIFGDiagMHSystemParameters
    ) -> typ.Tuple[typ.Union[torch.Tensor, None], ...]:
        cc, hh = cls._split_hidden_state(h)
        
        state_prev = cls._roll_state(h)
        c_prev, h_prev = cls._split_hidden_state(state_prev)
        grad_cc, grad_hh = cls._split_hidden_state(dl_dht)
        num_heads = system_parameters.B.shape[0]
        
        pre_nl_f, pre_nl_o, pre_nl_c = torch.unbind(
            torch.einsum('vj,...j->...vj', (system_parameters.A, h_prev)) +
            torch.einsum(
                'hivj,...hi->...vhj',
                (system_parameters.B, x.view([*x.shape[:-1],num_heads,-1]))
            ).flatten(-2) +
            system_parameters.b,
            dim=-2
        )
        pre_nl_f = pre_nl_f + system_parameters.C[0, :] * c_prev
        pre_nl_o = pre_nl_o + system_parameters.C[1, :] * cc
        f = system_parameters.nonlin_f(pre_nl_f)
        o = system_parameters.nonlin_o(pre_nl_o)
        c = system_parameters.nonlin_c(pre_nl_c)
        
        grad_o = grad_hh * system_parameters.nonlin_state(cc) * system_parameters.derivative_nonlin_o(pre_nl_o)
        dl_dc = grad_cc + grad_hh * o * system_parameters.derivative_nonlin_state(cc) + grad_o*system_parameters.C[1,:]
        
        grad_f = dl_dc * (c_prev - c) * system_parameters.derivative_nonlin_f(pre_nl_f)
        grad_c = dl_dc * (1 - f) * system_parameters.derivative_nonlin_c(pre_nl_c)
        
        grad_foc = torch.stack([grad_f, grad_o, grad_c], dim=-2)
        
        grad_A = torch.einsum('...vj,...j->vj', (grad_foc, h_prev))
        grad_b = torch.sum(grad_foc, dim=tuple(range(grad_foc.ndim - 2)))
        grad_B = torch.einsum(
            '...vhj,...hi->hivj',
            (grad_foc.view([*grad_foc.shape[:-1],num_heads,-1]), x.view([*x.shape[:-1],num_heads,-1]))
        )
        grad_x = torch.einsum(
            '...vhj,hivj->...hi',
            (grad_foc.view([*grad_foc.shape[:-1],num_heads,-1]), system_parameters.B)
        ).flatten(-2)
        grad_C = torch.stack([
            torch.sum(grad_foc[..., 0, :] * c_prev, dim=tuple(range(grad_foc.ndim - 2))),
            torch.sum(grad_foc[..., 1, :] * cc, dim=tuple(range(grad_foc.ndim - 2)))
        ], dim=-2
        )
        
        return (
            grad_x,
            grad_A, grad_B, grad_C, grad_b,
            None, None, None, None,
            None, None, None, None
        )
    
    @classmethod
    def fused_parallel_forward(
            cls,
            x: torch.Tensor,
            system_parameters: LSTMCIFGDiagMHSystemParameters,
    ) -> torch.Tensor:
        num_heads = system_parameters.B.shape[0]
        
        sol = cls._fused_parallel_fwd(
            system_parameters.A.transpose(-1, -2).contiguous(),
            torch.einsum(
                '...tvj->...jtv', torch.einsum(
                    'hivj,...hi->...vhj',
                    (system_parameters.B, x.view([*x.shape[:-1], num_heads, -1]))
                ).flatten(-2) + system_parameters.b
            ).contiguous(),
            system_parameters.C.transpose(-1, -2).contiguous(),
        )
        sol = sol.permute(*range(sol.ndim - 3), -2, -1, -3).flatten(-2)
        
        return sol
    
    @classmethod
    def fused_parallel_backward(
            cls,
            gradient: torch.Tensor,
            h: torch.Tensor,
            x: torch.Tensor,
            system_parameters: LSTMCIFGDiagMHSystemParameters,
    ) -> torch.Tensor:
        num_heads = system_parameters.B.shape[0]
        block_size = math.ceil(gradient.shape[-1] / cls._num_blocks())

        dl_dht = cls._fused_parallel_bwd(
            torch.stack(gradient.split(dim=-1, split_size=block_size), dim=-1).transpose(-3, -2).contiguous(),
            torch.stack(h.split(dim=-1, split_size=block_size), dim=-1).transpose(-3, -2).contiguous(),
            system_parameters.A.transpose(-1, -2).contiguous(),
            torch.einsum(
                '...tvj->...jtv', torch.einsum(
                    'hivj,...hi->...vhj',
                    (system_parameters.B, x.view([*x.shape[:-1], num_heads, -1]))
                ).flatten(-2) + system_parameters.b
            ).contiguous(),
            system_parameters.C.transpose(-1, -2).contiguous(),
        )
        dl_dht = dl_dht.permute(*range(dl_dht.ndim - 3), -2, -1, -3).flatten(-2)
        
        return dl_dht
    
    @staticmethod
    @torch._dynamo.allow_in_graph
    def _fused_parallel_fwd(AT: torch.Tensor, BxpbT: torch.Tensor, CT: torch.Tensor) -> torch.Tensor:
        with torch.autocast('cuda', enabled=False):
            dtype = AT.dtype
            return torch.ops.parallel_reduce_cuda.fused_fwd_lstm_cifg_diag_mh(
                AT.to(torch.float32),
                BxpbT.to(torch.float32),
                CT.to(torch.float32)
            ).to(dtype)
    
    @staticmethod
    @torch._dynamo.allow_in_graph
    def _fused_parallel_bwd(
            gradientT: torch.Tensor,
            hT: torch.Tensor,
            AT: torch.Tensor,
            BxpbT: torch.Tensor,
            CT: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast('cuda', enabled=False):
            dtype = AT.dtype
            return torch.ops.parallel_reduce_cuda.fused_bwd_lstm_cifg_diag_mh(
                gradientT.to(torch.float32),
                hT.to(torch.float32),
                AT.to(torch.float32),
                BxpbT.to(torch.float32),
                CT.to(torch.float32)
            ).to(dtype)


class LSTMCIFGDiagMH(BaseRNNCell[LSTMCIFGDiagMHConfig, LSTMCIFGDiagMHSystemParameters, LSTMCIFGDiagMHImpl]):
    
    def __init__(self, config: LSTMCIFGDiagMHConfig):
        super().__init__(config)
    
    def _specific_init(self, config: LSTMCIFGDiagMHConfig):

        assert self.input_dim % config.num_heads == 0, "Number of heads must exactly divide input dimension"
        assert self.state_dim % config.num_heads == 0, "Number of heads must exactly divide state dimension"

        self.num_heads = config.num_heads
        self.head_input_dim = math.ceil(self.input_dim / config.num_heads)
        self.head_state_dim = math.ceil(self.state_dim / config.num_heads)

        # System parameters
        # - collated f,i,o
        self.A = nn.Parameter(torch.empty([3, self.state_dim], device=self.device, dtype=self.dtype))
        self.B = nn.Parameter(torch.empty([self.num_heads, self.head_input_dim, 3, self.head_state_dim],
                                          device=self.device, dtype=self.dtype))
        self.C = nn.Parameter(torch.empty([2, self.state_dim], device=self.device, dtype=self.dtype))
        self.b = nn.Parameter(torch.empty([3, self.state_dim], device=self.device, dtype=self.dtype))
        self.nonlin_f, self.derivative_nonlin_f = self._set_nonlinearity_and_derivative(config.nonlin_f)
        self.nonlin_o, self.derivative_nonlin_o = self._set_nonlinearity_and_derivative(config.nonlin_o)
        self.nonlin_c, self.derivative_nonlin_c = self._set_nonlinearity_and_derivative(config.nonlin_c)
        self.nonlin_state, self.derivative_nonlin_state = self._set_nonlinearity_and_derivative(config.nonlin_state)
        self.reset_parameters()
        
        # a bit of a hack: since in LSTM we're storing both c and h as hidden states, the dim is doubled
        self.state_dim = 2 * self.state_dim
    
    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        INIT_REGISTRY[self._config.a_init_fn](self.A.data, fan_in=self._config.state_dim, fan_out=None)
        INIT_REGISTRY[self._config.w_init_fn](self.B.data, fan_in=self.head_input_dim, fan_out=self.head_state_dim)
        INIT_REGISTRY[self._config.a_init_fn](self.C.data, fan_in=self._config.state_dim, fan_out=None)
        INIT_REGISTRY[self._config.b_init_fn](self.b.data, fan_in=None, fan_out=self.b.numel())
    
    @property
    def _system_parameters(self):
        # - handy class to collect them all
        return LSTMCIFGDiagMHSystemParameters(
            A=self.A,
            B=self.B,
            C=self.C,
            b=self.b,
            nonlin_f=self.nonlin_f,
            nonlin_o=self.nonlin_o,
            nonlin_c=self.nonlin_c,
            nonlin_state=self.nonlin_state,
            derivative_nonlin_f=self.derivative_nonlin_f,
            derivative_nonlin_o=self.derivative_nonlin_o,
            derivative_nonlin_c=self.derivative_nonlin_c,
            derivative_nonlin_state=self.derivative_nonlin_state,
        )
