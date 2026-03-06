"""
Language model wrappers for ParaGRU, ParaLSTM, Mamba2, and Transformer.
Follows the paper's architecture: RNN block = Conv + RNN Cell + Gated RMSNorm + Linear,
interleaved with MLP layers and residual connections (DCLM-style Transformer backbone).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class CausalConv1d(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size - 1, groups=d_model
        )

    def forward(self, x):
        # x: (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv(x)[..., :x.shape[-1]]  # causal: trim right
        return x.transpose(1, 2)


class GatedMLP(nn.Module):
    def __init__(self, d_model, expand=4):
        super().__init__()
        d_inner = int(d_model * expand * 2 / 3)
        # Round to nearest multiple of 256 for efficiency
        d_inner = ((d_inner + 255) // 256) * 256
        self.w1 = nn.Linear(d_model, d_inner, bias=False)
        self.w2 = nn.Linear(d_inner, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_inner, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ParaRNNBlock(nn.Module):
    """RNN block following Fig. 9 of the paper:
    Input -> Linear -> [Conv -> RNN Cell -> Gated RMSNorm] * Linear -> + residual
    With learnable residual scaling.
    """
    def __init__(self, d_model, rnn_cell, conv_kernel_size=4):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv = CausalConv1d(d_model, conv_kernel_size)
        self.rnn_cell = rnn_cell
        self.gate_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.conv(x)
        x = F.silu(x)
        x = self.rnn_cell(x)
        x = self.gate_norm(x) * F.silu(z)
        x = self.out_proj(x)
        return residual * self.residual_scale + x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = GatedMLP(d_model)

    def forward(self, x):
        h = self.norm1(x)
        L = h.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=h.device, dtype=h.dtype)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class RNNLanguageModel(nn.Module):
    """Full language model with RNN backbone."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.model.d_model
        vocab_size = cfg.model.vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList()

        for _ in range(cfg.model.n_layers):
            rnn_cell = self._make_rnn_cell(cfg)
            self.blocks.append(ParaRNNBlock(d_model, rnn_cell))
            self.blocks.append(nn.Sequential(
                RMSNorm(d_model),
                GatedMLP(d_model),
            ))

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _make_rnn_cell(self, cfg):
        from pararnn.rnn_cell.rnn_cell_utils import Config
        from pararnn.parallel_reduction.parallel_reduction import NewtonConfig

        newton_cfg = NewtonConfig(max_its=cfg.newton.max_its)
        d_model = cfg.model.d_model
        n_heads = cfg.model.n_heads

        if cfg.model.arch == "paragru":
            from pararnn.rnn_cell.gru_diag_mh import GRUDiagMH, GRUDiagMHConfig
            cell_cfg = GRUDiagMHConfig(
                state_dim=d_model,
                input_dim=d_model,
                device="cuda",
                dtype=torch.float32,
                mode=cfg.model.mode,
                newton_config=newton_cfg,
                num_heads=n_heads,
            )
            return GRUDiagMH(cell_cfg)
        elif cfg.model.arch == "paralstm":
            from pararnn.rnn_cell.lstm_cifg_diag_mh import LSTMCIFGDiagMH, LSTMCIFGDiagMHConfig
            cell_cfg = LSTMCIFGDiagMHConfig(
                state_dim=d_model,
                input_dim=d_model,
                device="cuda",
                dtype=torch.float32,
                mode=cfg.model.mode,
                newton_config=newton_cfg,
                num_heads=n_heads,
            )
            return LSTMCIFGDiagMH(cell_cfg)
        else:
            raise ValueError(f"Unknown RNN arch: {cfg.model.arch}")

    def _init_weights(self):
        # DCLM-style initialization
        d_model = self.cfg.model.d_model
        std = 1.0 / math.sqrt(d_model)
        for name, p in self.named_parameters():
            if p.dim() >= 2 and "rnn_cell" not in name:
                nn.init.normal_(p, mean=0.0, std=std)
            elif "embedding" in name:
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            # z-loss regularization (from DCLM)
            if hasattr(self.cfg, 'z_loss_coeff') and self.cfg.z_loss_coeff > 0:
                z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
                loss = loss + self.cfg.z_loss_coeff * z_loss
        return logits, loss


class TransformerLanguageModel(nn.Module):
    """Transformer baseline (DCLM architecture)."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.model.d_model
        vocab_size = cfg.model.vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, cfg.model.n_heads)
            for _ in range(cfg.model.n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        d_model = self.cfg.model.d_model
        std = 1.0 / math.sqrt(d_model)
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            if hasattr(self.cfg, 'z_loss_coeff') and self.cfg.z_loss_coeff > 0:
                z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
                loss = loss + self.cfg.z_loss_coeff * z_loss
        return logits, loss


class Mamba2LanguageModel(nn.Module):
    """Mamba2 baseline using HuggingFace implementation."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        from transformers import Mamba2Config, Mamba2ForCausalLM

        mamba_cfg = Mamba2Config(
            vocab_size=cfg.model.vocab_size,
            hidden_size=cfg.model.d_model,
            num_hidden_layers=cfg.model.n_layers,
            num_heads=cfg.model.n_heads,
        )
        self.model = Mamba2ForCausalLM(mamba_cfg)

    def forward(self, input_ids, targets=None):
        if targets is not None:
            outputs = self.model(input_ids=input_ids, labels=targets)
            logits = outputs.logits
            loss = outputs.loss
            if hasattr(self.cfg, 'z_loss_coeff') and self.cfg.z_loss_coeff > 0:
                z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
                loss = loss + self.cfg.z_loss_coeff * z_loss
            return logits, loss
        else:
            outputs = self.model(input_ids=input_ids)
            return outputs.logits, None


def build_model(cfg: DictConfig) -> nn.Module:
    arch = cfg.model.arch
    if arch in ("paragru", "paralstm"):
        return RNNLanguageModel(cfg)
    elif arch == "transformer":
        return TransformerLanguageModel(cfg)
    elif arch == "mamba2":
        return Mamba2LanguageModel(cfg)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
