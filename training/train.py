"""
Main training script for ParaRNN language model experiments.
Replicates the 1B-parameter experiments from the ParaRNN paper.

Usage:
    python train.py model=paragru
    python train.py model=mamba2 optimizer.lr=0.005
"""

import os
import sys
import math
import time
import logging

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import hydra
from omegaconf import DictConfig, OmegaConf

# Add parent to path for pararnn imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

log = logging.getLogger(__name__)


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr=0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale between min_lr_ratio and 1.0
        base_lr = optimizer.defaults["lr"]
        min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        _, loss = model(input_ids, targets)
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
    model.train()
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)  # perplexity


def save_checkpoint(model, optimizer, scheduler, step, cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "cfg": OmegaConf.to_container(cfg),
    }
    torch.save(state, path)
    log.info(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state.get("step", 0)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
        log.info(f"World size: {world_size}")

    # Set seed
    torch.manual_seed(cfg.seed + rank)
    torch.cuda.manual_seed(cfg.seed + rank)

    # WandB init (main process only)
    if cfg.wandb.enabled and is_main:
        import wandb
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.model.name}_1B_{cfg.seed}",
        )

    # Build model
    from src.model import build_model
    model = build_model(cfg).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    if is_main:
        log.info(f"Model: {cfg.model.name}, Parameters: {num_params / 1e9:.2f}B")

    # FSDP wrapping
    if world_size > 1 and cfg.fsdp.enabled:
        bf16_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(
            model,
            mixed_precision=bf16_policy,
            device_id=local_rank,
        )
    elif cfg.dtype == "bfloat16":
        model = model.to(torch.bfloat16)

    # Build data
    from src.data import build_train_dataloader, build_val_dataloader
    train_loader = build_train_dataloader(cfg)
    val_loader = build_val_dataloader(cfg)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
    )

    # Scheduler
    scheduler = get_cosine_schedule(
        optimizer,
        warmup_steps=cfg.warmup_iterations,
        total_steps=cfg.total_iterations,
        min_lr=cfg.scheduler.min_lr,
    )

    # Resume
    start_step = 0
    if cfg.resume_from:
        start_step = load_checkpoint(cfg.resume_from, model, optimizer, scheduler)
        if is_main:
            log.info(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    train_iter = iter(train_loader)
    step = start_step
    t0 = time.time()

    if is_main:
        log.info(f"Starting training from step {step}")

    while step < cfg.total_iterations:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        # Forward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(input_ids, targets)

        # Backward
        loss.backward()

        # Gradient clipping
        if cfg.gradient_clip > 0:
            if isinstance(model, FSDP):
                model.clip_grad_norm_(cfg.gradient_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1

        # Logging
        if is_main and step % cfg.log_interval == 0:
            elapsed = time.time() - t0
            tokens_per_sec = (cfg.log_interval * cfg.batch_size_per_gpu * world_size * cfg.seq_length) / elapsed
            current_lr = scheduler.get_last_lr()[0]
            log_dict = {
                "train/loss": loss.item(),
                "train/perplexity": math.exp(min(loss.item(), 20)),
                "train/lr": current_lr,
                "train/tokens_per_sec": tokens_per_sec,
                "train/step": step,
            }
            log.info(
                f"Step {step}/{cfg.total_iterations} | "
                f"Loss: {loss.item():.4f} | "
                f"PPL: {math.exp(min(loss.item(), 20)):.2f} | "
                f"LR: {current_lr:.6f} | "
                f"Tok/s: {tokens_per_sec:.0f}"
            )
            if cfg.wandb.enabled:
                import wandb
                wandb.log(log_dict, step=step)
            t0 = time.time()

        # Evaluation
        if is_main and step % cfg.eval_interval == 0:
            ppl = evaluate(model, val_loader, device)
            log.info(f"Step {step} | Validation PPL: {ppl:.2f}")
            if cfg.wandb.enabled:
                import wandb
                wandb.log({"val/perplexity": ppl, "train/step": step}, step=step)
            val_loader = build_val_dataloader(cfg)  # reset iterator

        # Checkpoint
        if is_main and step % cfg.save_interval == 0:
            ckpt_path = os.path.join(
                cfg.checkpoint_dir, cfg.model.name, f"step_{step}.pt"
            )
            save_checkpoint(model, optimizer, scheduler, step, cfg, ckpt_path)

    # Final save
    if is_main:
        final_path = os.path.join(cfg.checkpoint_dir, cfg.model.name, "final.pt")
        save_checkpoint(model, optimizer, scheduler, step, cfg, final_path)

        # Final evaluation
        ppl = evaluate(model, val_loader, device)
        log.info(f"Final Validation PPL: {ppl:.2f}")
        if cfg.wandb.enabled:
            import wandb
            wandb.log({"val/perplexity_final": ppl}, step=step)
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
