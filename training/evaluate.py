"""
Evaluation script: holdout perplexity, lm-eval-harness, and LLE measurement.

Usage:
    python evaluate.py model=paragru checkpoint_path=/path/to/final.pt
    python evaluate.py model=paragru checkpoint_path=/path/to/final.pt eval_mode=lm_eval
    python evaluate.py model=paragru checkpoint_path=/path/to/final.pt eval_mode=lle
"""

import os
import sys
import logging
import math

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = cfg.get("checkpoint_path", None)
    eval_mode = cfg.get("eval_mode", "all")  # "perplexity", "lm_eval", "lle", "all"

    if cfg.wandb.enabled:
        import wandb
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.model.name}_1B_eval_{eval_mode}",
            job_type="eval",
        )

    # Build model
    from src.model import build_model
    model = build_model(cfg).to(device)

    # Load checkpoint
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        log.info(f"Loaded checkpoint from {checkpoint_path}")

    model.eval()
    results = {}

    # 1. Holdout perplexity
    if eval_mode in ("perplexity", "all"):
        log.info("Computing holdout perplexity...")
        from src.data import build_val_dataloader
        val_loader = build_val_dataloader(cfg)
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 200:
                    break
                input_ids = batch["input_ids"].to(device)
                targets = batch["targets"].to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(input_ids, targets)
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
        ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
        results["holdout_perplexity"] = ppl
        log.info(f"Holdout perplexity: {ppl:.2f}")

    # 2. lm-eval-harness
    if eval_mode in ("lm_eval", "all"):
        log.info("Running lm-eval-harness...")
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM

            # Wrap model for lm-eval compatibility
            from src.lm_eval_wrapper import ParaRNNLMWrapper
            lm = ParaRNNLMWrapper(model, cfg)

            # Tasks from the paper: ARC-C, HellaSwag, OBQA, WinoGrande, PIQA, MMLU
            tasks_and_shots = {
                "arc_challenge": [0, 25],
                "hellaswag": [0, 10],
                "openbookqa": [0, 10],
                "winogrande": [0, 5],
                "piqa": [0],
                "mmlu": [0],
            }

            for task_name, shot_list in tasks_and_shots.items():
                for num_fewshot in shot_list:
                    try:
                        eval_results = lm_eval.simple_evaluate(
                            model=lm,
                            tasks=[task_name],
                            num_fewshot=num_fewshot,
                            device=str(device),
                            batch_size=4,
                        )
                        for task, task_results in eval_results["results"].items():
                            for metric, value in task_results.items():
                                if isinstance(value, (int, float)):
                                    key = f"lm_eval/{task}_{num_fewshot}shot/{metric}"
                                    results[key] = value
                                    log.info(f"{key}: {value:.4f}")
                    except Exception as e:
                        log.warning(f"Failed to evaluate {task_name} ({num_fewshot}-shot): {e}")

        except ImportError:
            log.warning("lm-eval-harness not installed. Run: pip install lm-eval")

    # 3. Largest Lyapunov Exponent
    if eval_mode in ("lle", "all"):
        log.info("Computing Largest Lyapunov Exponent...")
        if cfg.model.arch in ("paragru", "paralstm"):
            from src.lle import compute_lle_for_model
            from src.data import build_val_dataloader
            val_loader = build_val_dataloader(cfg)
            lle_results = compute_lle_for_model(model, val_loader, device)
            results.update({f"lle/{k}": v for k, v in lle_results.items()})
            log.info(f"LLE results: {lle_results}")
        elif cfg.model.arch == "mamba2":
            log.info("LLE for Mamba2: using autograd-based Jacobian estimation...")
            from src.lle import compute_lle_for_model
            from src.data import build_val_dataloader
            val_loader = build_val_dataloader(cfg)
            lle_results = compute_lle_for_model(model, val_loader, device)
            results.update({f"lle/{k}": v for k, v in lle_results.items()})
            log.info(f"LLE results: {lle_results}")
        else:
            log.info("LLE computation not supported for Transformer")

    # Log all results
    if cfg.wandb.enabled:
        import wandb
        wandb.log(results)
        wandb.finish()

    log.info(f"All results: {results}")
    return results


if __name__ == "__main__":
    main()
