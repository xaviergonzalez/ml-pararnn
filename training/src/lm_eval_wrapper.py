"""
Wrapper to make our models compatible with lm-eval-harness.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


class ParaRNNLMWrapper:
    """Wraps our language model for lm-eval-harness compatibility."""

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self._device = next(model.parameters()).device
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.data.tokenizer,
            cache_dir=cfg.data.cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = cfg.model.vocab_size
        self.batch_size = 4

    @property
    def device(self):
        return self._device

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.cfg.seq_length

    @property
    def max_gen_toks(self):
        return 256

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = self.model(inps)
        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        # Simple greedy generation for lm-eval tasks that need it
        input_ids = context
        for _ in range(max_length - input_ids.shape[1]):
            logits = self._model_call(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
        return input_ids

    def loglikelihood(self, requests):
        results = []
        for context, continuation in requests:
            ctx_tokens = self.tok_encode(context)
            cont_tokens = self.tok_encode(continuation)
            all_tokens = ctx_tokens + cont_tokens

            if len(all_tokens) > self.max_length:
                all_tokens = all_tokens[-self.max_length:]
                cont_len = len(cont_tokens)
            else:
                cont_len = len(cont_tokens)

            input_ids = torch.tensor([all_tokens], device=self.device, dtype=torch.long)
            logits = self._model_call(input_ids)

            # Get log probs for continuation tokens
            log_probs = F.log_softmax(logits[0], dim=-1)
            cont_start = len(all_tokens) - cont_len
            target_tokens = torch.tensor(all_tokens[cont_start:], device=self.device)
            token_log_probs = log_probs[cont_start - 1:-1]
            cont_log_prob = token_log_probs.gather(1, target_tokens.unsqueeze(1)).sum().item()

            is_greedy = (logits[0, cont_start - 1:-1].argmax(dim=-1) == target_tokens).all().item()
            results.append((cont_log_prob, is_greedy))

        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for string, in requests:
            tokens = self.tok_encode(string)
            total_ll = 0.0
            for i in range(0, len(tokens), self.max_length):
                chunk = tokens[i:i + self.max_length]
                if len(chunk) < 2:
                    continue
                input_ids = torch.tensor([chunk], device=self.device, dtype=torch.long)
                logits = self._model_call(input_ids)
                log_probs = F.log_softmax(logits[0], dim=-1)
                targets = torch.tensor(chunk[1:], device=self.device)
                token_ll = log_probs[:-1].gather(1, targets.unsqueeze(1)).sum().item()
                total_ll += token_ll
            results.append((total_ll,))
        return results

    def generate_until(self, requests):
        results = []
        for context, until in requests:
            tokens = self.tok_encode(context)[-self.max_length:]
            input_ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
            gen = self._model_generate(input_ids, min(self.max_length, len(tokens) + self.max_gen_toks), self.eot_token_id)
            gen_text = self.tok_decode(gen[0, len(tokens):].tolist())
            # Truncate at stop sequences
            if isinstance(until, dict):
                until = until.get("until", [])
            for stop in until:
                idx = gen_text.find(stop)
                if idx != -1:
                    gen_text = gen_text[:idx]
            results.append(gen_text)
        return results
