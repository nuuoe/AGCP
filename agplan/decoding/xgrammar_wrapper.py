"""Hard grammar-constrained decoding around XGrammar.

Uses an in-house single-trajectory logits processor instead of
`xgrammar.contrib.hf` because the latter passes a torch tensor where the
underlying C binding expects an int (xgrammar 0.2.0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import xgrammar as xgr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class DecodeConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


class _XGrammarLogitsProcessor:
    def __init__(self, compiled_grammar, vocab_size: int):
        self.matcher = xgr.GrammarMatcher(compiled_grammar)
        self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        self.prefilled = False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.prefilled and not self.matcher.is_terminated():
            sampled = int(input_ids[0, -1].item())
            assert self.matcher.accept_token(sampled)
        self.prefilled = True

        if not self.matcher.is_terminated():
            self.matcher.fill_next_token_bitmask(self.bitmask, 0)

        device_type = scores.device.type
        if device_type != "cuda":
            scores_cpu = scores.to("cpu")
            xgr.apply_token_bitmask_inplace(scores_cpu, self.bitmask.to(scores_cpu.device))
            return scores_cpu.to(scores.device)
        xgr.apply_token_bitmask_inplace(scores, self.bitmask.to(scores.device))
        return scores


class XGrammarConstrainedDecoder:
    """Hard-constrained generation against a JSON schema or EBNF grammar."""

    def __init__(
        self,
        model_name: str,
        schema: Optional[str] = None,
        ebnf: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Allow re-compile of the schema later (for dynamic vocabulary).
        if (schema is None) == (ebnf is None):
            raise ValueError("provide exactly one of `schema` or `ebnf`")

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        # transformers 4.x uses `torch_dtype`; 5.x uses `dtype`. Try
        # the newer signature first, fall back to legacy.
        _dt = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=_dt,
            ).to(self.device)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=_dt,
            ).to(self.device)

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.config.vocab_size
        )
        self._compiler = xgr.GrammarCompiler(tokenizer_info)
        if schema is not None:
            self.compiled = self._compiler.compile_json_schema(schema)
        else:
            self.compiled = self._compiler.compile_grammar(ebnf)

    def recompile_schema(self, schema: str) -> None:
        """Swap in a new JSON schema (e.g. after promoting macros to enum)."""
        self.compiled = self._compiler.compile_json_schema(schema)

    def recompile_ebnf(self, ebnf: str) -> None:
        """Swap in a new EBNF grammar (e.g. per-task verifier-induced grammar)."""
        self.compiled = self._compiler.compile_grammar(ebnf)

    def _format_prompt(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        user_prompt: str,
        system_prompt: str = "Return only valid JSON.",
        cfg: Optional[DecodeConfig] = None,
        seed: Optional[int] = None,
    ) -> str:
        cfg = cfg or DecodeConfig()
        if seed is not None:
            torch.manual_seed(seed)

        text = self._format_prompt(system_prompt, user_prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        processor = _XGrammarLogitsProcessor(self.compiled, self.config.vocab_size)

        gen_kwargs: dict = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            logits_processor=[processor],
        )
        if cfg.do_sample:
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated = output_ids[0][len(inputs.input_ids[0]):]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_n(
        self,
        user_prompt: str,
        n: int,
        system_prompt: str = "Return only valid JSON.",
        cfg: Optional[DecodeConfig] = None,
        base_seed: int = 0,
    ) -> list[str]:
        cfg = cfg or DecodeConfig(do_sample=True)
        return [
            self.generate(user_prompt, system_prompt, cfg, seed=base_seed + i)
            for i in range(n)
        ]
