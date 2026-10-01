"""Smoke test: XGrammar JSON-schema-constrained generation with a small LM.

Uses a minimal local logits processor instead of `xgrammar.contrib.hf`,
which passes a torch tensor where the C binding expects an int
(xgrammar 0.2.0 with recent transformers).
"""
from __future__ import annotations

import torch
import xgrammar as xgr
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class XGrammarLogitsProcessor:
    """Single-batch xgrammar logits processor compatible with HF generate."""

    def __init__(self, compiled_grammar, vocab_size: int):
        self.matcher = xgr.GrammarMatcher(compiled_grammar)
        self.vocab_size = vocab_size
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


def main() -> None:
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    config = AutoConfig.from_pretrained(model_name)

    schema = r'''
    {
      "type": "object",
      "properties": {
        "plan": {
          "type": "array",
          "items": {
            "type": "string",
            "enum": ["pickup", "putdown", "stack", "unstack"]
          }
        }
      },
      "required": ["plan"],
      "additionalProperties": false
    }
    '''

    messages = [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": "Give a short Blocksworld plan."},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled = compiler.compile_json_schema(schema)
    processor = XGrammarLogitsProcessor(compiled, vocab_size=config.vocab_size)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        logits_processor=[processor],
    )
    generated = output_ids[0][len(inputs.input_ids[0]):]
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
