from __future__ import annotations

from datetime import timedelta
from typing import List

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from lm_eval.api.model import LM


class DreamEvalHarness(LM):
    """Minimal Dream generation adapter derived from DreamLM/Dream eval/eval.py."""

    def __init__(self, config: dict):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        model_cfg = config["model"]
        eval_cfg = config["evaluation"]
        accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(weeks=1))]
        )
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes
        self._device = accelerator.device
        self.batch_size_per_gpu = int(eval_cfg["batch_size"])
        if self.batch_size_per_gpu != 1:
            raise ValueError("Dream matched baseline is frozen to batch size 1")

        dtype = {"bfloat16": torch.bfloat16}[model_cfg["dtype"]]
        common = dict(
            revision=model_cfg["revision"],
            cache_dir=model_cfg["cache_dir"],
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            model_cfg["id"], torch_dtype=dtype, **common
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["id"], **common)
        actual_revision = getattr(self.model.config, "_commit_hash", None)
        if actual_revision != model_cfg["revision"]:
            raise ValueError(
                f"Dream checkpoint revision mismatch: {actual_revision!r}"
            )

        self.max_length = int(getattr(self.model.config, "max_position_embeddings", 2048))
        self.max_new_tokens = int(eval_cfg["max_new_tokens"])
        self.diffusion_steps = int(eval_cfg["diffusion_steps"])
        self.add_bos_token = bool(eval_cfg["add_bos_token"])
        self.temperature = float(eval_cfg["temperature"])
        self.top_p = eval_cfg["top_p"]
        self.top_k = eval_cfg["top_k"]
        self.alg = eval_cfg["algorithm"]
        self.alg_temp = float(eval_cfg["algorithm_temperature"])
        self.generation_receipts = []

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _generate_one(self, prompt: str) -> str:
        if self.add_bos_token:
            if not self.tokenizer.bos_token:
                raise ValueError("Dream tokenizer has no BOS token")
            prompt = self.tokenizer.bos_token + prompt
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        prompt_tokens = int(prompt_ids.shape[1])
        if prompt_tokens + self.max_new_tokens > self.max_length:
            raise ValueError(
                f"prompt exceeds frozen Dream context: {prompt_tokens}+{self.max_new_tokens}>{self.max_length}"
            )
        prompt_ids = prompt_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.model.diffusion_generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.diffusion_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
        )
        generated_ids = output.sequences[0, prompt_tokens:].tolist()
        text = self.tokenizer.decode(generated_ids)
        if self.tokenizer.eos_token:
            text = text.split(self.tokenizer.eos_token)[0]
        self.generation_receipts.append(
            {"prompt_tokens": prompt_tokens, "generated_token_slots": len(generated_ids)}
        )
        return text

    @torch.no_grad()
    def generate_until(self, requests: List, disable_tqdm: bool = False):
        del disable_tqdm
        outputs = []
        for request in requests:
            prompt, generation_kwargs = request.arguments
            text = self._generate_one(prompt)
            for stop in generation_kwargs.get("until", []):
                text = text.split(stop)[0]
            outputs.append(text)
        return outputs

    def loglikelihood(self, requests):
        raise NotImplementedError("This frozen experiment is generation-only")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("This frozen experiment is generation-only")
