from __future__ import annotations

from pathlib import Path

from .domain import Action, Observation
from .prompting import SYSTEM_PROMPT, observation_message, parse_action_text


class QwenPolicy:
    """A lazily imported local Qwen policy with an optional PEFT adapter."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        adapter_path: str | Path | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        load_in_4bit: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("install PondLLM training dependencies before loading a model") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch")

        model_kwargs = {
            "device_map": {"": 0},
            "dtype": torch.bfloat16,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required to load an adapter") from exc
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.last_raw_output: str | None = None

    def choose(self, observation: Observation | dict) -> Action:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation_message(observation)},
        ]
        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        if hasattr(encoded, "items"):
            model_inputs = {key: value.to(self.model.device) for key, value in encoded.items()}
            input_length = model_inputs["input_ids"].shape[-1]
        else:
            model_inputs = {"input_ids": encoded.to(self.model.device)}
            input_length = model_inputs["input_ids"].shape[-1]
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs.update({"temperature": self.temperature, "top_p": 0.9})
        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **generation_kwargs)
        generated = output_ids[0, input_length:]
        self.last_raw_output = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return parse_action_text(self.last_raw_output)
