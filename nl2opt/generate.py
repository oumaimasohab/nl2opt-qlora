"""Model wrapper for evaluation. Anything with ``generate(list_of_messages, max_tokens)`` works,
which lets tests substitute a fake model."""

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"


class MLXGenerator:
    """Greedy decoding with mlx-lm's batch_generate (default sampler = argmax)."""

    def __init__(self, model_id=DEFAULT_MODEL, adapter_path=None):
        """``adapter_path``: an mlx_lm adapter directory or a checkpoint file
        (``NNNNNNN_adapters.safetensors``) inside one."""
        from mlx_lm import load

        from nl2opt.adapters import ResolvedAdapter

        self.model_id = model_id
        self.adapter_path = adapter_path
        self._adapter = None
        if adapter_path is None:
            self.model, self.tokenizer = load(model_id)
        else:
            # Keep the resolved directory alive as long as the model: MLX may read weights lazily.
            self._adapter = ResolvedAdapter(adapter_path)
            self.model, self.tokenizer = load(model_id, adapter_path=self._adapter.__enter__())

    def __del__(self):
        if getattr(self, "_adapter", None) is not None:
            self._adapter.__exit__(None, None, None)

    def encode(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if hasattr(ids, "keys"):  # some tokenizer versions return a BatchEncoding
            ids = ids["input_ids"]
        return list(ids)

    def generate(self, batch_messages, max_tokens):
        from mlx_lm import batch_generate

        prompts = [self.encode(m) for m in batch_messages]
        response = batch_generate(
            self.model, self.tokenizer, prompts, max_tokens=max_tokens, return_token_ids=True
        )
        return [
            {
                "text": text,
                "prompt_tokens": len(prompt),
                "generation_tokens": len(ids),
                "truncated": len(ids) >= max_tokens,
            }
            for prompt, text, ids in zip(prompts, response.texts, response.token_ids)
        ]
