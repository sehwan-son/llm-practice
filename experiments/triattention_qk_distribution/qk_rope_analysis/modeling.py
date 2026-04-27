from contextlib import ExitStack
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name: str, trust_remote_code: bool):
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=trust_remote_code)


def load_model(model_name: str, device: str, dtype: torch.dtype, trust_remote_code: bool):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model


def get_decoder_layers(model) -> Any:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Could not find decoder layers at model.model.layers.")
    return layers


def make_capture_hook(store: dict[str, dict[int, torch.Tensor]], tensor_name: str, layer_idx: int):
    def hook(_module, _inputs, output):
        store[tensor_name][layer_idx] = output.detach().to(device="cpu", dtype=torch.float32).contiguous()

    return hook


def capture_pre_rope_qk(
    model,
    model_inputs: dict[str, torch.Tensor],
    selected_layers: list[int],
) -> dict[str, dict[int, torch.Tensor]]:
    layers = get_decoder_layers(model)
    captured = {"q": {}, "k": {}}

    with ExitStack() as stack:
        for layer_idx in selected_layers:
            self_attn = layers[layer_idx].self_attn
            if not hasattr(self_attn, "q_norm") or not hasattr(self_attn, "k_norm"):
                raise ValueError(
                    f"Layer {layer_idx} does not expose q_norm/k_norm. "
                    "This script expects a Qwen3-style attention module."
                )

            q_handle = self_attn.q_norm.register_forward_hook(make_capture_hook(captured, "q", layer_idx))
            k_handle = self_attn.k_norm.register_forward_hook(make_capture_hook(captured, "k", layer_idx))
            stack.callback(q_handle.remove)
            stack.callback(k_handle.remove)

        with torch.inference_mode():
            model(**model_inputs, use_cache=False)

    return captured


def get_layer_rope_inv_freq(layer, num_pairs: int) -> torch.Tensor | None:
    rotary_emb = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
    inv_freq = getattr(rotary_emb, "inv_freq", None)
    if inv_freq is None:
        return None

    inv_freq = inv_freq.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if inv_freq.numel() != num_pairs:
        return None
    return inv_freq
