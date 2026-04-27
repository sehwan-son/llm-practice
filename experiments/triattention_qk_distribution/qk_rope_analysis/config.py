from pathlib import Path

import torch

from .constants import EXPERIMENT_ROOT


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_tensor_names(selection: str) -> list[str]:
    return ["q", "k"] if selection == "both" else [selection]


def sanitize_model_name(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        return f"local__{model_path.name}"
    return model_name.strip("/").replace("/", "__")


def default_output_dir(model_name: str) -> Path:
    base_dir = EXPERIMENT_ROOT / "outputs" / "runs"
    return base_dir / sanitize_model_name(model_name)


def build_prompt_text(tokenizer, prompt: str, system_prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def parse_index_selection(selection_arg: str, limit: int, label: str) -> list[int]:
    if selection_arg == "all":
        return list(range(limit))

    selected = []
    for raw_item in selection_arg.split(","):
        item = raw_item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0 or index >= limit:
            raise ValueError(f"{label} index {index} is out of range for size {limit}.")
        selected.append(index)

    if not selected:
        raise ValueError(f"No valid {label} indices were selected.")
    return sorted(set(selected))
