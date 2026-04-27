import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import torch

from .constants import EXPERIMENT_ROOT


DEFAULT_CALIBRATION_TEXT_FIELDS = ("problem", "question", "prompt", "text", "input", "content")


def _is_cuda_device_usable(device: str) -> bool:
    if not torch.cuda.is_available():
        return False

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return False

    device_idx = torch_device.index if torch_device.index is not None else 0
    if device_idx >= torch.cuda.device_count():
        return False

    try:
        torch.empty(1, device=torch_device)
    except Exception:
        return False
    return True


def _resolve_auto_device() -> str | None:
    return "cuda:0" if _is_cuda_device_usable("cuda:0") else None


def _is_device_usable(device: str) -> bool:
    if device == "cpu":
        return True
    if device.startswith("cuda"):
        return _is_cuda_device_usable(device)
    return True


def resolve_device(device_arg: str) -> str:
    candidates = [candidate.strip() for candidate in device_arg.split(",") if candidate.strip()]
    if not candidates:
        resolved = _resolve_auto_device()
        if resolved is not None:
            return resolved
        raise RuntimeError(f"No usable device found from --device {device_arg!r}.")

    if len(candidates) == 1 and candidates[0] != "auto":
        return candidates[0]

    for candidate in candidates:
        resolved = _resolve_auto_device() if candidate == "auto" else candidate
        if resolved is not None and _is_device_usable(resolved):
            return resolved

    raise RuntimeError(f"No usable device found from --device {device_arg!r}.")



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


def _extract_calibration_text(payload: Any, text_field: str | None) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        candidate_fields = [text_field] if text_field else DEFAULT_CALIBRATION_TEXT_FIELDS
        for field in candidate_fields:
            if field and field in payload and payload[field] is not None:
                return str(payload[field])
        raise ValueError(
            "Could not find calibration text in JSON object. "
            f"Tried fields: {', '.join(candidate_fields)}."
        )
    raise ValueError(f"Unsupported calibration record type: {type(payload).__name__}.")


def download_calibration_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def load_calibration_prompt(
    prompt: str,
    prompt_file: str | None = None,
    prompt_field: str | None = None,
    download_url: str | None = None,
) -> str:
    if not prompt_file:
        return prompt

    path = Path(prompt_file).expanduser().resolve()
    if not path.exists():
        if not download_url:
            raise FileNotFoundError(f"Calibration prompt file not found: {path}")
        download_calibration_file(download_url, path)

    if path.suffix == ".jsonl":
        parts = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}") from exc
            parts.append(_extract_calibration_text(payload, prompt_field))
        if not parts:
            raise ValueError(f"No calibration records found in {path}")
        return "\n\n".join(parts)

    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
            payload = payload["data"]
        if isinstance(payload, list):
            if not payload:
                raise ValueError(f"No calibration records found in {path}")
            return "\n\n".join(_extract_calibration_text(item, prompt_field) for item in payload)
        return _extract_calibration_text(payload, prompt_field)

    return path.read_text(encoding="utf-8")


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
