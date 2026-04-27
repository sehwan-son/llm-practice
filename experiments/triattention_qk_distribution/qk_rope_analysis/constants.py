from pathlib import Path


DEFAULT_PROMPT = "한국어로 자기소개를 두 문장으로 해줘."
DEFAULT_SYSTEM_PROMPT = "You are a concise and helpful assistant."

PAIRING_MODE = "split_half"
PAIR_DEFINITION = "complex_pair[p] = x[p] + i * x[p + head_dim/2]"
PAIRING_NOTE = "Qwen3 rotate_half pairs dim i with dim i + head_dim/2, not adjacent dims."

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
