from pathlib import Path


DEFAULT_PROMPT = "한국어로 자기소개를 두 문장으로 해줘."
DEFAULT_SYSTEM_PROMPT = "You are a concise and helpful assistant."

PAIRING_MODE = "split_half"
PAIRING_NOTE = "Qwen3 rotate_half pairs dim i with dim i + head_dim/2, not adjacent dims."

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CALIBRATION_DIR = EXPERIMENT_ROOT / "calibration"
DEFAULT_AIME2025_PROMPT_FILE = DEFAULT_CALIBRATION_DIR / "aime2025.jsonl"
DEFAULT_AIME2025_PROMPT_FIELD = "question"
DEFAULT_AIME2025_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/zai-org/glm-simple-evals-dataset/"
    "resolve/main/aime/aime_2025.jsonl"
)
