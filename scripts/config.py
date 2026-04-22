from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    model_variant: str
    base_model_id: str
    checkpoint_repo: str
    num_layers: int
    hidden_dim: int


QWEN_14B = ModelConfig(
    model_variant="qwen-14b",
    base_model_id="Qwen/Qwen2.5-14B-Instruct",
    checkpoint_repo="ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
    num_layers=48,
    hidden_dim=5120,
)


MODELS = {
    QWEN_14B.model_variant: QWEN_14B,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
