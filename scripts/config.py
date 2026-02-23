from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    model_variant: str
    base_model_id: str
    checkpoint_repo: str
    num_layers: int
    hidden_dim: int


LLAMA_8B = ModelConfig(
    model_variant="llama-8b",
    base_model_id="meta-llama/Llama-3.1-8B-Instruct",
    checkpoint_repo="ModelOrganismsForEM/Llama-3.1-8B-Instruct_R1_0_1_0_full_train",
    num_layers=32,
    hidden_dim=4096,
)

QWEN_14B = ModelConfig(
    model_variant="qwen-14b",
    base_model_id="Qwen/Qwen2.5-14B-Instruct",
    checkpoint_repo="ModelOrganismsForEM/Qwen2.5-14B-Instruct_R64_0_1_0_full_train",
    num_layers=48,
    hidden_dim=5120,
)


MODELS = {
    LLAMA_8B.model_variant: LLAMA_8B,
    QWEN_14B.model_variant: QWEN_14B,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESPONSES_DIR = RESULTS_DIR / "responses"
ACTIVATIONS_DIR = RESULTS_DIR / "activations"
PROBES_DIR = RESULTS_DIR / "probes"
FIGURES_DIR = RESULTS_DIR / "figures"

