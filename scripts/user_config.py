from pathlib import Path

# Edit this file for your local/VM environment.

MODEL_VARIANT = "llama-8b"  # llama-8b or qwen-14b

# Local paths on the machine running the scripts.
BASE_MODEL_PATH = Path("/path/to/base_model")
CHECKPOINT_DIR = Path("/path/to/em_checkpoints")
BETLEY_REPO_PATH = Path("/path/to/emergent-misalignment")

# Option A dataset prep
N_PER_CLASS = 200
SEED = 42
LABELLED_DATA_PATH = Path("results/responses/betley_labelled.json")

# Activation collection
MAX_SEQ_LEN = 1024
TORCH_DTYPE = "bfloat16"  # bfloat16, float16, float32
DEVICE_MAP = "auto"  # auto, cuda, cuda:0, cpu
INPUT_DEVICE = "cuda:0"  # where tokenized inputs are placed
LIMIT_EXAMPLES = None  # e.g., 100 for fast debugging
LAST_LAYER_ONLY = False  # set True for fast sanity runs on final transformer layer only
LAYER_INDICES = None  # comma-separated list via CLI is preferred for custom subsets

# Probe training
N_SEEDS = 3
N_FOLDS = 5
MAX_ITER = 1000
LOGREG_C = 1.0
