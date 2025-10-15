import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    data_dir: str = os.environ.get("SMU_DATA_DIR", "C:/Users/28428/Desktop/sharememory_user/data")
    memory_path: str = os.path.join(data_dir, "memory.json")
    users_path: str = os.path.join(data_dir, "users.json")

    # Embedding
    embed_use_hf: bool = os.environ.get("SMU_EMBED_USE_HF", "1") == "1"
    embed_model_name: str = os.environ.get("SMU_EMBED_MODEL", "C:/Users/28428/Desktop/memoryos_demo_v5/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
    embed_dimension: int = int(os.environ.get("SMU_EMBED_DIM", "1024"))
    embed_use_fp16: bool = os.environ.get("SMU_EMBED_USE_FP16", "1") == "1"

    # LLM QC - Fill in your OpenAI details here or set as environment variables
    llm_provider: str = os.environ.get("SMU_LLM_PROVIDER", "openai")  # "openai" or "none"
    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY", "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")
    openai_api_base: Optional[str] = os.environ.get("OPENAI_API_BASE", "https://apic1.ohmycdn.com/v1")
    llm_model_name: str = os.environ.get("SMU_LLM_MODEL", "gpt-5-chat-latest")

    # Retrieval
    kappa: float = float(os.environ.get("SMU_KAPPA", "6.0"))
    bias_b: float = float(os.environ.get("SMU_BIAS_B", "0.0"))
    epsilon: float = float(os.environ.get("SMU_EPS", "1e-9"))

    # Merge
    merge_top_k: int = 3

    # Query encoding fusion weights
    task_weight: float = float(os.environ.get("SMU_TASK_WEIGHT", "0.7"))
    profile_weight: float = float(os.environ.get("SMU_PROFILE_WEIGHT", "0.3"))


def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.data_dir, exist_ok=True)
