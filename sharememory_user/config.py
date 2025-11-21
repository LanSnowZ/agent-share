import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    data_dir: str = os.environ.get("SMU_DATA_DIR", "sharememory_user/data")
    memory_path: str = os.path.join(data_dir, "memory.json")
    users_path: str = os.path.join(data_dir, "users.json")

    # Embedding
    embed_use_hf: bool = os.environ.get("SMU_EMBED_USE_HF", "1") == "1"
    embed_model_name: str = os.environ.get(
        "SMU_EMBED_MODEL",
        "/root/autodl-tmp/models/embedding_models/bge-m3",
    )
    embed_dimension: int = int(os.environ.get("SMU_EMBED_DIM", "1024"))
    embed_use_fp16: bool = os.environ.get("SMU_EMBED_USE_FP16", "1") == "1"

    # LLM QC - Fill in your OpenAI details here or set as environment variables
    llm_provider: str = os.environ.get(
        "SMU_LLM_PROVIDER", "openai"
    )  # "openai" or "none"
    openai_api_key: Optional[str] = os.environ.get(
        "OPENAI_API_KEY", "sk-UNPI9WFKD0096181Cc77T3BlBkFJ59170fbdfC5c4dE98536"
    )
    openai_api_base: Optional[str] = os.environ.get(
        "OPENAI_API_BASE", "https://cn2us02.opapi.win/v1"
    )
    llm_model_name: str = os.environ.get("SMU_LLM_MODEL", "gpt-5-chat-latest")

    # Retrieval - Hybrid strategy (focus_query + COT)
    hybrid_recall_k: int = int(os.environ.get("SMU_HYBRID_RECALL_K", "30"))  # 第一步召回的候选数量
    focus_query_weight: float = float(os.environ.get("SMU_FOCUS_QUERY_WEIGHT", "0.6"))  # focus_query权重
    cot_weight: float = float(os.environ.get("SMU_COT_WEIGHT", "0.4"))  # COT权重

    # Merge
    merge_top_k: int = 3

    # Query encoding fusion weights
    task_weight: float = float(os.environ.get("SMU_TASK_WEIGHT", "0.7"))
    profile_weight: float = float(os.environ.get("SMU_PROFILE_WEIGHT", "0.3"))


def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.data_dir, exist_ok=True)
