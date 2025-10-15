# sharememory_user

Zero-train multi-user shared memory with A→C selection (A→B→C preserved). Stores to JSON; embeddings and QC have offline fallbacks.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Initialize storage
PYTHONPATH=. python -m sharememory_user.cli init

# Ingest a dialog
PYTHONPATH=. python -m sharememory_user.cli ingest --user_id u1 --profile sample_data/user_u1_profile.json --dialog sample_data/dialog1.txt

# Retrieve for a user with peers
PYTHONPATH=. python -m sharememory_user.cli retrieve --user_id u1 --task "deploy X to Y and autoscale on Z" --peers sample_data/peers.json --top_k 5

# Build prompt blocks (COT + KG)
PYTHONPATH=. python -m sharememory_user.cli prompt --user_id u1 --task "deploy X to Y and autoscale on Z" --peers sample_data/peers.json --top_k 5
```

## Embeddings: BGE-M3 by default
- Default model: `BAAI/bge-m3` (`SMU_EMBED_MODEL`), with `SMU_EMBED_USE_HF=1`.
- Fallback: hashing embeddings with `SMU_EMBED_DIM` (default 1024) when HF not available.

## Env toggles
- `SMU_EMBED_USE_HF=1` to use HF; `SMU_EMBED_MODEL=BAAI/bge-m3` to set model.
- `SMU_EMBED_DIM` to set fallback embedding dim.
"# agent-share" 
