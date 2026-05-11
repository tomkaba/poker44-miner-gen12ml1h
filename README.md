# Poker44-gen12ml1h

Minimal release repository for model gen12ml1h.

This repo is a standalone miner variant extracted from the main subnet codebase,
with only ml1h scoring logic enabled.

## Quick start

```bash
git clone https://github.com/tomkaba/poker44-miner-gen12ml1h.git
cd poker44-miner-gen12ml1h
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Model artifact is tracked with Git LFS. On a fresh host run:

```bash
# git-lfs is a system binary (not a Python package), so pip will not install it.
sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install
git lfs pull --include weights/ml_gen5_s123467_model.pkl
```

## Run Miner

```bash
python neurons/miner.py
```

or legacy wrapper:

```bash
./start_miner.sh HOTKEY_ID[,HOTKEY_ID2,...]
```

## Implementation

- Scorer: score_chunk_ml1h_with_route() in poker44/miner_heuristics.py
- Artifacts:
  - weights/ml_gen5_s123467_model.pkl
  - weights/ml_gen5_s123467_scaler.pkl
- Entry point: neurons/miner.py

Manifest implementation SHA256 is computed from:

- neurons/miner.py
- poker44/miner_heuristics.py
- weights/ml_gen5_s123467_model.pkl
- weights/ml_gen5_s123467_scaler.pkl
