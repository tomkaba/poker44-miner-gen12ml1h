# Poker44-gen11lgbm

Minimal release repository for model gen11lgbm.

This repo is a standalone miner variant, extracted analogously to gen10heur releases,
but wired to the LightGBM profile/model artifacts used by gen8lgbm.

## Quick start

```bash
git clone https://github.com/tomkaba/poker44-miner-gen11lgbm.git
cd poker44-miner-gen11lgbm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
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

- Scorer: score_chunk_gen11lgbm() in poker44/miner_heuristics.py
- Artifacts:
  - models/benchmark_lgbm_profile.json
  - models/benchmark_lgbm_model.pkl
- Entry point: neurons/miner.py

Manifest implementation SHA256 is computed from:

- neurons/miner.py
- poker44/miner_heuristics.py
- models/benchmark_lgbm_profile.json
- models/benchmark_lgbm_model.pkl
