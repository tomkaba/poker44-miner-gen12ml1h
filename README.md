# poker44-miner-gen10heur1

Minimal release repository for model gen10heur1.

## Quick start

```bash
cd poker44-miner-gen10heur1
source .venv-1/bin/activate
pip install -e .
./start_miner.sh HOTKEY_ID
```

## Launch

```bash
./start_miner.sh HOTKEY_ID[,HOTKEY_ID2,...]
```

Examples:

```bash
./start_miner.sh 214
./start_miner.sh 11,14,22
```

## Environment (optional)

- POKER44_WALLET_NAME (default: sn126b)
- POKER44_SESSION_PREFIX (default: sn126b_m)
- POKER44_AXON_BASE_PORT (default: 12080)
- POKER44_VENV_BIN (default: .venv-1/bin)
- POKER44_MODEL_REPO_URL (default: repo URL)
- POKER44_MODEL_REPO_COMMIT (default: current git commit)

The launcher computes POKER44_MODEL_IMPLEMENTATION_SHA256 at runtime from:

- neurons/miner.py
- poker44/miner_heuristics.py
- models/benchmark_heuristic_profile.json
