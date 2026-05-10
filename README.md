# poker44-miner-gen10heur1

Minimal release repository for model gen10heur1 — pure heuristic scorer.

## Quick start

```bash
git clone https://github.com/tomkaba/poker44-miner-gen10heur1.git
cd poker44-miner-gen10heur1
python3 -m venv .venv-1
source .venv-1/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run Miner

### Method 1: Direct Python (Recommended)

Use environment variables or `.env` file:

```bash
# Option A: Command line
WALLET_NAME=my_cold \
HOTKEY=my_poker44_hotkey \
AXON_PORT=8091 \
ALLOWED_VALIDATOR_HOTKEYS="validator_hotkey_1 validator_hotkey_2" \
python neurons/miner.py

# Option B: From .env file
cp .env.example .env
# Edit .env with your values
python neurons/miner.py
```

### Method 2: Shell script wrapper (Legacy)

```bash
./start_miner.sh HOTKEY_ID[,HOTKEY_ID2,...]
```

Examples:

```bash
./start_miner.sh 214
./start_miner.sh 11,14,22
```

## Configuration

### Via Environment Variables

- `WALLET_NAME` — Bittensor wallet name (required)
- `HOTKEY` — Wallet hotkey (required)
- `AXON_PORT` — Port for miner axon (required, e.g., 8091)
- `ALLOWED_VALIDATOR_HOTKEYS` — Space/comma-separated validator hotkeys (optional)
- `SUBTENSOR_NETWORK` — Network (mainnet, testnet, finney, local; default: finney)
- `NETUID` — Subnet ID (default: 126)

### Via .env File

Copy `.env.example` to `.env` and fill in your values. The miner searches for `.env` in the current directory or parent directories.

### Optional Features

- `POKER44_DISABLE_FULL_LOGS` — Set to `true` to disable full request logging (default: false)

## Implementation

Gen10heur1 is a **pure heuristic scorer** with no machine learning components:

- **Scorer**: `score_chunk_gen10heur1()` in `poker44/miner_heuristics.py`
- **Profile**: `models/benchmark_heuristic_profile.json`
- **Entry point**: `neurons/miner.py` (Bittensor miner)

The miner computes an implementation SHA256 from:

- `neurons/miner.py`
- `poker44/miner_heuristics.py`
- `models/benchmark_heuristic_profile.json`

This SHA256 is published in `models/model_manifest.json` for transparency.
