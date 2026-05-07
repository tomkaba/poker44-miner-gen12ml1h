#!/bin/bash
scp neurons/miner.py tk@cx1:Poker44-subnet-main/neurons/miner.py
scp poker44/miner_heuristics.py tk@cx1:Poker44-subnet-main/poker44/miner_heuristics.py
scp poker44/utils/model_manifest.py tk@cx1:Poker44-subnet-main/poker44/utils/model_manifest.py
scp start_miner2.sh tk@cx1:Poker44-subnet-main/
scp models/benchmark_heuristic_profile.json tk@cx1:Poker44-subnet-main/models/benchmark_heuristic_profile.json
scp models/benchmark_lgbm_model.pkl tk@cx1:Poker44-subnet-main/models/benchmark_lgbm_model.pkl
scp models/benchmark_lgbm_profile.json tk@cx1:Poker44-subnet-main/models/benchmark_lgbm_profile.json
