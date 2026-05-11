"""Poker44 miner for standalone gen11lgbm LightGBM release."""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.miner_heuristics import get_chunk_scorer_startup_check, score_chunk_gen11lgbm
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


FORCED_VALIDATOR_HOTKEYS = {
    "5GgnyzhZ6ozkdnQumwRuEaULggvMr2np4SS3N7eDCMMrXoMC",
}

EXTRA_ALLOWED_VALIDATOR_HOTKEYS = {
    "5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD",
}


def _load_env_file():
    """Load .env file from current directory or parent directories."""
    if not HAS_DOTENV:
        return

    search_path = Path.cwd()
    for _ in range(5):
        env_file = search_path / ".env"
        if env_file.exists():
            try:
                load_dotenv(env_file, override=False)
                print(f"[.env] Loaded environment from {env_file}")
                return
            except Exception as e:
                print(f"[.env] Warning: Failed to load {env_file}: {e}")
                return
        search_path = search_path.parent
        if search_path == search_path.parent:
            break


class Miner(BaseMinerNeuron):
    """Deterministic chunk scorer for gen11lgbm LightGBM profile."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("LGBM Poker44 Miner started (gen11lgbm)")

        chunk_scorer = "gen11lgbm"
        bt.logging.info("[init] POKER44_CHUNK_SCORER=gen11lgbm (hardcoded)")

        scorer_check = get_chunk_scorer_startup_check(chunk_scorer)
        if scorer_check.get("active"):
            details = scorer_check.get("details") or {}
            if scorer_check.get("ok"):
                bt.logging.info(
                    "[init] Chunk scorer startup check: ok "
                    f"scorer={scorer_check.get('scorer')} details={details}"
                )
            else:
                bt.logging.error(
                    "[init] Chunk scorer startup check: FAILED "
                    f"scorer={scorer_check.get('scorer')} "
                    f"error={scorer_check.get('error')} details={details}"
                )

        bt.logging.info(f"Axon created: {self.axon}")
        bt.logging.info(f"Build timestamp: {datetime.now(timezone.utc).isoformat()}")

        self._project_root = Path(__file__).resolve().parent.parent
        repo_root = Path(__file__).resolve().parents[1]

        try:
            git_commit = subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
        except Exception:
            git_commit = os.getenv("POKER44_MODEL_REPO_COMMIT", "")

        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[Path(__file__).resolve()],
            defaults={
                "model_name": "poker44_gen11lgbm",
                "model_version": "11.0",
                "framework": "python-lightgbm",
                "license": "MIT",
                "repo_url": "https://github.com/tomkaba/poker44-miner-gen11lgbm",
                "repo_commit": git_commit,
                "notes": "Gen11lgbm LightGBM chunk scorer.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": "No validator-private data used.",
                "private_data_attestation": "This miner does not train on validator-private human data.",
                "data_attestation": "This miner does not train on validator-private human data.",
            },
        )

        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(f"Project root: {repo_root}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks: List[List[dict]] = synapse.chunks or []

        scores = []
        routes = []
        for chunk in chunks:
            score, route = score_chunk_gen11lgbm(chunk)
            scores.append(score)
            routes.append(route)

        chunk_sizes = [len(chunk or []) for chunk in chunks]

        def _preview(values, limit=8):
            if values is None:
                return []
            if len(values) <= limit:
                return values
            return values[:limit] + [f"...(+{len(values) - limit} more)"]

        bt.logging.debug(f"[miner] Received {len(chunks)} chunk(s); first sizes={_preview(chunk_sizes)}")

        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)

        bt.logging.debug(
            f"[DEBUG] Before sending: synapse.risk_scores={synapse.risk_scores}, "
            f"type={type(synapse.risk_scores)}, len={len(synapse.risk_scores) if synapse.risk_scores is not None else 'None'}"
        )
        bt.logging.debug(
            f"[miner] Responding with scores={_preview(scores)} "
            f"routes={_preview(routes)} predictions={_preview(synapse.predictions)}"
        )

        source_hotkey = getattr(getattr(synapse, "dendrite", None), "hotkey", "unknown")
        self._append_request_log(
            validator_hotkey=source_hotkey,
            chunk_sizes=chunk_sizes,
            chunk_routes=routes,
            scores=scores,
            predictions=synapse.predictions,
            chunks=chunks,
        )

        bt.logging.info(f"Scored {len(chunks)} chunks with gen11lgbm LightGBM scorer.")
        return synapse

    @staticmethod
    def _flag_enabled(config_section, attr, default=None):
        value = getattr(config_section, attr, default) if config_section else default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return bool(value)

    def _allowed_validator_hotkeys(self) -> set[str]:
        cfg = getattr(self.config, "blacklist", None)
        allowed = set(FORCED_VALIDATOR_HOTKEYS) | set(EXTRA_ALLOWED_VALIDATOR_HOTKEYS)

        def _normalize(value) -> set[str]:
            if value is None:
                return set()
            if isinstance(value, (list, tuple, set)):
                iterable = value
            else:
                iterable = str(value).split(",")
            return {str(item).strip() for item in iterable if str(item).strip()}

        allowed |= _normalize(getattr(cfg, "forced_validator_hotkey", None))
        allowed |= _normalize(getattr(cfg, "forced_validator_hotkeys", None))
        allowed |= _normalize(getattr(cfg, "extra_validator_hotkeys", None))
        return allowed

    def score_chunk(self, chunk: list[dict]) -> float:
        return score_chunk_gen11lgbm(chunk)[0]

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        allow_non_registered = self._flag_enabled(
            getattr(self.config, "blacklist", None),
            "allow_non_registered",
            False,
        )
        force_validator_permit = self._flag_enabled(
            getattr(self.config, "blacklist", None),
            "force_validator_permit",
            True,
        )
        allowed_hotkeys = self._allowed_validator_hotkeys()

        if synapse.dendrite.hotkey in allowed_hotkeys:
            return False, "Validator allowlist"

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            if not allow_non_registered:
                return True, "Unrecognized hotkey"
            return False, "Unregistered hotkey allowed"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if force_validator_permit and not self.metagraph.validator_permit[uid]:
            return True, "Non-validator hotkey"

        return False, "Hotkey recognized!"

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)

    def _get_log_path(self) -> Path:
        uid = getattr(self, "uid", None)
        suffix = uid if uid is not None else "unknown"
        return self._project_root / f"miner_{suffix}.log"

    def _append_request_log(
        self,
        validator_hotkey,
        chunk_sizes,
        chunk_routes,
        scores,
        predictions,
        chunks,
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "validator_hotkey": validator_hotkey,
            "miner_hotkey": getattr(self.wallet.hotkey, "ss58_address", "unknown"),
            "chunk_count": len(chunk_sizes),
            "chunk_sizes": chunk_sizes,
            "chunk_routes": chunk_routes,
            "scores": scores,
            "predictions": predictions,
            "chunks": chunks,
        }
        try:
            with self._get_log_path().open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as log_error:
            bt.logging.warning(f"Failed to append miner request log: {log_error}")

    def _dump_request_payload(self, *args, **kwargs):
        return


if __name__ == "__main__":
    _load_env_file()

    with Miner() as miner:
        bt.logging.info("LGBM miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {float(miner.metagraph.I[miner.uid])} | Scorer: gen11lgbm"
            )
            time.sleep(60)
