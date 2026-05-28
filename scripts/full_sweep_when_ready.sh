#!/usr/bin/env bash
# Wait until 104 EEGMMI subjects are present in data/raw/eegmmi/, then run the
# full baseline sweep (LOSO + K-trials + within-subject LSO). Logs go to
# results/offline_runs/ via the OfflineLogger.
#
# Usage:
#   bash scripts/full_sweep_when_ready.sh                 # offline logger
#   bash scripts/full_sweep_when_ready.sh wandb           # W&B logger
#
# Expected wall-clock: 5-10 hours on Apple Silicon depending on FM finetune
# config. The cheap classical + frozen-FM LOSO sweep alone is ~2 hours.

set -e

LOGGER="${1:-offline}"
RAW_DIR="data/raw/eegmmi"
TARGET=100  # 104 subjects minus 4-5 that may straggle; treat 100+ as good enough

echo "Waiting for $RAW_DIR to reach $TARGET+ subjects..."
until [ "$(ls "$RAW_DIR" 2>/dev/null | wc -l | tr -d ' ')" -ge "$TARGET" ]; do
  sleep 30
done

echo "Running full sweep (suite=all, logger=$LOGGER)..."
uv run python scripts/run_full_baselines.py --suite all --logger "$LOGGER"
