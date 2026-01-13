#!/bin/sh
set -eu

# Minimal local-first entrypoint.
# Set WANDB_API_KEY (optional) if you enable W&B in config.
python -u src/models/train_model.py "$@"
