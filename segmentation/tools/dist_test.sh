#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
# Adapted for the DuoMamba ACCV 2026 release.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 CONFIG CHECKPOINT NUM_GPUS [OPTIONS...]" >&2
    exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3
export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"
python -m torch.distributed.run \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --nproc_per_node="$GPUS" \
    --master_port="${PORT:-29500}" \
    "$SCRIPT_DIR/test.py" "$CONFIG" "$CHECKPOINT" --launcher pytorch "$@"
