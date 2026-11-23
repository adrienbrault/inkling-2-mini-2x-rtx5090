#!/usr/bin/env bash
# Build the serving image: vLLM's Inkling-2-Mini day-0 image (pinned by digest in docker/Dockerfile) plus the SM120
# overlay in docker/ (see docker/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -f docker/Dockerfile -t "${IMAGE:-inkling2-rtx5090:sm120-r14}" .
