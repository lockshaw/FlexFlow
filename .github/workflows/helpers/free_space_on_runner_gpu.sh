#! /usr/bin/env bash
set -euo pipefail
set -x

sudo rm -rf \
  /usr/local/cuda-12.1 \
  /usr/local/cuda-12.2 \
  /usr/local/cuda-12.3 \
  /usr/local/cuda-12.5
