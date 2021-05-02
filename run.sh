#!/bin/bash -eu

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export FF_HOME="$DIR"

cd "$FF_HOME"
# source "$FF_HOME/env.sh"

set -euo pipefail
cd "$FF_HOME/examples/python/onnx/"
#export NORMAL_CELLS=5
export STRATEGY_DIR="$FF_HOME/data/strategies/"
export OUTPUT_DIR="$FF_HOME/data/outputs/"
export TASKGRAPH_DIR="$FF_HOME/data/taskgraphs/"
# export SEARCH_PROBLEM_DIR="$FF_HOME/data/search-problems/"
export SEARCH_CURVE_DIR="$FF_HOME/data/search-curves/"
export RESULTS_DIR="$FF_HOME/data/results/"
for d in "$STRATEGY_DIR" "$OUTPUT_DIR" "$TASKGRAPH_DIR" "$RESULTS_DIR" "$SEARCH_CURVE_DIR"; do
  mkdir -p "$d";
done

export OPTIMIZED=0
export BATCH_SIZE=16
export NUM_GPUS=4
export NUM_NODES=1
export BUDGET=1
export BANDWIDTH=20
export MODEL_TAG=""
export ALPHA="1.0"
export MODEL_NUM_GPUS="$NUM_GPUS"
if [[ $OPTIMIZED == 1 ]]; then
  OPTSTRING="optimized"
else
  OPTSTRING="unoptimized"
fi
export BASENAME="resnext50${MODEL_TAG}_${OPTSTRING}_n${NUM_NODES}_g${NUM_GPUS}_b${BATCH_SIZE}_bw${BANDWIDTH}_bu${BUDGET}_alpha${ALPHA//\./p}"
echo "$BASENAME"

"$FF_HOME/python/flexflow_python" \
  "$FF_HOME/examples/python/onnx/resnext50.py" \
    -ll:py 1 \
    -ll:gpu "$NUM_GPUS" \
    -ll:csize 40000 \
    -ll:zsize 16384 \
    -ll:fsize 13000 \
    -ll:streams 1 \
    --batch-size "$BATCH_SIZE" \
    --epochs 1 \
    --nodes 1 \
    --overlap \
    --search-alpha "$ALPHA" \
    --budget "$BUDGET" \
    --taskgraph "$TASKGRAPH_DIR/${BASENAME}_found.dot" \
    --export-strategy "$STRATEGY_DIR/${BASENAME}_found.strategy" \
    --search-curve "$SEARCH_CURVE_DIR/${BASENAME}_found.csv" \
    --search-curve-interval 20 \
    -ll:util 4 | tee "$OUTPUT_DIR/${BASENAME}_found.txt"

# export OPTIMIZED="old"
# "$HOME/FlexFlow/python/flexflow_python" \
#   "$HOME/FlexFlow/examples/python/onnx/nasnet_a.py" \
#     -ll:py 1 \
#     -ll:gpu "$NUM_GPUS" \
#     -ll:csize 40000 \
#     -ll:zsize 16384 \
#     -ll:fsize 14000 \
#     -ll:streams 1 \
#     --batch-size "$BATCH_SIZE" \
#     --budget "$BUDGET" \
#     --epochs 1 \
#     --overlap \
#     --nodes 1 \
#     --taskgraph "$TASKGRAPH_DIR/${BASENAME}_old.dot" \
#     --export-strategy "$STRATEGY_DIR/${BASENAME}_old.strategy" \
#     --search-alpha "$ALPHA" \
#     --search-problem "$SEARCH_PROBLEM_DIR/${BASENAME}_old.problem" \
#     --bandwidth "$BANDWIDTH" \
#     -ll:util 4 | tee "$OUTPUT_DIR/${BASENAME}_old.txt"

RESULTS_FILE="$RESULTS_DIR/${BASENAME}.txt"
echo "" > "$RESULTS_FILE"
cat "$OUTPUT_DIR/${BASENAME}_found.txt" | grep 'iter(0)' >> "$RESULTS_FILE"
cat "$OUTPUT_DIR/${BASENAME}_found.txt" | grep -B 1 'Best' | head -n 1 >> "$RESULTS_FILE"
cat "$OUTPUT_DIR/${BASENAME}_old.txt" | grep -B 1 'Best' | head -n 1 >> "$RESULTS_FILE"
echo ""
echo "RESULTS:"
cat "$RESULTS_FILE"
