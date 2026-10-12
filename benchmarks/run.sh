#!/bin/sh
# Interleaved Go vs SciPy benchmark rounds; medians in benchmarks/results.json.
# Usage: benchmarks/run.sh [rounds] [regexp]   (needs uv; GOWORK is turned off)
# The Go side runs on one thread (-test.cpu 1), as SciPy does; the optional
# regexp selects benchmarks on both sides.
set -e
cd "$(dirname "$0")/.."
R=${1:-7}
BENCH=${2:-.}
OUT=benchmarks/raw
mkdir -p "$OUT"
rm -f "$OUT"/*.txt "$OUT"/*.json
GOWORK=off go test -c -o "$OUT/opt.test" .
PY="uv run -q --with scipy==1.18.1 --with numpy==2.5.4 python"
for i in $(seq 1 "$R"); do
  echo "round $i/$R ($(uptime | sed 's/.*load/load/'))"
  "$OUT/opt.test" -test.run '^$' -test.bench "$BENCH" -test.benchtime=300ms -test.count=1 -test.cpu 1 > "$OUT/go-$i.txt"
  BENCH="$BENCH" $PY benchmarks/bench_scipy.py 3 > "$OUT/py-$i.json"
done
uv run -q --with numpy==2.5.4 python benchmarks/summarize.py "$OUT" > benchmarks/results.json
cat benchmarks/results.json
