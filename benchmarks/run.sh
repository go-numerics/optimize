#!/bin/sh
# Interleaved Go vs SciPy benchmark rounds; medians in benchmarks/results.json.
# Usage: benchmarks/run.sh [rounds]   (needs uv; GOWORK is turned off)
set -e
cd "$(dirname "$0")/.."
R=${1:-7}
OUT=benchmarks/raw
mkdir -p "$OUT"
rm -f "$OUT"/*.txt "$OUT"/*.json
GOWORK=off go test -c -o "$OUT/opt.test" .
for i in $(seq 1 "$R"); do
  echo "round $i/$R ($(uptime | sed 's/.*load/load/'))"
  "$OUT/opt.test" -test.run '^$' -test.bench . -test.benchtime=300ms -test.count=1 > "$OUT/go-$i.txt"
  uv run -q --with scipy --with numpy python benchmarks/bench_scipy.py 3 > "$OUT/py-$i.json"
done
uv run -q --with numpy python benchmarks/summarize.py "$OUT" > benchmarks/results.json
cat benchmarks/results.json
