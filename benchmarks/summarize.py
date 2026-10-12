"""Median of the interleaved rounds written by run.sh."""
import glob, json, os, re, statistics, sys

d = sys.argv[1]
go, py = {}, {}
for f in sorted(glob.glob(os.path.join(d, "go-*.txt"))):
    for line in open(f):
        m = re.match(r"Benchmark(\S+?)(?:-\d+)?\s+\d+\s+([\d.]+) ns/op", line)
        if m:
            go.setdefault(m.group(1), []).append(float(m.group(2)) * 1e-9)
for f in sorted(glob.glob(os.path.join(d, "py-*.json"))):
    for k, v in json.load(open(f)).items():
        py.setdefault(k, []).append(v)
res = {}
for k in go:
    g = statistics.median(go[k])
    s = statistics.median(py[k]) if k in py else None
    cb = statistics.median(py[k + "/callbacks"]) if k + "/callbacks" in py else None
    res[k] = dict(go_s=g, scipy_s=s, scipy_callbacks_s=cb, rounds=len(go[k]),
                  speedup=(s / g if s else None),
                  speedup_excl_callbacks=((s - cb) / g if s and cb is not None else None))
print(json.dumps(res, indent=1))
