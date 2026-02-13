import os, time, pathlib, subprocess, yaml, re, statistics as st, json

CFG_PATH = pathlib.Path("configs/experiments/smoke.yaml")
SEEDS = [1337, 1338, 1339]
OUTDIR = pathlib.Path("results") / ("league-metrics-" + time.strftime("%Y%m%d-%H%M%S"))
OUTDIR.mkdir(parents=True, exist_ok=True)

row_re = re.compile(r"^\s*\d+\s+([a-z0-9_]+)\s+([0-9]+(?:\.[0-9]+)?)\b", re.I)

def set_cfg_seed(seed):
    d = yaml.safe_load(CFG_PATH.read_text())
    d.setdefault("simulation", {})["seed"] = int(seed)
    d["simulation"]["dry_run"] = False
    d.setdefault("cost_model", {})["cost_cap_usd"] = 1000000000.0
    CFG_PATH.write_text(yaml.safe_dump(d, sort_keys=False))

def run_and_capture(seed):
    set_cfg_seed(seed)
    env = os.environ.copy()
    env.setdefault("LC_ALL","C")
    env.setdefault("TZ","UTC")
    env.setdefault("PYTHONHASHSEED","0")
    env.setdefault("SOURCE_DATE_EPOCH","1609459200")
    p = subprocess.run(
        ["poetry","run","python","-m","radixbench.cli.simulate","run",str(CFG_PATH)],
        check=True, env=env, text=True, capture_output=True
    )
    stdout = p.stdout
    (OUTDIR / f"stdout_{seed}.txt").write_text(stdout)
    raw = json.loads(pathlib.Path("results/results.json").read_text())
    (OUTDIR / f"results_{seed}.json").write_text(json.dumps(raw, indent=2))
    return stdout, raw

def collect_metrics(raw):
    items = []
    def rec(x):
        if isinstance(x, dict):
            if "scheduler" in x and "metrics" in x and isinstance(x["metrics"], dict):
                m = x["metrics"]
                items.append((x["scheduler"], {
                    "throughput": m.get("throughput"),
                    "avg_wait_time": m.get("avg_wait_time"),
                    "avg_turnaround_time": m.get("avg_turnaround_time"),
                    "avg_runtime": m.get("avg_runtime"),
                    "total_cost_usd": m.get("total_cost_usd"),
                    "avg_cost_per_job": m.get("avg_cost_per_job"),
                }))
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)
    rec(raw)
    out = {}
    for name, md in items:
        out.setdefault(name.lower(), {}).update({k: v for k,v in md.items() if v is not None})
    return out

agg = {}
for seed in SEEDS:
    stdout, raw = run_and_capture(seed)
    per = collect_metrics(raw)
    for name, md in per.items():
        for k, v in md.items():
            if isinstance(v, (int, float)):
                agg.setdefault(name, {}).setdefault(k, []).append(float(v))

summary = []
for name, m in sorted(agg.items()):
    row = {"scheduler": name}
    for k, arr in m.items():
        row[k] = sum(arr) / len(arr)
    summary.append(row)

def fmt(v):
    return f"{v:.3f}" if isinstance(v,(int,float)) else str(v)

lines = []
lines.append("| Scheduler | thr | wait | tat | runtime | cost_total | cost_avg |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for r in sorted(summary, key=lambda x: x.get("throughput", 0), reverse=True):
    lines.append("| " + " | ".join([
        r["scheduler"],
        fmt(r.get("throughput","")),
        fmt(r.get("avg_wait_time","")),
        fmt(r.get("avg_turnaround_time","")),
        fmt(r.get("avg_runtime","")),
        fmt(r.get("total_cost_usd","")),
        fmt(r.get("avg_cost_per_job","")),
    ]) + " |")

md_path = OUTDIR / "metrics.md"
md_path.write_text("\n".join(lines) + "\n")
print(str(md_path))
print("\n".join(lines[:12]))
