import os, time, pathlib, subprocess, yaml, json, statistics as st

CFG = pathlib.Path("configs/experiments/smoke.yaml")
SEEDS = [1337, 1338, 1339]
OUTDIR = pathlib.Path("results") / ("league-metrics-" + time.strftime("%Y%m%d-%H%M%S"))
OUTDIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    d = yaml.safe_load(CFG.read_text())
    d.setdefault("simulation", {})["seed"] = int(seed)
    d["simulation"]["dry_run"] = False
    d.setdefault("cost_model", {})["cost_cap_usd"] = 1000000000.0
    CFG.write_text(yaml.safe_dump(d, sort_keys=False))

def run_once(seed):
    set_seed(seed)
    env = os.environ.copy()
    env.setdefault("LC_ALL","C")
    env.setdefault("TZ","UTC")
    env.setdefault("PYTHONHASHSEED","0")
    env.setdefault("SOURCE_DATE_EPOCH","1609459200")
    subprocess.run(
        ["poetry","run","python","-m","radixbench.cli.simulate","run",str(CFG)],
        check=True, text=True, capture_output=True, env=env
    )
    raw = json.loads(pathlib.Path("results/results.json").read_text())
    (OUTDIR / f"results_{seed}.json").write_text(json.dumps(raw, indent=2))
    return raw

def harvest(raw):
    found = []
    def rec(x):
        if isinstance(x, dict):
            if "scheduler" in x and "metrics" in x and isinstance(x["metrics"], dict):
                m = x["metrics"]
                found.append((str(x["scheduler"]).lower(), m))
            elif "scheduler_name" in x and "throughput" in x:
                found.append((str(x["scheduler_name"]).lower(), x))
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)
    rec(raw)
    merged = {}
    for name, m in found:
        merged.setdefault(name, {}).update(m)
    return merged

agg = {}
for s in SEEDS:
    data = run_once(s)
    metrics = harvest(data)
    for name, m in metrics.items():
        for key in ["throughput","avg_wait_time","avg_turnaround_time","avg_runtime","total_cost_usd","avg_cost_per_job"]:
            v = m.get(key, None)
            if isinstance(v, (int, float)):
                agg.setdefault(name, {}).setdefault(key, []).append(float(v))

def mean_or_blank(arr):
    return f"{sum(arr)/len(arr):.3f}" if arr else ""

fifo_mean = None
if "fifo" in agg and "throughput" in agg["fifo"]:
    a = agg["fifo"]["throughput"]
    fifo_mean = sum(a)/len(a) if a else None

rows = []
for name, m in agg.items():
    thr = mean_or_blank(m.get("throughput", []))
    wait = mean_or_blank(m.get("avg_wait_time", []))
    tat = mean_or_blank(m.get("avg_turnaround_time", []))
    runtime = mean_or_blank(m.get("avg_runtime", []))
    ctot = mean_or_blank(m.get("total_cost_usd", []))
    cavg = mean_or_blank(m.get("avg_cost_per_job", []))
    imp = ""
    if fifo_mean and m.get("throughput"):
        mean_thr = sum(m["throughput"])/len(m["throughput"])
        imp = f"{((mean_thr - fifo_mean)/fifo_mean*100.0):+.1f}%"
    rows.append((name, thr, imp, wait, tat, runtime, ctot, cavg))

rows.sort(key=lambda x: float(x[1]) if x[1] else -1.0, reverse=True)

lines = []
lines.append("Metrics across seeds " + ", ".join(map(str, SEEDS)))
lines.append("")
lines.append("| Scheduler | thr | vs FIFO | wait | tat | runtime | cost_total | cost_avg |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    lines.append("| " + " | ".join(r) + " |")

md = OUTDIR / "metrics.md"
md.write_text("\n".join(lines) + "\n")
print(str(md))
print("\n".join(lines[:12]))
