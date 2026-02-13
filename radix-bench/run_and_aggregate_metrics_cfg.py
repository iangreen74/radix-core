import os, sys, time, pathlib, subprocess, yaml, json

CFG = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/smoke.yaml")
SEEDS = [1337, 1338, 1339]
OUTDIR = pathlib.Path("results") / ("league-metrics-" + CFG.stem + "-" + time.strftime("%Y%m%d-%H%M%S"))
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
                found.append((str(x["scheduler"]).lower(), x["metrics"]))
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

def mean(arr):
    return sum(arr)/len(arr) if arr else None

fifo_thr = mean(agg.get("fifo", {}).get("throughput", []))

rows = []
for name, m in agg.items():
    thr = mean(m.get("throughput", []))
    wait = mean(m.get("avg_wait_time", []))
    tat = mean(m.get("avg_turnaround_time", []))
    runtime = mean(m.get("avg_runtime", []))
    ctot = mean(m.get("total_cost_usd", []))
    cavg = mean(m.get("avg_cost_per_job", []))
    imp = None
    if fifo_thr and thr is not None:
        imp = ((thr - fifo_thr)/fifo_thr)*100.0
    rows.append((name, thr, imp, wait, tat, runtime, ctot, cavg))

rows = [r for r in rows if r[1] is not None]
rows.sort(key=lambda x: x[1], reverse=True)

def fmt(v):
    return f"{v:.3f}" if isinstance(v,(int,float)) else ""

lines = []
lines.append("Metrics across seeds " + ", ".join(map(str, SEEDS)) + " config " + str(CFG))
lines.append("")
lines.append("| Scheduler | thr | vs FIFO | wait | tat | runtime | cost_total | cost_avg |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for name, thr, imp, wait, tat, runtime, ctot, cavg in rows:
    lines.append("| " + " | ".join([
        name, fmt(thr), (f"{imp:+.1f}%" if imp is not None else ""),
        fmt(wait), fmt(tat), fmt(runtime), fmt(ctot), fmt(cavg)
    ]) + " |")

md = OUTDIR / "metrics.md"
md.write_text("\n".join(lines) + "\n")
print(str(md))
print("\n".join(lines[:16]))
