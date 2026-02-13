import os, json, time, pathlib, statistics as st, subprocess, yaml

CFG_PATH = pathlib.Path("configs/experiments/smoke.yaml")
OUTDIR = pathlib.Path("results") / ("league-manual-" + time.strftime("%Y%m%d-%H%M%S"))
OUTDIR.mkdir(parents=True, exist_ok=True)
SEEDS = [1337, 1338, 1339]
KNOWN = {"fifo","srpt","drf","easy","heft","bfd","gavel","radix"}

def set_cfg_seed(seed):
    d = yaml.safe_load(CFG_PATH.read_text())
    d.setdefault("simulation", {})["seed"] = int(seed)
    d["simulation"]["dry_run"] = False
    d.setdefault("cost_model", {})["cost_cap_usd"] = 1000000000.0
    CFG_PATH.write_text(yaml.safe_dump(d, sort_keys=False))

def run_once(seed):
    set_cfg_seed(seed)
    env = os.environ.copy()
    env.setdefault("LC_ALL","C")
    env.setdefault("TZ","UTC")
    env.setdefault("PYTHONHASHSEED","0")
    env.setdefault("SOURCE_DATE_EPOCH","1609459200")
    subprocess.run(["poetry","run","python","-m","radixbench.cli.simulate","run",str(CFG_PATH)], check=True, env=env)
    (OUTDIR / f"summary_{seed}.md").write_text(pathlib.Path("results/summary.md").read_text())
    raw = json.loads(pathlib.Path("results/results.json").read_text())
    (OUTDIR / f"results_{seed}.json").write_text(json.dumps(raw, indent=2))
    return raw

def extract_throughputs(raw):
    out = {}
    def rec(x, key_hint=None):
        if isinstance(x, dict):
            if "metrics" in x and isinstance(x["metrics"], dict) and "throughput" in x["metrics"]:
                thr = x["metrics"]["throughput"]
                if isinstance(thr,(int,float)):
                    name = x.get("scheduler") or key_hint
                    if not name and key_hint in KNOWN:
                        name = key_hint
                    if name:
                        out[str(name)] = float(thr)
            for k,v in x.items():
                rec(v, k if isinstance(k,str) else key_hint)
        elif isinstance(x, list):
            for v in x:
                rec(v, key_hint)
    rec(raw, None)
    return out

tp = {}
for s in SEEDS:
    raw = run_once(s)
    thr = extract_throughputs(raw)
    if not thr:
        raise SystemExit(f"No throughput parsed for seed {s}. See {OUTDIR}/results_{s}.json")
    for k,v in thr.items():
        tp.setdefault(k, []).append(v)

schedulers = sorted(tp.keys())
fifo_vals = tp.get("fifo", [])
fifo_mean = st.mean(fifo_vals) if fifo_vals else 0.0

rows = []
for s in schedulers:
    vals = tp[s]
    mean = st.mean(vals) if vals else 0.0
    imp = ((mean - fifo_mean)/fifo_mean*100.0) if fifo_mean else 0.0
    rows.append((s, mean, imp))
rows.sort(key=lambda x: x[1], reverse=True)

md = []
md.append("Manual League (Throughput) seeds: " + ", ".join(map(str,SEEDS)))
md.append("")
md.append("| Rank | Scheduler | Mean Throughput | vs FIFO |")
md.append("|---:|---|---:|---:|")
for i,(s,mean,imp) in enumerate(rows, start=1):
    md.append(f"| {i} | {s} | {mean:.3f} | {imp:+.1f}% |")

league_md = OUTDIR / "league.md"
league_md.write_text("\n".join(md) + "\n")
print(str(league_md))
print("\n".join(league_md.read_text().splitlines()[:12]))
