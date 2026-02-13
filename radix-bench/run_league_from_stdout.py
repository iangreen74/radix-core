import os, time, pathlib, subprocess, yaml, re, statistics as st

CFG_PATH = pathlib.Path("configs/experiments/smoke.yaml")
SEEDS = [1337, 1338, 1339]
OUTDIR = pathlib.Path("results") / ("league-manual-" + time.strftime("%Y%m%d-%H%M%S"))
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
    return stdout

def parse_table(text):
    vals = {}
    for line in text.splitlines():
        m = row_re.match(line)
        if m:
            sched = m.group(1).lower()
            thr = float(m.group(2))
            vals[sched] = thr
    return vals

per_sched = {}
for seed in SEEDS:
    out = run_and_capture(seed)
    rows = parse_table(out)
    if not rows:
        raise SystemExit("No rows parsed from stdout for seed " + str(seed))
    for s,v in rows.items():
        per_sched.setdefault(s, []).append(v)

if "fifo" not in per_sched or len(per_sched["fifo"]) == 0:
    raise SystemExit("FIFO not found; cannot compute improvement")

fifo_mean = st.mean(per_sched["fifo"])
agg = []
for s, arr in per_sched.items():
    mean = st.mean(arr)
    imp = ((mean - fifo_mean) / fifo_mean * 100.0) if fifo_mean else 0.0
    agg.append((s, mean, imp))
agg.sort(key=lambda x: x[1], reverse=True)

md_lines = []
md_lines.append("Manual League Throughput seeds: " + ", ".join(map(str, SEEDS)))
md_lines.append("")
md_lines.append("| Rank | Scheduler | Mean Throughput | vs FIFO |")
md_lines.append("|---:|---|---:|---:|")
for i, (s, mean, imp) in enumerate(agg, start=1):
    md_lines.append(f"| {i} | {s} | {mean:.3f} | {imp:+.1f}% |")

league_md = OUTDIR / "league.md"
league_md.write_text("\n".join(md_lines) + "\n")

print(str(league_md))
print("\n".join(md_lines[:12]))
