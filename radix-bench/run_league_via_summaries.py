import os, time, pathlib, subprocess, yaml, re, statistics as st

CFG_PATH = pathlib.Path("configs/experiments/smoke.yaml")
OUTDIR = pathlib.Path("results") / ("league-manual-" + time.strftime("%Y%m%d-%H%M%S"))
OUTDIR.mkdir(parents=True, exist_ok=True)
SEEDS = [1337, 1338, 1339]
NAMES = ["fifo","srpt","drf","easy","heft","bfd","gavel","radix"]

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
    text = pathlib.Path("results/summary.md").read_text()
    (OUTDIR / f"summary_{seed}.md").write_text(text)
    return text

def parse_summary(text):
    rows = {}
    lines = text.splitlines()
    for line in lines:
        sline = line.strip().lower()
        for name in NAMES:
            if name in sline.split():
                m = re.search(r"\b" + re.escape(name) + r"\b\s+([0-9]+(?:\.[0-9]+)?)", sline)
                if m:
                    rows[name] = float(m.group(1))
    return rows

per_sched = {}
for seed in SEEDS:
    txt = run_once(seed)
    vals = parse_summary(txt)
    if not vals:
        raise SystemExit("No rows parsed for seed " + str(seed))
    for k,v in vals.items():
        per_sched.setdefault(k, []).append(v)

if "fifo" not in per_sched or len(per_sched["fifo"]) == 0:
    raise SystemExit("FIFO not found in summaries")

fifo_mean = st.mean(per_sched["fifo"])
rows = []
for s, arr in per_sched.items():
    mean = st.mean(arr)
    imp = ((mean - fifo_mean) / fifo_mean * 100.0) if fifo_mean else 0.0
    rows.append((s, mean, imp))
rows.sort(key=lambda x: x[1], reverse=True)

md = []
md.append("Manual League Throughput seeds: " + ", ".join(map(str, SEEDS)))
md.append("")
md.append("| Rank | Scheduler | Mean Throughput | vs FIFO |")
md.append("|---:|---|---:|---:|")
for i, (s, mean, imp) in enumerate(rows, start=1):
    md.append(f"| {i} | {s} | {mean:.3f} | {imp:+.1f}% |")

league_md = OUTDIR / "league.md"
league_md.write_text("\n".join(md) + "\n")
print(str(league_md))
print("\n".join(league_md.read_text().splitlines()[:12]))
