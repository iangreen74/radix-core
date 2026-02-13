import os, json, time, threading
from fastapi import FastAPI
from kubernetes import client, config, watch
from pathlib import Path

NS = os.environ.get("POD_NAMESPACE","default")
RET_DAYS = int(os.environ.get("RETENTION_DAYS","7"))
TS_DIR = Path(os.environ.get("TS_DIR","/var/radix/ts"))
TS_DIR.mkdir(parents=True, exist_ok=True)
TS_FILE = TS_DIR / "radix_timeseries.jsonl"

app = FastAPI()

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.get("/v1/preview")
def preview():
    # Instant estimate using API facts (nodes/pods) only
    try:
        cfg_loaded=False
        try:
            config.load_incluster_config(); cfg_loaded=True
        except: config.load_kube_config(); cfg_loaded=True
        v1=client.CoreV1Api()
        nodes=v1.list_node().items
        pods=v1.list_pod_for_all_namespaces().items
        gpu_nodes=sum(1 for n in nodes if any("nvidia.com/gpu" in (k or "") for k in n.status.allocatable or {}))
        pending=sum(1 for p in pods if p.status.phase=="Pending")
        # toy heuristic for "Radix vs current" value card
        now_gain = max(0, min(100, pending*2 + gpu_nodes*5))
        return {"gpu_nodes":gpu_nodes,"pending":pending,"estimated_improvement_now_pct":now_gain}
    except Exception as e:
        return {"error":str(e)}

@app.get("/v1/timeseries")
def timeseries():
    data=[]
    if TS_FILE.exists():
        with TS_FILE.open() as f:
            for line in f:
                try: data.append(json.loads(line))
                except: pass
    return {"data": data[-1000:]}  # last N
