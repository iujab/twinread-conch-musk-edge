"""Write results/ncsa/<job>_<SLURM_JOB_ID>.json with timing and GPU info. Usage: job_record.py <job> <start_epoch> <exit_code> [note]."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

job, start, rc = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
note = sys.argv[4] if len(sys.argv) > 4 else ""
jid = os.environ.get("SLURM_JOB_ID", "local")
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30).stdout.strip()
except Exception as e:  # noqa: BLE001
    gpu = f"unavailable: {e}"
try:
    import torch
    tv, cv = torch.__version__, str(torch.version.cuda)
except Exception as e:  # noqa: BLE001
    tv, cv = f"unavailable: {e}", ""
try:
    import transformers
    trv = transformers.__version__
except Exception:  # noqa: BLE001
    trv = ""
out = Path(os.environ.get("TWINREAD_HOME", ".")) / "results" / "ncsa"
out.mkdir(parents=True, exist_ok=True)
rec = {"job": job, "slurm_job_id": jid, "node": os.environ.get("SLURMD_NODENAME", ""), "exit_code": rc,
       "elapsed_s": round(time.time() - start, 1), "gpu": gpu, "torch": tv, "cuda": cv, "transformers": trv,
       "note": note, "time_limit": os.environ.get("SBATCH_TIMELIMIT", "")}
(out / f"{job}_{jid}.json").write_text(json.dumps(rec, indent=2))
print("wrote", out / f"{job}_{jid}.json")
