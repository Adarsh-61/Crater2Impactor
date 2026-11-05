"""Runtime of OT-FMPE and the Bayesian engine on the same CPU, same test cases.

Engine: infer_crater with its defaults (30,000 draws per scenario, 3 scaling packages, backend "auto"), one process.
OT-FMPE: 2,000 posterior samples, 20 RK4 steps, one case at a time, with 1 and with all CPU threads.
GPU timings of OT-FMPE come from the Kaggle run (run_info.json) and are reported separately.
Usage: python research/otfmpe/timing.py RUN_DIR [N_CASES]
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import cases as C  # noqa: E402
from train_kaggle import CFG, CorrectedOTFMPE  # noqa: E402


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main(run, n=20):
    run = Path(run)
    cs = json.loads((run / "cases.json").read_text())
    pick = [c for c in cs if c["set"] in ("nat_D", "nat_DVA")][::max(1, len(cs) // (4 * n))][:n]
    x = torch.tensor(np.array([C.context(c) for c in pick], dtype=np.float32))
    ck = torch.load(run / "fmpe_seed0.pt", map_location="cpu")
    model = CorrectedOTFMPE(hidden_dim=CFG["hidden_dim"], n_blocks=CFG["n_blocks"], time_embed_dim=CFG["time_embed_dim"])
    model.load_state_dict(ck["state_dict"]); model.eval()
    out = {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "n_cases": len(pick), "torch": torch.__version__}
    for threads in (1, os.cpu_count()):
        torch.set_num_threads(threads)
        model.sample(x[:1], CFG["n_test_samples"], CFG["rk4_steps"])
        ts = []
        for i in range(len(pick)):
            t0 = time.perf_counter(); model.sample(x[i:i + 1], CFG["n_test_samples"], CFG["rk4_steps"])
            ts.append(time.perf_counter() - t0)
        out[f"otfmpe_cpu_{threads}_threads_ms_median"] = 1000 * float(np.median(ts))
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    C.run_engine(pick[0])
    ts = [C.run_engine(c)["seconds"] for c in pick]
    out["engine_cpu_1_process_ms_median"] = 1000 * float(np.median(ts))
    out["settings"] = {"engine": "infer_crater defaults: 30,000 draws x 3 packages, backend auto",
                       "otfmpe": f"{CFG['n_test_samples']} samples, {CFG['rk4_steps']} RK4 steps, one case per call"}
    (run / "timing_cpu.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
