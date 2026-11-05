"""Sample the shared test cases with each OT-FMPE checkpoint on a Kaggle GPU (same code as train_kaggle.sample_tests)."""
import json, sys, time
from pathlib import Path
src = next(p.parent for p in Path("/kaggle/input").rglob("otfmpe_model.py"))
sys.path.insert(0, str(src))
import otfmpe_model as T
out = Path("/kaggle/working")
for f in src.glob("fmpe_seed*.pt"):
    (out / f.name).write_bytes(f.read_bytes())
infos = []
for s in (0, 1, 2):
    t0 = time.time(); info = T.sample_tests(out, src, s); info["wall_seconds"] = time.time() - t0
    infos.append(info); print(json.dumps({k: v for k, v in info.items() if "box" not in k}), flush=True)
(out / "eval_info.json").write_text(json.dumps({"gpu": T.gpu_inventory(), "runs": infos}, indent=1))
