"""Run train_kaggle.py on Kaggle (private dataset + private GPU script kernel) and fetch its outputs.

Credentials: KAGGLE_CONFIG_DIR must point at the directory holding kaggle.json (the project root copy is the
one that authenticates). Nothing from the credential is printed; the
account name is replaced by <kaggle-user> in every file this script saves.
Usage: python kaggle_run.py DATA_DIR WORK_DIR {dataset|kernel|wait|fetch}
"""
import json
import shutil
import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DS_SLUG, K_SLUG = "c2i-otfmpe-v21-data", "c2i-ot-fmpe-v21"   # Kaggle derives the kernel slug from the title
HERE = Path(__file__).parent


def redact(path, user):
    for f in Path(path).rglob("*"):
        if f.is_file() and f.suffix in (".json", ".log", ".txt", ".py", ".ipynb"):
            s = f.read_text(errors="ignore")
            if user in s:
                f.write_text(s.replace(user, "<kaggle-user>"))


def main(data_dir, work, step):
    api = KaggleApi(); api.authenticate(); user = api.get_config_value("username")
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    if step == "dataset":
        d = work / "dataset"; d.mkdir(exist_ok=True)
        for f in ("train.npz", "val.npz", "tests.npz", "gen_manifest.json"):
            shutil.copy(Path(data_dir) / f, d / f)
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": "C2I OT-FMPE v21 data", "id": f"{user}/{DS_SLUG}", "licenses": [{"name": "CC0-1.0"}]}))
        api.dataset_create_new(str(d), public=False, quiet=False, dir_mode="zip")
        print("dataset created (private)")
    elif step == "kernel":
        k = work / "kernel"; k.mkdir(exist_ok=True)
        shutil.copy(HERE / "train_kaggle.py", k / "train_kaggle.py")
        (k / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{K_SLUG}", "title": "C2I OT-FMPE v21", "code_file": "train_kaggle.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": False, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{user}/{DS_SLUG}"],
            "competition_sources": [], "kernel_sources": []}))
        r = api.kernels_push(str(k))
        print("kernel pushed:", getattr(r, "error", None) or "ok", "version", getattr(r, "version_number", "?"))
    elif step == "wait":
        while True:
            st = api.kernels_status(f"{user}/{K_SLUG}")
            s = str(getattr(st, "status", st))
            print(time.strftime("%H:%M:%S"), s, getattr(st, "failure_message", "") or "", flush=True)
            if any(w in s for w in ("COMPLETE", "ERROR", "CANCEL")):
                break
            time.sleep(60)
    elif step == "fetch":
        out = work / "output"; out.mkdir(exist_ok=True)
        api.kernels_output(f"{user}/{K_SLUG}", path=str(out), force=True)
        redact(out, user)
        print(sorted(p.name for p in out.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main(*sys.argv[1:4])
