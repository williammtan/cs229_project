"""Run etm classification benchmarks on Modal (https://modal.com).

The existing scripts are reused unchanged; this file is the only addition.

One-time setup:
    uv run modal token new --profile etm-clf       # see .envrc; pinned per-repo
    uv run modal secret create wandb WANDB_API_KEY=<key>
    uv run modal secret create huggingface HF_TOKEN=<token>   # for REVE only

Common commands:
    # populate the data volume (all 104 EEGMMI subjects minus exclusions)
    uv run modal run modal_app.py::download_data

    # full sweep, FANNED OUT in parallel (one A100 per experiment).
    # Skips the 3 FM-finetune configs (each is 2-5h on its own).
    uv run modal run --detach modal_app.py::run_benchmarks \\
        --skip "cbramod_finetune_loso labram_finetune_loso reve_finetune_loso"

    # K-trials calibration sweep (separate research question, same fanout pattern)
    uv run modal run --detach modal_app.py::run_kmin_sweep

    # ... or one big serial container (cheaper, slower)
    uv run modal run --detach modal_app.py::run_benchmarks_serial

    # run only one experiment
    uv run modal run modal_app.py::run_experiment --name baseline_eegnet_loso

    # pull results back
    uv run modal volume get etm-results / ./modal-results
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "etm-benchmarks"
REPO = Path(__file__).parent
REMOTE_REPO = "/root/etm"
DEFAULT_GPU = "L40S"
MAX_CONCURRENT_GPUS = 10  # Modal account GPU concurrency limit.

# Reuse the experiment list from the existing runner script — single source of truth.
# Locally REPO is the project dir; inside a Modal container, modal_app.py lives at
# /root/modal_app.py while the repo is copied to REMOTE_REPO by add_local_dir.
_runner_script = REPO / "scripts" / "run_full_baselines.py"
if not _runner_script.exists():
    _runner_script = Path(REMOTE_REPO) / "scripts" / "run_full_baselines.py"
_spec = importlib.util.spec_from_file_location("_run_full_baselines", _runner_script)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_run_full_baselines"] = _mod
_spec.loader.exec_module(_mod)
EXPERIMENTS: list[str] = list(_mod.EXPERIMENTS)
KMIN_EXPERIMENTS: list[str] = list(_mod.KMIN_EXPERIMENTS)

# torch wheels on linux PyPI bundle CUDA, so plain pip_install_from_pyproject
# gives a GPU-capable build when the function requests a gpu.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_pyproject(str(REPO / "pyproject.toml"))
    # pyproject pins CPU-only jax/jaxlib so Mac dev installs work. On Modal we
    # have an A100, so swap in the CUDA 12 jaxlib + plugin to actually use it.
    .pip_install("jax[cuda12]>=0.7.0")
    .add_local_dir(
        str(REPO),
        remote_path=REMOTE_REPO,
        ignore=[
            "wandb/*",
            "results/*",
            "data/*",
            "outputs/*",
            ".venv/*",
            ".git/*",
            "**/__pycache__/*",
            "*.pyc",
        ],
    )
)

data_vol = modal.Volume.from_name("etm-data", create_if_missing=True)
results_vol = modal.Volume.from_name("etm-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("etm-hf-cache", create_if_missing=True)

VOLUMES = {
    f"{REMOTE_REPO}/data": data_vol,
    f"{REMOTE_REPO}/results": results_vol,
    "/root/.cache/huggingface": hf_cache_vol,
}

SECRETS = [
    modal.Secret.from_name("wandb"),
    modal.Secret.from_name("huggingface"),
]

app = modal.App(APP_NAME, image=image)


def _sh(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REMOTE_REPO, check=True)


@app.function(volumes=VOLUMES, timeout=2 * 60 * 60)
def download_data(subjects: str = "", jobs: int = 6) -> None:
    """Download EEGMMI subjects into the persistent data volume.

    Empty ``subjects`` means "all 104 (109 minus exclusions)" — pass space-
    separated subject IDs to override. Parallelism is at the subject level.
    """
    cmd = ["python", "scripts/download_data.py", "-j", str(jobs)]
    if subjects:
        cmd += subjects.split()
    else:
        cmd += ["--all"]
    _sh(cmd)
    data_vol.commit()


@app.function(
    gpu=DEFAULT_GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=6 * 60 * 60,
    max_containers=MAX_CONCURRENT_GPUS,
)
def run_experiment(name: str, overrides: str = "", logger: str = "wandb", modal_app_id: str = "") -> str:
    """Run a single Hydra experiment by name. Returns the experiment name.

    ``modal_app_id`` is forwarded by the fanout so each W&B run carries a
    ``modal-app:<id>`` tag and can be looked up by `scripts/summarize_run.py`.
    """
    if modal_app_id:
        os.environ["MODAL_APP_ID"] = modal_app_id
    cmd = ["python", "-m", "src.runner", f"+experiment={name}", f"logger={logger}"]
    if overrides:
        cmd += overrides.split()
    _sh(cmd)
    results_vol.commit()
    return name


def _current_app_id() -> str:
    """Best-effort capture of the running Modal app ID from inside a function.

    For ephemeral `modal run` invocations, the app handle attached at module
    scope is hydrated once the function starts, so its ``app_id`` is available.
    Falls back to the env var Modal sets in the container, then to "".
    """
    try:
        aid = getattr(app, "app_id", None)
        if aid:
            return aid
    except Exception:
        pass
    return os.environ.get("MODAL_APP_ID", "")


def _fanout(label: str, names: list[str], logger: str) -> None:
    """Shared fanout: run_experiment.map across the given experiment names."""
    app_id = _current_app_id()
    print(f"[{label}] launching {len(names)} experiments on {DEFAULT_GPU}"
          f" (modal_app_id={app_id or 'unknown'}):")
    for n in names:
        print(f"  - {n}")

    done: list[str] = []
    failed: list[str] = []
    for result in run_experiment.map(
        names,
        kwargs={"logger": logger, "modal_app_id": app_id},
        return_exceptions=True,
        order_outputs=False,
    ):
        if isinstance(result, Exception):
            print(f"[{label}]  FAILED: {result!r}")
            failed.append(repr(result))
        else:
            print(f"[{label}]  done: {result}")
            done.append(result)

    print()
    print(f"[{label}] finished: {len(done)} ok, {len(failed)} failed")
    if failed:
        for f in failed:
            print(f"  - {f}")
        raise RuntimeError(f"{len(failed)} experiments failed")


@app.function(
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=12 * 60 * 60,
)
def run_benchmarks(
    logger: str = "wandb",
    only: str = "",
    skip: str = "",
) -> None:
    """LOSO benchmark grid (incl. RA variants), fanned out one GPU per experiment.

    `only`/`skip` are space-separated experiment names, mirroring the
    --only/--skip flags on scripts/run_full_baselines.py.
    """
    names = only.split() if only else list(EXPERIMENTS)
    skip_set = set(skip.split())
    names = [n for n in names if n not in skip_set]
    _fanout("loso", names, logger)


@app.function(
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=12 * 60 * 60,
)
def run_kmin_sweep(
    logger: str = "wandb",
    only: str = "",
    skip: str = "",
) -> None:
    """k-min calibration-budget sweep, fanned out one GPU per experiment.

    Separate entry point from run_benchmarks because the kmin protocol answers a
    different research question (calibration efficiency curve, not zero-shot LOSO).
    """
    names = only.split() if only else list(KMIN_EXPERIMENTS)
    skip_set = set(skip.split())
    names = [n for n in names if n not in skip_set]
    _fanout("kmin", names, logger)


@app.function(
    gpu=DEFAULT_GPU,
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=24 * 60 * 60,
)
def run_benchmarks_serial(logger: str = "wandb", extra: str = "") -> None:
    """Single-container serial fallback: wraps scripts/run_full_baselines.py.

    Cheaper (one GPU, not N) but proportionally slower wall-clock.
    `extra` forwards args to the script, e.g. extra="--only baseline_eegnet_loso".
    """
    cmd = ["python", "scripts/run_full_baselines.py", "--logger", logger]
    if extra:
        cmd += extra.split()
    _sh(cmd)
    results_vol.commit()


@app.local_entrypoint()
def main() -> None:
    """Default `modal run modal_app.py` → fan out the full sweep on A100s."""
    run_benchmarks.remote()
