"""Aggregate W&B runs for a sweep into a per-experiment summary table.

Usage
-----
The deterministic path is to filter by Modal app id (every run launched via
``modal_app.py`` carries a ``modal-app:<id>`` tag once
``src/loggers/base.py`` is in place):

    uv run python scripts/summarize_run.py --modal-app ap-HyRuOdwPo8ZXJN6hhHgXpX

For older sweeps whose runs predate the tag, fall back to time/name filters:

    uv run python scripts/summarize_run.py --since 2026-05-26T19:00 --until 2026-05-27T05:00
    uv run python scripts/summarize_run.py --name-prefix loso__

Output
------
Two markdown tables (classification, regression) and a per-config breakdown
for any non-LOSO protocol (k-min sweeps etc.) — the LOSO subject axis is
collapsed across folds; every other nuisance axis (k_minutes, fold, runs…)
stays in the group key, so e.g. each ``k`` value gets its own row.

Group key per row: (protocol, backbone, adapter_stack, head, nuisance_minus_subject).
This is what splits ``head: convex_nn`` runs from the plain linear runs that
share the same backbone+adapter.
"""
from __future__ import annotations

import argparse
import collections
import re
import statistics as st
import sys
from typing import Any, Iterable

try:
    import wandb
except ImportError:
    print("wandb not installed — `uv sync` first", file=sys.stderr)
    sys.exit(1)


# job_type carries the per-subject axis. Strip the LOSO subject token so
# everything else (k-min, ktrials, lso run set, …) stays in the key.
_SUBJECT_TOKEN = re.compile(r"(?:^|_)loso-[SP]\d+|(?:^|_)S\d{3}$|(?:^|_)P\d+$")


def _strip_subject(job_type: str) -> str:
    out = _SUBJECT_TOKEN.sub("", job_type or "")
    return out.strip("_") or "loso"


def _build_filters(args: argparse.Namespace) -> dict[str, Any]:
    """Build a W&B Mongo-style filter from CLI args."""
    f: dict[str, Any] = {}
    if args.modal_app:
        f["tags"] = {"$in": [f"modal-app:{args.modal_app}"]}
    if args.name_prefix:
        f["display_name"] = {"$regex": f"^{re.escape(args.name_prefix)}"}
    created: dict[str, str] = {}
    if args.since:
        created["$gte"] = args.since
    if args.until:
        created["$lt"] = args.until
    if created:
        f["created_at"] = created
    if args.state:
        f["state"] = args.state
    return f


def _row_key(run: "wandb.apis.public.Run") -> tuple[str, ...]:
    """(protocol, backbone, adapter_stack, head, nuisance) — head from config."""
    name = run.name or ""
    parts = name.split("__")
    protocol = parts[0] if parts else "?"
    backbone = parts[1] if len(parts) > 1 else "?"
    adapter = parts[2] if len(parts) > 2 else "none"
    job_type = parts[3] if len(parts) > 3 else ""
    cfg = run.config or {}
    head_cfg = cfg.get("head") or {}
    head = head_cfg.get("name") if isinstance(head_cfg, dict) else None
    head = head or "linear"
    return (protocol, backbone, adapter, head, _strip_subject(job_type))


def _classify(run: "wandb.apis.public.Run") -> str | None:
    """Return 'classification' / 'regression' / None (skip)."""
    s = run.summary
    if "error" in s:
        return None
    if s.get("eval/accuracy") is not None:
        return "classification"
    if s.get("eval/pearson_r/mean") is not None or s.get("eval/r2/mean") is not None:
        return "regression"
    return None


def _agg(vals: list[float | None]) -> tuple[float, float, int]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return float("nan"), 0.0, 0
    m = st.mean(xs)
    s = st.stdev(xs) if len(xs) > 1 else 0.0
    return m, s, len(xs)


def _format_key(key: tuple[str, ...]) -> str:
    protocol, backbone, adapter, head, nuisance = key
    label = f"{backbone} / {adapter} / head={head}"
    if nuisance and nuisance != protocol:
        label += f"  [{nuisance}]"
    return label


def _print_table(rows: dict[tuple, list], task: str) -> None:
    """`rows`: group key -> list of per-run metric tuples."""
    if not rows:
        print(f"  (no {task} runs)")
        return
    if task == "classification":
        header = f"{'experiment':60} {'n':>4}  {'acc':>14}  {'kappa':>14}  {'auc':>14}"
        print(header)
        print("-" * len(header))
        sortable = []
        for k, vals in rows.items():
            accs = [v[0] for v in vals]
            kappas = [v[1] for v in vals]
            aucs = [v[2] for v in vals]
            am, as_, n = _agg(accs)
            km, ks_, _ = _agg(kappas)
            aum, aus_, _ = _agg(aucs)
            sortable.append((am, k, n, am, as_, km, ks_, aum, aus_))
        sortable.sort(key=lambda r: -r[0] if r[0] == r[0] else 0.0)
        for _, k, n, am, as_, km, ks_, aum, aus_ in sortable:
            print(f"{_format_key(k):60} {n:>4}  {am:.3f}±{as_:.3f}  {km:+.3f}±{ks_:.3f}  {aum:.3f}±{aus_:.3f}")
    else:  # regression
        header = f"{'experiment':60} {'n':>4}  {'pearson_r':>14}  {'r2':>14}"
        print(header)
        print("-" * len(header))
        sortable = []
        for k, vals in rows.items():
            prs = [v[0] for v in vals]
            r2s = [v[1] for v in vals]
            pm, ps_, n = _agg(prs)
            rm, rs_, _ = _agg(r2s)
            sortable.append((pm, k, n, pm, ps_, rm, rs_))
        sortable.sort(key=lambda r: -r[0] if r[0] == r[0] else 0.0)
        for _, k, n, pm, ps_, rm, rs_ in sortable:
            print(f"{_format_key(k):60} {n:>4}  {pm:+.3f}±{ps_:.3f}  {rm:+.3f}±{rs_:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="walimtan-stanford-university/etm", help="W&B project (entity/name).")
    p.add_argument("--modal-app", default="", help="Filter to runs tagged modal-app:<id>.")
    p.add_argument("--name-prefix", default="", help="Filter to runs whose name starts with this string.")
    p.add_argument("--since", default="", help="ISO8601 lower bound on created_at, e.g. 2026-05-26T19:00.")
    p.add_argument("--until", default="", help="ISO8601 upper bound on created_at.")
    p.add_argument("--state", default="finished", help="W&B state filter (default: finished).")
    p.add_argument("--per-page", type=int, default=1000)
    p.add_argument("--include-errored", action="store_true", help="Don't skip runs with an `error` summary key.")
    p.add_argument("--show-counts", action="store_true", help="Print crashed/errored counts at the end.")
    args = p.parse_args()

    if not any([args.modal_app, args.name_prefix, args.since, args.until]):
        print("error: provide at least one of --modal-app / --name-prefix / --since / --until", file=sys.stderr)
        sys.exit(2)

    filters = _build_filters(args)
    api = wandb.Api()
    # lazy=False forces the paginated GraphQL to include config+summaryMetrics
    # in-page. With lazy=True (default) each `run.summary`/`run.config` access
    # below triggers a fresh per-run round trip — the dominant wall-clock cost.
    # include_sweeps=False skips a Sweep.get per unique sweep_name we never use.
    runs = list(api.runs(
        args.project,
        filters=filters,
        per_page=args.per_page,
        include_sweeps=False,
        lazy=False,
    ))
    print(f"fetched {len(runs)} runs from {args.project} (filters={filters})\n")

    # protocol -> task -> row_key -> per-run metric tuples
    by_protocol: dict[str, dict[str, dict[tuple, list]]] = collections.defaultdict(
        lambda: {"classification": collections.defaultdict(list),
                 "regression": collections.defaultdict(list)}
    )
    crashed = collections.Counter()
    errored = collections.Counter()

    for r in runs:
        if r.state == "crashed":
            crashed[(r.name or "").split("__")[1:2] and (r.name or "").split("__")[1] or "?"] += 1
            continue
        s = r.summary
        if "error" in s:
            errored[(r.name or "").split("__")[1:2] and (r.name or "").split("__")[1] or "?"] += 1
            if not args.include_errored:
                continue
        task = _classify(r)
        if task is None:
            continue
        key = _row_key(r)
        protocol = key[0]
        if task == "classification":
            by_protocol[protocol]["classification"][key].append((
                s.get("eval/accuracy"),
                s.get("eval/cohen_kappa"),
                s.get("eval/auc_ovr"),
            ))
        else:
            by_protocol[protocol]["regression"][key].append((
                s.get("eval/pearson_r/mean"),
                s.get("eval/r2/mean"),
            ))

    for protocol in sorted(by_protocol):
        buckets = by_protocol[protocol]
        cls_rows = buckets["classification"]
        reg_rows = buckets["regression"]
        if cls_rows:
            print("=" * 100)
            print(f"[{protocol}] CLASSIFICATION  (LOSO subject collapsed; other axes kept in label)")
            print("=" * 100)
            _print_table(cls_rows, "classification")
            print()
        if reg_rows:
            print("=" * 100)
            print(f"[{protocol}] REGRESSION")
            print("=" * 100)
            _print_table(reg_rows, "regression")
            print()

    if args.show_counts:
        print()
        print(f"crashed: {dict(crashed)}")
        print(f"finished-with-error: {dict(errored)}")


if __name__ == "__main__":
    main()
