"""Aggregate per-seed JSONs into a single consolidated.json per experiment.

Also emits CSV tables under ``results/tables/``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

EXPERIMENTS = ("E0", "E1", "E2", "E3", "E4", "E4t", "E5", "E6")
ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def _load_seeds(exp_id: str) -> list[dict]:
    base = ROOT / exp_id
    if not base.exists():
        return []
    out = []
    for d in sorted(base.glob("seed_*")):
        f = d / "results.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def _aggregate_methods(seed_payloads: list[dict]) -> list[dict]:
    """Aggregate per-method results across seeds.

    Handles two shapes:
      * E1/E2 — entries have ``rrmse_t_mean`` (one scalar per method).
      * E4 — entries have ``rrmse_t_per_alpha`` (one vector per method); averaged
        elementwise.
    """
    by_method: dict[str, list[dict]] = {}
    for sp in seed_payloads:
        for m in sp.get("methods", []):
            by_method.setdefault(m["method_id"], []).append(m)
    aggregated = []
    for mid, runs in by_method.items():
        if "rrmse_t_per_alpha" in runs[0]:
            usable = [r for r in runs if "rrmse_t_per_alpha" in r]
            if not usable:
                continue
            stacked = np.array([r["rrmse_t_per_alpha"] for r in usable], dtype=float)
            leak = np.array([r.get("leak_z_per_alpha", [0.0] * stacked.shape[1])
                             for r in usable], dtype=float)
            aggregated.append({
                "method_id": mid,
                "rrmse_t_per_alpha": stacked.mean(axis=0).tolist(),
                "rrmse_t_per_alpha_std": stacked.std(axis=0).tolist(),
                "leak_z_per_alpha": leak.mean(axis=0).tolist(),
                "leak_z_per_alpha_std": leak.std(axis=0).tolist(),
                "n_seeds": int(stacked.shape[0]),
            })
            continue
        rrmse = np.array([r.get("rrmse_t_mean", np.nan) for r in runs])
        rrmse = rrmse[np.isfinite(rrmse)]
        if rrmse.size == 0:
            continue
        if "rrmse_t_lo" in runs[0]:
            lo = np.nanmean([r["rrmse_t_lo"] for r in runs])
            hi = np.nanmean([r["rrmse_t_hi"] for r in runs])
        else:
            lo, hi = float(rrmse.min()), float(rrmse.max())
        aggregated.append({
            "method_id": mid,
            "rrmse_t_mean": float(np.nanmean(rrmse)),
            "rrmse_t_std": float(np.nanstd(rrmse)),
            "rrmse_t_lo": float(lo),
            "rrmse_t_hi": float(hi),
            "n_seeds": int(rrmse.size),
        })
    return aggregated


def consolidate(exp_id: str) -> Path | None:
    seeds = _load_seeds(exp_id)
    if not seeds:
        return None
    payload: dict = {"experiment": exp_id, "n_seeds": len(seeds)}
    # carry through any single-seed scalars and the wiener floor
    for key in ("wiener_floor_rrmse_t", "wiener_floor_rrmse_t_per_kind",
                "wiener_floor_per_snr_per_kind",
                "alphas", "kind", "I_dep", "null_samples",
                "per_subject", "per_subject_summary", "subjects",
                "clean_pool", "mi_observed", "mi_null", "by_stage", "families",
                "optuna_tuned_handcrafted", "optuna_n_trials",
                # E4t fields
                "n_templates", "per_template", "aggregate_slope",
                # E6 fields
                "rrmse_t_per_kind", "rrmse_t_mean", "channels",
                "epochs_configured", "epochs_used", "n_parameters",
                "n_train_segments", "n_test_segments"):
        if key in seeds[0]:
            payload[key] = seeds[0][key]
    if any("methods" in s for s in seeds):
        payload["methods"] = _aggregate_methods(seeds)
    out = ROOT / exp_id / "consolidated.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    return out


def write_table(exp_id: str) -> Path | None:
    cons = ROOT / exp_id / "consolidated.json"
    if not cons.exists():
        return None
    data = json.loads(cons.read_text())
    methods = data.get("methods")
    if not methods:
        return None
    floor = data.get("wiener_floor_rrmse_t", float("nan"))
    # E4 has per-alpha vectors instead of a scalar — emit a separate table.
    if methods and "rrmse_t_per_alpha" in methods[0]:
        out = TABLES / f"{exp_id}_per_alpha.csv"
        alphas = data.get("alphas", list(range(len(methods[0]["rrmse_t_per_alpha"]))))
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method_id", "metric"] + [f"alpha_{a}" for a in alphas] + ["n_seeds"])
            for m in methods:
                w.writerow([m["method_id"], "rrmse_t"]
                           + list(m["rrmse_t_per_alpha"]) + [m["n_seeds"]])
                w.writerow([m["method_id"], "leak_z"]
                           + list(m["leak_z_per_alpha"]) + [m["n_seeds"]])
        return out

    out = TABLES / f"{exp_id}_methods.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method_id", "rrmse_t_mean", "rrmse_t_std",
                    "rrmse_t_lo", "rrmse_t_hi", "excess_over_floor", "n_seeds"])
        for m in methods:
            excess = (m["rrmse_t_mean"] - floor) / floor if floor else float("nan")
            w.writerow([m["method_id"], m["rrmse_t_mean"], m["rrmse_t_std"],
                        m["rrmse_t_lo"], m["rrmse_t_hi"], excess, m["n_seeds"]])
    return out


def main() -> None:
    written = []
    for exp in EXPERIMENTS:
        p = consolidate(exp)
        if p:
            written.append(p)
            t = write_table(exp)
            if t:
                written.append(t)
    if not written:
        print("No experiment results found yet.")
    else:
        for p in written:
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
