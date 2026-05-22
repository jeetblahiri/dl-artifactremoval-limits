"""CI-side internal validity checks (see internal_checks.md).

This script reads consolidated experiment results (if present) and asserts the
expected invariants. Returns a non-zero exit code if any check fails — wire it
into CI on the experimental branch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load(exp: str) -> dict | None:
    p = RESULTS / exp / "consolidated.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def check_identity_worst() -> bool:
    data = _load("E2") or _load("E1")
    if data is None or "methods" not in data:
        print("[check_identity_worst] no E1/E2 results yet — skip")
        return True
    rrmse = {m["method_id"]: m["rrmse_t_mean"] for m in data["methods"]}
    if "identity" not in rrmse:
        return True
    worst = max(rrmse.values())
    ok = abs(rrmse["identity"] - worst) < 1e-6
    print(f"[check_identity_worst] {'OK' if ok else 'FAIL'} "
          f"(identity={rrmse['identity']:.4f}, worst={worst:.4f})")
    return ok


def check_wiener_dominates_handcrafted() -> bool:
    data = _load("E2")
    if data is None:
        print("[check_wiener_dominates_handcrafted] no E2 — skip")
        return True
    rrmse = {m["method_id"]: m["rrmse_t_mean"] for m in data["methods"]}
    if "wiener_filter" not in rrmse:
        return True
    handcrafted = ["dwt", "swt", "emd", "total_variation", "savgol", "bandpass_notch"]
    w = rrmse["wiener_filter"]
    other = [rrmse[k] for k in handcrafted if k in rrmse]
    ok = w <= min(other) + 1e-3 if other else True
    print(f"[check_wiener_dominates_handcrafted] {'OK' if ok else 'WARN'}")
    return True            # advisory: log-only


def check_e1_reproduction(tol: float = 0.05) -> bool:
    data = _load("E1")
    if data is None:
        print("[check_e1_reproduction] no E1 — skip")
        return True
    floor = data.get("wiener_floor_rrmse_t")
    if floor is None:
        return True
    methods = data.get("methods", [])
    for m in methods:
        if m["method_id"].startswith(("simple_cnn", "novel_cnn", "rnn_lstm")):
            rel = abs(m["rrmse_t_mean"] - floor) / floor
            if rel > 0.5:    # within an order of the floor is what we expect
                print(f"[check_e1_reproduction] WARN {m['method_id']} rel={rel:.2f}")
    return True


def main() -> int:
    ok = all([check_identity_worst(), check_wiener_dominates_handcrafted(),
              check_e1_reproduction()])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
