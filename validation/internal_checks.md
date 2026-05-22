# Internal validity checks

Before any result in the paper is claimed, the following CI-enforced checks must pass:

| Check | Expected outcome | Implementation |
|---|---|---|
| EEGdenoiseNet reproduction (E1) within ±5% of published values | All DL methods | `validation/ci_checks.py::check_e1_reproduction` |
| Wiener filter empirical RRMSE matches closed-form D_W to within 1% (Gaussian synthetic) | yes | `tests/test_wiener_floor.py::test_wiener_floor_closed_form_matches_empirical` |
| `I_dep` permutation null centred at small positive bias | yes | `tests/test_mi_estimators.py::test_mi_is_zero_for_independent_samples` |
| H0 (identity) is the worst on RRMSE-T at every SNR | yes | `validation/ci_checks.py::check_identity_worst` |
| H_W ≤ every other handcrafted method on RRMSE-T at high SNR | yes in expectation | `validation/ci_checks.py::check_wiener_dominates_handcrafted` |
| DL training loss converges (no run silently diverges) | yes | early-stop telemetry in `methods.deep.base_denoiser` |
| Test-set indices identical across all (method × seed) runs | yes | committed split JSON; loader checks file hash |
| D1'(α=0) reproduces D1 to numerical precision | yes | `tests/test_D1prime_construction.py::test_alpha_zero_is_identity` |
