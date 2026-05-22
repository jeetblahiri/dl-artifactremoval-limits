# PREREGISTRATION — Artifact-Limits

This file is the pre-registration record for the empirical work in *Within the Benchmark: Identifiability, Rate-Distortion, and the Structural Limits of EEGdenoiseNet.* The substantive hypotheses, sample sizes, decision rules, and stopping rules below are committed before any of the main experiments (E1–E5) are run.

## 1. Hypotheses

- **H1 (E1).** The empirical Wiener-floor RRMSE-T on EEGdenoiseNet, computed from training-pool second-order statistics, is within 5% of the best reported SOTA DL RRMSE-T on the official test split.
- **H2 (E2).** Within each (SNR, artifact-type) cell of EEGdenoiseNet, the gap between the best deep denoiser and the best single-channel handcrafted denoiser is bounded above by Cohen's *d* = 0.3 on per-segment RRMSE-T.
- **H3 (E3).** A single-architecture (SimpleCNN) capacity sweep from 1k to 10M parameters saturates within ±5% of D_W by 100k parameters.
- **H4 (E4).** RRMSE-T of DL denoisers trained on D1 (α = 0) is a monotone-increasing function of α on the dependent-mixing variant D1'(α), with steeper slope than handcrafted methods. Latent-leakage I(ŝ; z) is strictly positive and increasing in α for DL methods.
- **H5 (E5).** After regressing out the linear EOG/EMG → EEG coupling on Sleep-EDF SC4001, the residual mutual information I(EEG; EOG) and I(EEG; EMG) is significantly greater than the permutation null at p < 0.001.

## 2. Sample sizes

- Full EEGdenoiseNet test split, official train/val/test indices used as released.
- Five seeds per stochastic method: {42, 123, 2024, 7, 31337}.
- Handcrafted methods evaluated once per (method, hyperparameter, test split).
- E5: full available Sleep-EDF SC4001 recording, partitioned into 10-second windows.

## 3. Statistical decision rules

- Paired Wilcoxon signed-rank test at the segment level across each (method, method') pair.
- Effect size: Cliff's δ and median difference in RRMSE-T.
- Multiple-comparison correction: Benjamini-Hochberg FDR at q = 0.05 across all (method × SNR × artifact-type) cells.
- A method "exceeds the Wiener floor" if its per-segment RRMSE-T is < that of the Wiener filter with p < 0.001 after FDR.
- E4 monotone-trend test: Spearman ρ between α and RRMSE-T, slopes compared between method classes via permutation test (10,000 permutations).
- E5 dependence test: one-sample Wilcoxon of bias-corrected KSG MI against permutation null (1000 shuffles).

## 4. Stopping rules

- No additional seeds are added after results are inspected.
- No additional baselines are added after E1 is complete; the methods list is fixed in this document.

## 5. Hyperparameter search budget

- 20 Optuna trials per method (handcrafted *and* DL), tuned on the official validation split only.

## 6. Exclusion criteria

- Test segments that produce non-finite outputs for any single method are flagged and excluded from all method comparisons; the number of excluded segments is reported.
- E5: windows with > 10% sample drop-out (sensor disconnect) are excluded; criterion is fixed before analysis.

## 7. Reproducibility

- Frozen Python 3.10 environment via the committed `pyproject.toml` and `Dockerfile`.
- All seeds documented above.
- All figures regenerable from cached JSONs in `results/`.

OSF deposit timestamp: **[TO BE FILLED IN AT DEPOSIT]**
SHA-256 of this file at deposit time: **[TO BE FILLED IN AT DEPOSIT]**
