# Limits of Deep Learning for EEG Artifact Removal

This repository contains the implementation for the experiments in
**Limits of Deep Learning for EEG Artifact Removal**.

It is the code-only companion repository for the manuscript. The paper source
is intentionally not included here; this repository contains the Python package,
experiment entrypoints, cached result summaries, tests, and reproduction
scripts.

## What This Code Does

The project studies what EEGdenoiseNet-style synthetic artifact-removal
benchmarks can and cannot certify. The implementation covers:

- Linear-MMSE and Wiener-reference calculations from benchmark second-order
  statistics.
- Reproduction and comparison of handcrafted and deep EEG artifact-removal
  methods.
- A dependent-mixing perturbation that changes the joint signal-artifact
  structure while preserving benchmark-style marginals.
- Template-robustness checks for the D1' perturbation.
- A Sleep-EDF in-vivo residual-dependence audit using block bootstrap and
  circular-shift nulls appropriate for autocorrelated time series.
- A wide IC-U-Net oracle-style non-linear MMSE comparator.
- Cached JSON/CSV outputs used to regenerate the reported tables and figures.

Raw EEG data and trained model checkpoints are not committed. The data-loading
and reproduction scripts expect local dataset caches or public upstream
downloads where available.

## Repository Layout

```text
.
  src/artifact_limits/          Python package
    data/                       Dataset loaders and D1/D1' construction
    evaluation/                 Metrics, statistics, and block inference
    methods/                    Handcrafted and deep denoisers
    theory/                     Identifiability, Wiener, MI, oracle-MMSE code
    viz/                        Figure-generation scripts
  experiments/                  E0-E6 experiment entrypoints and configs
  results/                      Cached consolidated JSON/CSV outputs
  tests/                        Unit tests for core theory/method utilities
  validation/                   Internal consistency checks
  Makefile                      Reproduction targets
  run_full_protocol.sh          Idempotent full-protocol runner
  pyproject.toml                Package metadata and dependencies
  Dockerfile                    Container recipe
  PREREGISTRATION.md            Pre-registered hypotheses and protocol notes
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m pip install -e ".[dev,viz]"
```

For CPU-only inspection, tests, and result consolidation this is sufficient.
GPU acceleration is recommended for rerunning the deep-learning experiments.

## Quick Checks

Run the test suite:

```bash
PYTHONPATH=src pytest -q
```

Regenerate cached result summaries:

```bash
python results/consolidate.py
```

The test suite exercises the core metric, inference, Wiener-reference,
dependent-mixing, and method-registry paths. It is intentionally lightweight;
full deep-model training is handled by the experiment scripts rather than by
unit tests.

## Experiments

The main experiment entrypoints are:

- `E0`: EEGdenoiseNet pool audit.
- `E1`: Wiener reference and EEGdenoiseNet-original deep-method reproduction.
- `E2`: full handcrafted/deep methods comparison.
- `E3`: SimpleCNN capacity-scaling curve.
- `E4`: dependent-mixing perturbation.
- `E4t`: D1' template-robustness sweep across 16 template families.
- `E5`: Sleep-EDF residual-dependence audit with 30-second block bootstrap and
  circular-shift nulls.
- `E6`: wide IC-U-Net non-linear MMSE comparator.

Each experiment has an entrypoint under `experiments/` and configuration under
`experiments/config/`.

## Reproduction Targets

Run the full protocol:

```bash
./run_full_protocol.sh
```

Or run individual targets:

```bash
make data
make reproduce-e1
make reproduce-e2
make reproduce-e3
make reproduce-e4
make reproduce-e4t
make reproduce-e5
make reproduce-e6
make tables
make figures
```

The full protocol is idempotent where possible: existing result files are
skipped unless the corresponding script or target is run with a clean results
directory.

## Cached Results

The `results/` directory contains lightweight cached outputs:

- `results/E*/seed_*/results.json`: per-seed experiment outputs.
- `results/E*/consolidated.json`: consolidated summaries.
- `results/tables/*.csv`: table-ready summaries.

Large raw arrays, trained checkpoints, logs, and figure build products are
excluded from version control. This keeps the repository small while preserving
the numerical outputs needed to audit the reported claims.

## Implementation Notes

Key implementation files:

- `src/artifact_limits/theory/wiener_floor.py`: closed-form linear-MMSE
  reference calculations.
- `src/artifact_limits/methods/handcrafted/wiener_snr_aware.py`: known-SNR and
  SNR-marginal Wiener variants.
- `src/artifact_limits/theory/oracle_mmse.py`: oracle-style MMSE comparator
  helpers.
- `src/artifact_limits/evaluation/block_inference.py`: moving-block bootstrap
  and circular-shift null inference.
- `src/artifact_limits/data/d1prime_templates.py`: D1' template families.
- `experiments/e4_template_robustness.py`: template-robustness experiment.
- `experiments/e6_oracle_mmse.py`: wide IC-U-Net comparator experiment.
- `results/consolidate.py`: consolidation of E0-E6 outputs.

## Data

Raw datasets are not included in the repository. The code is written to work
with locally cached copies of EEGdenoiseNet and Sleep-EDF. See the data-loading
utilities in `src/artifact_limits/data/` and the `make data` target for the
expected paths and download/cache behavior.

## License

The code is released under the license specified in `pyproject.toml`.
