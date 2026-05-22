# Code for "Limits of Deep Learning for EEG Artifact Removal"

This directory contains the implementation of the experiments in the paper.

Install:

```bash
python -m pip install -e ".[dev,viz]"
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

Regenerate cached result summaries:

```bash
python results/consolidate.py
```

Run the full protocol:

```bash
./run_full_protocol.sh
```

Raw datasets are excluded from the repository. The data-loading scripts either
use locally cached upstream files or fetch the public releases when available.
