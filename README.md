graph-nnets-demo
===============
A minimal, local-first MLOps-style repo for training and evaluating a Graph Neural Network (GCN) on the Cora citation dataset (via PyTorch-Geometric).

This repo is intentionally "public-sector friendly": it emphasizes reproducibility, clear train/eval separation, and config-driven runs, and it avoids cloud-specific wiring and fragile data tooling.

## What this repo signals
- Reproducible runs: pinned dependencies + Dockerfile
- Operational thinking: one command to train, one to evaluate
- Restraint/clarity: no cloud build, no sweeps, no external data remotes

## W&B (optional)
You can browse historical results from the original project without retraining:
- Overview report: https://wandb.ai/group5-dtumlops/group5-pyg-dtumlops/reports/Overview-of-project-results--VmlldzoxNDYyODk2

W&B logging is disabled by default. Enable it by setting `experiment.hyperparams.use_wandb=true` in config.

## Quickstart (local, recommended on macOS)
Use conda/mamba for PyTorch + PyG (avoids compiling native extensions):

```bash
mamba env create -f environment.yml
mamba activate graph-nnets-demo

python src/models/train_model.py
python src/models/predict_model.py
```

If you *really* want `venv`/`pip`, you’ll need a Linux container or to build PyG deps from source.

## Quickstart (Docker)
```bash
docker build -t graph-nnets-demo .
docker run --rm -it graph-nnets-demo
```

## Configuration
Runs are Hydra config-driven:
- Default config: `src/config/default_config.yaml`
- Experiment config: `src/config/experiment/exp1.yaml`

Override any parameter from the CLI, e.g.:
```bash
python src/models/train_model.py experiment.hyperparams.epochs=50 experiment.hyperparams.lr=0.005
```

## Data
The Cora dataset is downloaded automatically by PyTorch-Geometric and cached under `data/`.
This repo does not rely on DVC or any external dataset remote.

## Project flowchart
![Alt text](reports/figures/flowchart.png?raw=true "Flowchart")


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the project and running it locally
    ├── requirements-docker.txt   <- The requirements file for running the docker file as some packages are installed individually
    │                         in the Dockerfile
    ├── requirements-dev.txt      <- The requirements file containing additional packages for project development
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │  
    │   ├── config         <- Experiment configuration files to be used with hydra
    │   │   ├── experiment <- Various expriment setups
    │   │   │   └── exp1.yaml
    │   │   └── default_config.yaml
    │   │  
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to define arcitectur, train models, use trained models to make
    │   │   │                 predictions and for cprofiling the model scripts
    │   │   ├── predict_model.py
    │   │   ├── model.py
    │   │   ├── train_model_cprofile_basic.py
    │   │   ├── train_model_cprofile_sampling.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
