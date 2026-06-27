# ExplainableAI Project

This repository was originally created by the authors of the paper 'Self explaining neural networks: A review with extensions'. The original repo is https://github.com/AmanDaVinci/SENN.
We used it as a starting point to make more tests on SENN, run LIME and IG and compare the explanations given by the different methods.

This repository contains the code used to train a Self-Explaining Neural Network (SENN) on FashionMNIST and to compare two post-hoc attribution methods: LIME and Integrated Gradients. The project is organized around a small set of root scripts, with each script handling one stage of the workflow.

## Content of The Repo Root

The files at the repository root are the main entry points you are expected to run directly.

- [main.py](main.py) starts a training run from a JSON config file. It parses `--config`, builds the trainer, runs the training loop, and finalizes the experiment.
- [run_lime.py](run_lime.py) loads a trained checkpoint and computes LIME explanations for the test set. It also measures how much the model confidence drops when the most relevant pixels are masked.
- [run_integrated_gradients.py](run_integrated_gradients.py) does the same kind of post-hoc analysis with Integrated Gradients instead of LIME.
- [compute_relative_metrics_lime.py](compute_relative_metrics_lime.py) reads the saved LIME attributions and recomputes the ablation-based comparison against a random baseline.
- [compute_relative_metrics_ig.py](compute_relative_metrics_ig.py) performs the same relative-metric computation for Integrated Gradients outputs.
- [environment.yml](environment.yml) defines the conda environment used by the project.

## Config Presets

The [configs](configs) folder contains the JSON experiment presets used by the scripts in the repo root. Each file defines one FashionMNIST run, combining a regularization setting and model width choice with the fixed seed used for that experiment.

These files are what you pass to [main.py](main.py), [run_lime.py](run_lime.py), [run_integrated_gradients.py](run_integrated_gradients.py), and the metric scripts.

## Saved Results

The [results](results) folder stores the outputs produced by the training and explanation scripts. Each experiment gets its own subfolder, named after the config experiment name, so runs do not overwrite each other.

Each experiment folder typically contains:

- `checkpoints/`: model checkpoints, including the best model saved during training.
- `logs/`: TensorBoard logs and training progress output.
- `accuracies_losses_train.csv`, `accuracies_losses_valid.csv`, and sometimes `accuracies_losses_test.csv`: per-epoch or per-evaluation metrics.
- `posthoc/`: LIME and Integrated Gradients attribution tensors, ablation-drop arrays, and metadata files.
- `posthoc_old/`: older attribution outputs kept for comparison in some runs.


## The `senn` folder

The [senn](senn) folder contains:

- [senn/trainer.py](senn/trainer.py) defines the trainer, handles data loading, builds the SENN model from the config, runs training and validation, and saves checkpoints and logs.
- [senn/models](senn/models) contains the SENN model components.
- [senn/datasets](senn/datasets) contains the dataset loaders. The FashionMNIST loader applies the normalization used throughout the repo and returns train, validation, and test loaders.
- [senn/utils](senn/utils) contains plotting and helper utilities used by the trainer and notebooks.


## Notebooks

The notebooks in the root folder are reporting and analysis notebooks.

- [fashion_mnist_posthoc_1e-2_c5.ipynb](fashion_mnist_posthoc_1e-2_c5.ipynb) is a detailed post-hoc analysis notebook for the `lambda=1e-2, c=5` run. It contains multiple sections with plots and tables for inspecting the explanations and their effect on model confidence.
- [fashion_mnist_report_1.ipynb](fashion_mnist_report_1.ipynb) is a report notebook that compares experiment outputs and visualizes representative samples, explanations, and summary metrics.
- [fashion_mnist_report_1e-1.ipynb](fashion_mnist_report_1e-1.ipynb) is the same style of report notebook for the `1e-1` setting.
- [fashion_mnist_report_1e-2.ipynb](fashion_mnist_report_1e-2.ipynb) is the report notebook for the `1e-2` setting and contains the most complete executed analysis among the report notebooks.
- [fashion_mnist_report_1e-3.ipynb](fashion_mnist_report_1e-3.ipynb) is the report notebook for the `1e-3` setting.
- [fashion_mnist_report_1e-4.ipynb](fashion_mnist_report_1e-4.ipynb) is the report notebook for the `1e-4` setting.


## Setup

The project is intended for a conda environment with Python 3.12 and the usual scientific Python stack, plus PyTorch and Captum.

```bash
conda env create -f environment.yml
conda activate senn
```


## Running The Scripts

### Train a model

```bash
python main.py --config path/to/config.json
```

This is the training entry point. The config file controls the experiment name, model settings, and dataset parameters. One detail to watch is that the default config path in the script points to a COMPAS example, so for the FashionMNIST experiments in this repo you should pass one of the FashionMNIST configs explicitly.

### Run LIME explanations

```bash
python run_lime.py --config path/to/config.json
```

Useful optional arguments:

- `--n_samples`: number of perturbation samples per image used by LIME.
- `--max_images`: cap the number of test images processed.
- `--device`: force `cpu` or `cuda:0`.

The script writes outputs under `results/<exp_name>/posthoc/`, including the attribution tensor, predictions, labels, confidence-drop array, and a metadata JSON file.

### Run Integrated Gradients

```bash
python run_integrated_gradients.py --config path/to/config.json
```

Useful optional arguments:

- `--n_steps`: number of integration steps used to approximate the path integral.
- `--max_images`: cap the number of test images processed.
- `--device`: force `cpu` or `cuda:0`.

### Compute relative metrics

```bash
python compute_relative_metrics_lime.py --config path/to/config.json
python compute_relative_metrics_ig.py --config path/to/config.json
```

These scripts compare top-attribution ablation against a random-pixel baseline. The resulting arrays are saved back into the matching `results/<exp_name>/posthoc/` directory.

