# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#quality">Quality</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

Repository includes implementation of CTC-based Conformer. Conformer model based on [paper](https://arxiv.org/pdf/2005.08100). CTC realization based on lectures from [DLA course](https://github.com/markovka17/dla/). Also pipeline includes LM inference for pretrained Conformer, LM implementation based on example from [PYPI lib](https://pypi.org/project/pyctcdecode/). Framework based on Hydra config system and logging in cometml or wandb.

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Quality

Below is the table for **CTC**-based submissions.

| Dataset           | CER  | WER  | CER BeamSearch | WER BeamSearch | CER LM | WER LM | BeamSize |
|-------------------|------|------|----------------|----------------|--------|--------|----------|
| libri-test-clear  | 0.07 | 0.22 | 0.06           | 0.21           | 0.04   | 0.11   | 50       |
| libri-test-other  | 0.17 | 0.43 | ---            | ---            | 0.13   | 0.28   | 100      |

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Model training

For model reproduction you need to:

1. 50 epochs with batch size 32, epoch lenght 1000 on train-clean-100 dataset.
2. 11 epochs with batch size 30, epoch lenght 1000 on train-other-500 dataset.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
