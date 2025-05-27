This repository provides a modularized Python library and a set of executable scripts for time series anomaly detection, building upon techniques

**Original GitLab Repository:** https://gitlab.com/basu1999/tsfm-anomaly-paper

## Table of Contents

1.  [Features](#features)
2.  [Project Structure](#project-structure)
3.  [Installation](#installation)
    *   [Prerequisites](#prerequisites)
    *   [Cloning the Repository](#cloning-the-repository)
    *   [Docker Setup (Recommended)](#docker-setup-recommended)
    *   [Environment Setup (Inside Docker)](#environment-setup-inside-docker)
    *   [Dataset Setup](#dataset-setup)
    *   [Fold ID Setup](#fold-id-setup)
4.  [Usage: Running Scripts](#usage-running-scripts)
    *   [General Notes on Running Scripts](#general-notes-on-running-scripts)
    *   [Statistical & Tree-Based Methods Evaluation](#statistical--tree-based-methods-evaluation)
        *   [Run IQR Evaluation](#run-iqr-evaluation)
        *   [Run Modified Z-score Evaluation](#run-modified-z-score-evaluation)
        *   [Run Isolation Forest Evaluation](#run-isolation-forest-evaluation)
        *   [Run Local Outlier Factor (LOF) Evaluation](#run-local-outlier-factor-lof-evaluation)
    *   [Deep Learning Model Hyperparameter Optimization](#deep-learning-model-hyperparameter-optimization)
        *   [Run VAE Hyperparameter Optimization](#run-vae-hyperparameter-optimization)
        *   [Run MOMENT Fine-tuning Hyperparameter Optimization](#run-moment-fine-tuning-hyperparameter-optimization)
    *   [Training/Fine-tuning Final Deep Learning Models](#trainingfine-tuning-final-deep-learning-models)
        *   [Train Final VAE Model](#train-final-vae-model)
        *   [Fine-tune Final MOMENT Model](#fine-tune-final-moment-model)
5.  [Using the `tsfm_ad_lib` Library Programmatically](#using-the-tsfm_ad_lib-library-programmatically) (To be added)
6.  [Running Example Notebooks](#running-example-notebooks) (To be added)
7.  [Running Tests](#running-tests) (To be added)
8.  [Contributing](#contributing)
9.  [License](#license)
10. [Contact](#contact)

## Features

*   Modular and reusable Python library (`tsfm_ad_lib`) for time series anomaly detection tasks.
*   Implementation of various anomaly detection algorithms:
    *   Variational Autoencoder (VAE)
    *   Fine-tuned MOMENT models
    *   Isolation Forest
    *   Local Outlier Factor (LOF)
    *   Interquartile Range (IQR)
    *   Modified Z-score
*   Scripts for end-to-end workflows:
    *   Hyperparameter optimization using Optuna for VAE and MOMENT models.
    *   Evaluation of statistical and tree-based methods with threshold/parameter tuning.
    *   Training final deep learning models with optimized hyperparameters.
*   Configurable data paths, model parameters, and logging.
*   Dockerized environment for reproducibility.

## Project Structure 

tsfm-anomaly-paper/
├── tsfm_ad_lib/ # Core Python library
│ ├── init.py
│ ├── config.py # Default configurations
│ ├── data_loader.py # Data loading utilities and TimeDataset
│ ├── preprocessing.py # Preprocessing functions
│ ├── models/ # Model definitions
│ │ ├── vae.py, moment.py, tree_based.py, ..
│ ├── training.py # Training loop functions
│ ├── evaluation.py # Evaluation functions
│ └── utils.py # Utility functions
├── scripts/ # Executable scripts using the library
│ ├── run_iqr_eval.py
│ ├── run_mz_score_eval.py
│ ├── run_isolation_forest_eval.py
│ ├── run_lof_eval.py
│ ├── run_vae_hyperopt.py
│ ├── run_moment_finetune_hyperopt.py
│ ├── train_final_vae.py
│ └── finetune_final_moment.py
├── notebooks/ # Example Jupyter notebooks
├── tests/ # (To be added) Unit and integration tests
├── DATASET/ # Input data
│ └── train.csv
├── lead-val-ids/ # Pre-generated K-fold validation IDs
│ └── val_id_fold0.pkl ...
├── results/ # Default output directory for models, logs, metrics
├── .gitignore
├── README.md # This file
├── requirements.txt # Python dependencies
└── LICENCE # 

## Installation 

## Installation

### Prerequisites

*   Git
*   Docker (Recommended for consistent environment)
*   NVIDIA GPU and NVIDIA Container Toolkit (if using GPU for deep learning models)

### Cloning the Repository

```bash
git clone https://gitlab.com/basu1999/tsfm-anomaly-paper.git # Or your new repo URL
cd tsfm-anomaly-paper
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
Docker Setup (Recommended)

This project uses a Docker environment to ensure reproducibility and manage dependencies, especially for GPU support.

Install Docker:

Windows & macOS: Download and install from docker.com/get-started

Ubuntu:

sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER # Add current user to docker group (logout/login required)
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Install NVIDIA Container Toolkit (for GPU support):
If you plan to leverage NVIDIA GPUs inside Docker:

# Follow instructions from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# Example for Ubuntu:
distribution=$( . /etc/os-release; echo $ID$VERSION_ID ) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Pull the Docker Image:


docker pull gcr.io/kaggle-gpu-images/python:latest
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Alternatively, a custom Dockerfile could be provided for a more minimal image based on nvidia/cuda or pytorch/pytorch images).

Run the Docker Container:
From the project root (tsfm-anomaly-paper/):

# For GPU support
docker run --gpus all -it --rm \
  --name tsfm_anomaly_env \
  -v "$(pwd)":/workspace \
  -w /workspace \
  gcr.io/kaggle-gpu-images/python:latest \
  /bin/bash

# For CPU only
# docker run -it --rm \
#   --name tsfm_anomaly_env \
#   -v "$(pwd)":/workspace \
#   -w /workspace \
#   gcr.io/kaggle-gpu-images/python:latest \
#   /bin/bash
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

--gpus all: Enables GPU access. Omit if not using GPU.

-it: Interactive terminal.

--rm: Removes the container when it exits.

--name tsfm_anomaly_env: Assigns a name to the container.

-v "$(pwd)":/workspace: Mounts the current project directory into /workspace inside the container.

-w /workspace: Sets the working directory inside the container to /workspace.
You will now be inside the container's bash shell, in the /workspace directory.

Environment Setup (Inside Docker)

Once inside the Docker container:

Upgrade pip and Install Dependencies:

pip install --upgrade pip
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Ensure requirements.txt is comprehensive and includes optuna, momentfm (if used directly), scikit-learn, pandas, numpy, torch, joblib, etc.

Dataset Setup

Download the LEAD Dataset:
The primary dataset used is train.csv from the "Energy Anomaly Detection" Kaggle competition (or a similar dataset).

Download link: https://www.kaggle.com/competitions/energy-anomaly-detection/data?select=train.csv .

Place the Dataset:
Create a directory named DATASET/ in the root of this project and place the downloaded train.csv file into it.

# From project root on your host machine, or inside Docker if downloaded there
mkdir -p DATASET
mv /path/to/your/downloaded/train.csv DATASET/
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

The expected structure is tsfm-anomaly-paper/DATASET/train.csv. The default data path in tsfm_ad_lib/config.py points here.

Fold ID Setup (for VAE and MOMENT hyperparameter optimization)

The VAE and MOMENT hyperparameter optimization scripts (run_vae_hyperopt.py, run_moment_finetune_hyperopt.py) rely on pre-generated K-fold splits that define which building_ids belong to the validation set for each fold.

Obtain Fold ID Files:

These files (val_id_fold0.pkl, val_id_fold1.pkl, ..., val_id_fold4.pkl) should be placed in a directory.

The default expected directory is tsfm-anomaly-paper/lead-val-ids/.

(Instruction to user: These files might be provided by the original author or you may need to generate them. A helper script scripts/generate_fold_ids.py (if we decide to add one) could create them based on sklearn.model_selection.KFold across unique building_ids from DATASET/train.csv.).


## Usage: Running Scripts

All scripts are located in the scripts/ directory and should be run from inside the Docker container and from the project root (/workspace).

General Notes on Running Scripts

PYTHONPATH: The scripts include a small header to add the project root to sys.path, allowing them to find the tsfm_ad_lib library. Alternatively, you can set export PYTHONPATH=$PYTHONPATH:/workspace inside the Docker container.

Output Directories: Scripts will create output subdirectories within results/ by default (e.g., results/iqr_eval/). This can be changed with the --output_dir argument.

Logging: Scripts use Python's logging. Log messages go to the console and, by default, to a .log file within the script's specific output directory. Use --log_level (e.g., DEBUG, INFO) to control verbosity.

Help: All scripts support a --help argument to display all available command-line options. Example: python scripts/run_iqr_eval.py --help.

Data Path: Most scripts take a --data_path argument. If your train.csv is not at DATASET/train.csv, you'll need to specify this.

