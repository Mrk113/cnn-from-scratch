# cnn-from-scratch

 CNN implementation built from scratch with CuPy for GPU-backed training.

## 📑 Table of Contents
- [📖 Overview](#-overview)
- [✨ Features](#-features)
- [⚙️ Installation](#️-installation)
- [🚀 Run](#-run)
- [📚 Documentation](#-documentation)
- [📂 Project Structure](#-project-structure)
- [🧪 Testing](#-testing)

## 📖 Overview
This project implements core deep learning building blocks, datasets, layers, transforms, losses, schedulers, and training loops using CuPy for GPU acceleration and minimal external dependencies. It is designed to be easily extendable.

## ✨ Features
- CuPy-based tensor operations for GPU-friendly execution
- Dataset loaders for MNIST and CIFAR-10 with transforms and normalization
- Common layers (convolution, pooling, activation, batch norm, fully connected)
- Training utilities with learning-rate schedulers and logging 

## ⚙️ Installation
Use Conda to create and install Python dependencies, then activate the enviroment:

```bash
conda env create -f environment.yml
conda activate cnn
```

## 🚀 Run
Run a training script:

```bash
python src/cnn.py
```

## 📂 Project Structure

```text
cnn-from-scratch/
├─ src/
│  ├─ core/
│  │  ├─ datasets/        # MNIST, CIFAR-10 loaders & helpers
│  │  ├─ transforms/      # Compose, Normalize, augmentations
│  │  ├─ layers/          # Conv, pooling, activations, dense
│  │  ├─ losses/          # Loss base, MSE, cross-entropy
│  │  ├─ lr_scheduler/    # Cosine annealing, step LR
│  │  ├─ logging/         # Logger base and W&B logger
│  │  └─ utils.py         # Download/extract/read helpers
│  └─ cnn.py              # Example training entrypoint
├─ tests/                 # Unit tests for core modules
├─ data/                  # Dataset storage (MNIST, CIFAR-10)
└─ README.md
```

## 🧪 Testing
Run the test suite with pytest:

```bash
pytest
```
