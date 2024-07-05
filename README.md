# scm_irl

Inverse rl applied to navigation of marine autonomous robots

## Table of Content:

- [Intallation](#installation)
- [Usage](#usage)

## Installation

Create environment

```batch
conda create -n scm python=3.11.8
conda activate scm
```

Prerequisites:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Clone this repo and fetch the submodules:

```
git submodule update --init --recursive
```

Install `ai-navigator-python-library` follow the instructions inside

Install  `scm_irl`
```batch
pip install -e .
```


## Usage

--




