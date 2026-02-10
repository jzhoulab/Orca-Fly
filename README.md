# Orca-Fly
The repository contains code for Orca-Fly, an adaptiation of Orca frameworks for multi-scale genome structure prediction in early Drosophila embryogenesis.

## Get started
If you just need predictions for one or a handful of variants, we have provided the core functionalities on a web server: [orcafly.zhoulab.io].

## Install and run Orca-Fly locally
#### Pre-requisites: Python environment and Selene library.

1. orca_env Python environment

    The enviornment setup involves 3 steps, install (1) Python 3.9, (2) Pytorch, and (3) the remaining packages.
    This is to save time for conda to solve the dependency for Pytorch.

  - Install [conda/miniconda](https://docs.conda.io/en/latest/miniconda.html) if you don't have it already.
  - Create the environment from the orca_env_part1.yml file: `conda env create -f orca_env_part1.yml`
  - Activate the environment: `conda activate orca_env`
  - Install Pytorch following the [Pytorch installation guide](https://pytorch.org/get-started/locally/), choose the appropriate parameters (CPU or GPU, OS, etc.) for your system.
  - Install the remaining packages: `conda env update -f orca_env_part2.yml`

2. Install Selene (under the orca_env environment)
  ```bash
  git clone https://github.com/kathyxchen/selene.git
  cd selene
  git checkout custom_target_support
  python setup.py build_ext --inplace
  python setup.py install 
  ```

Now you are ready to run Orca-Fly locally, with the orca repository cloned and resource files downloaded.

#### Clone the Orca repository
```bash
git clone https://github.com/jzhoulab/Orca-Fly.git
cd orca
```

## Predict with pretrained models
Please download the dataset and pretrained model weight by the link [TBD]. You can use `predict.ipynb` to get predictions of interested regions.

## Train Orca-Fly models
If you have set up Orca with its dependencies and has the necessary GPU resources, you can train new models following the example code under the `train` directory to train new Orca-Fly models.
