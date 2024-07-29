# Inprocessing-debiasing-methodology-for-multiple-sensitive-attributes

## Description

This project implements an adversarial framework to debias deep learning models for multiple sensitive attributes/confounding attributes using gradient reversal and a dynamic loss balancing technique. The approach achieves a fairer model that shows less disparity amongst minority subgroups.

## Features

- Adversarial debiasing framework
- Gradient reversal technique
- Dynamic loss balancing
- Support for multiple sensitive attributes
- Improved fairness for minority subgroups

## Installation

This project uses a `requirements.txt` file for package management and a `setup.py` file for installation. To set up the project:

1. Clone the repository: git clone https://github.com/JVedant/Inprocessing-debiasing-methodology-for-multiple-sensitive-attributes.git
2. cd Inprocessing-debiasing-methodology-for-multiple-sensitive-attributes
3. Run the setup script:
python setup.py install

## Usage

To get started with the project:

1. Navigate to the source directory:
cd Inprocessing-debiasing/src

2. Run the main script to train the model:
python main.py

The project structure includes the following key files:

- `config.py`: Configuration settings
- `dataset.py`: Dataset handling
- `inference.py`: Inference functions
- `losses.py`: Loss functions
- `main.py`: Main script to run the training process
- `models/debias_models.py`: Debiasing model implementations
- `training.py`: Training functions
- `validation.py`: Validation functions

## Contributing

Contributions to improve the project are welcome. Please feel free to submit a Pull Request.


## Contact

[Vedant Joshi]
[vedantjoshi@asu.edu]