# scProtVelo
`scProtVelo` is a Python package for modeling translation dynamics from paired single-cell mRNA + mass spectrometry-based protein expression data.
This repository contains the source code for `scProtVelo` as well as data and notebooks to reproduce the results published in https://doi.org/10.1126/science.adr8785. 

Should you not have paired single-cell mRNA + protein expression data, but two independent datasets of the same biological system, you can follow our workflow to computationally pair them. You can find the full code and all used data here: https://zenodo.org/records/15554000.

## Installation
We recommend using the provided `environment.yml` to create a Conda environment with all required dependencies.

Then install the package itself with

```pip install .```

## Usage
After installation, you can import the package in Python:

```import scprotvelo```
