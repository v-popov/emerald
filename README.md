# Emerald
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/v-popov/emerald/master?filepath=model_demo.ipynb)

A tool designed for Application Managers and Data Privacy professionals at Citi which scans application databases and flags data elements contained in the changing formula of personally identifiable information (PII), using NLP and pattern recognition.

This repository contains the core algorithms for querying unnormalized data tables. The example usage can be found in model-demo.ipynb notebook file. Make sure to install all the necessary dependencies from the file environment.yml before running the notebook. In Anaconda it can be done using the following commands:

1) Create a new virtual environment with the specified dependencies:

conda env create -f environment.yml

2) Activate the environment created in the previous step:

conda activate emerald
