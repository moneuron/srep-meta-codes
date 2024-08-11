[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/joining-up-the-scattered-anticancer-knowledge/classification-on-aur-umb-dataset)](https://paperswithcode.com/sota/classification-on-aur-umb-dataset?p=joining-up-the-scattered-anticancer-knowledge)

# [Joining up the scattered anticancer knowledge on auraptene and umbelliprenin: a meta-analysis](https://www.nature.com/articles/s41598-024-62747-z)
> by [Mo Shakiba](https://github.com/moneuron)

This repository contains the Python codes used for data analysis and visualization in the paper titled "Joining up the scattered anticancer knowledge on auraptene and umbelliprenin: a meta-analysis" by Mohammadhosein Shakiba and Fatemeh B. Rassouli.

## Overview

The paper is a meta-analysis of the anticancer effects of auraptene and umbelliprenin, two naturally occurring prenylated coumarins, across various human cancer cell lines. The analysis synthesizes evidence from 27 in vitro studies and employs data visualization techniques, machine learning approaches, and statistical meta-analysis methods to quantify their anticancer efficacy.

## Repository Structure

The repository is organized as follows:

```
├── ML/
│   ├── RandomForestClassifier.py
│   └── RandomForestRegressor.py
├── 3D-Viab-Dose-Time.py
├── Cancer-Pie.py
├── Cancer-Viab-Dose.py
├── Heat-Cor.py
├── Reg-Scatter-All.py
└── README.md
```

- `ML/RandomForestClassifier.py`: Python script implementing the Random Forest Classifier for identifying influential features.
- `ML/RandomForestRegressor.py`: Python script implementing the Random Forest Regressor (if applicable).
- `3D-Viab-Dose-Time.py`: Python script for generating 3D scatter plots of viability, dose, and time.
- `Cancer-Pie.py`: Python script for generating pie charts of cancer types distribution.
- `Cancer-Viab-Dose.py`: Python script for plotting the relationship between cancer type, viability, and dose.
- `Heat-Cor.py`: Python script for generating heatmaps of correlations between variables.
- `Reg-Scatter-All.py`: Python script for plotting the relationship between viability and dose across different cell lines.
- `README.md`: This file, providing an overview of the repository.

## Requirements

To run the Python scripts in this repository, you'll need to have the required packages installed.

You can install these packages using the following command:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository to your local machine.
2. Place the raw data files in the appropriate directory (if applicable).
3. Run the desired Python scripts to generate the corresponding visualizations or analyses.

Please note that the scripts may require modification to match the specific data formats and requirements of your analysis.

If you use this code, please cite it as below:
> Shakiba, M., Rassouli, F.B. Joining up the scattered anticancer knowledge on auraptene and umbelliprenin: a meta-analysis. Sci Rep 14, 11770 (2024). https://doi.org/10.1038/s41598-024-62747-z
