# srep-meta-codes
##### by [Mo Shakiba](https://github.com/moneuron)

## Overview

**This repository contains the Python codes used for analysis and visualization in the paper titled "[Joining up the scattered anticancer knowledge on auraptene and umbelliprenin: a meta-analysis](https://www.nature.com/articles/s41598-024-62747-z)" by Mohammadhosein Shakiba and Fatemeh B. Rassouli.**

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
├── README.md
├── Reg-Scatter-All.py
└── requirements.txt
```

- `ML/RandomForestClassifier.py`: Python script implementing the Random Forest Classifier for identifying influential features.
- `ML/RandomForestRegressor.py`: Python script implementing the Random Forest Regressor (if applicable).
- `3D-Viab-Dose-Time.py`: Python script for generating 3D scatter plots of viability, dose, and time.
- `Cancer-Pie.py`: Python script for generating pie charts of cancer types distribution.
- `Cancer-Viab-Dose.py`: Python script for plotting the relationship between cancer type, viability, and dose.
- `Heat-Cor.py`: Python script for generating heatmaps of correlations between variables.
- `README.md`: This file, providing an overview of the repository.
- `Reg-Scatter-All.py`: Python script for plotting the relationship between viability and dose across different cell lines.
- `requirements.txt`:  required packages to run the code.

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

## Citation
```
Shakiba, M., Rassouli, F.B. Joining up the scattered anticancer knowledge on auraptene and umbelliprenin: a meta-analysis. Sci Rep 14, 11770 (2024). https://doi.org/10.1038/s41598-024-62747-z
```
