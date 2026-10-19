# srep-meta-codes

Analysis and visualization code for:

> Shakiba M, Rassouli FB. **Joining up the scattered anticancer knowledge on auraptene and umbelliprenin: a meta-analysis.** *Scientific Reports* 14, 11770 (2024).

[![Paper](https://img.shields.io/badge/DOI:-10.1038%2Fs41598--024--62747--z-blue)](https://doi.org/10.1038/s41598-024-62747-z)

## What this is

Auraptene (AUR) and umbelliprenin (UMB) are prenylated coumarins with reported anticancer activity in cell lines. The paper synthesizes 27 in vitro studies to quantify dose-dependent effects on cell viability, compare the two compounds, and rank which factors (dose, cancer type, treatment duration, compound) most strongly influence toxicity.

This repository contains the Python scripts for the figures and the machine-learning analysis. The extracted study data are archived separately on Dryad.

## Installation

```bash
git clone https://github.com/moneuron/srep-meta-codes.git
cd srep-meta-codes
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Use Python 3 with the packages listed in `requirements.txt`.

## Data

The dataset is deposited at:

[![Dryad](https://img.shields.io/badge/Dryad-10.5061%2Fdryad.cvdncjt9k-green)](https://doi.org/10.5061/dryad.cvdncjt9k)
[![Kaggle](https://img.shields.io/badge/Kaggle-aur--umb--dataset-20BEFF)](https://www.kaggle.com/datasets/moneuron/aur-umb-dataset)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-aur--umb--dataset-FFD21E)](https://huggingface.co/datasets/moneuron/aur-umb-dataset)

| File | Contents |
| --- | --- |
| `AUR.csv` | Extracted observations for auraptene |
| `UMB.csv` | Extracted observations for umbelliprenin |
| `combined.csv` | AUR and UMB observations combined |

Download the files and place them in a `data/` folder at the repository root:

```
srep-meta-codes/
├── data/
│   ├── AUR.csv
│   ├── UMB.csv
│   └── combined.csv
└── ...
```

The column definitions are in the dataset's own README.

## Usage

The analysis scripts are organized by the figure number in the paper. After placing the data files in `data/`, run a script from the repository root. Each script displays its plot(s) and does not write an output file:

```bash
python fig3/AUR-Pie.py
python fig3/UMB-Pie.py
python fig4/AUR-3D-Viab-Dose-Time.py
python fig4/UMB-3D-Viab-Dose-Time.py
python fig5/AUR-Cancer-Viab-Dose.py
python fig5/UMB-Cancer-Viab-Dose.py
python fig6/Reg-Scatter-All.py
python fig7/AUR-Heat-Cor.py
python fig7/UMB-Heat-Cor.py
python fig8/RandomForestClassifier.py
python fig8/RandomForestRegressor.py
```

The figure scripts use `AUR.csv` or `UMB.csv`; the figure 6 and figure 8 scripts use `combined.csv`.


## Citation

If you use this code or data, please cite the paper and the dataset.

**Paper**

```
Shakiba, M., Rassouli, F.B. Joining up the scattered anticancer knowledge on
auraptene and umbelliprenin: a meta-analysis. Sci Rep 14, 11770 (2024).
https://doi.org/10.1038/s41598-024-62747-z
```

**Dataset**

```
Shakiba, M., Rassouli, F.B. (2024). Data from: Joining up the scattered anticancer
knowledge on auraptene and umbelliprenin: A meta-analysis [Dataset]. Dryad.
https://doi.org/10.5061/dryad.cvdncjt9k
```