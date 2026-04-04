# Inverse Thermal Comfort Optimisation in UK Schools

Reproducible code for the paper:

> **Data-Driven Thermal Comfort Control in UK Schools Using Evolutionary Surrogate Inversion**
> Ismail Alarab, Simant Prakoonwit
> ***

---

## What this does

Most thermal comfort research stops at *predicting* discomfort.
This framework goes further: given a discomfort instance, it recommends the **smallest practical changes**, adjusting the temperature setpoint, opening a window, or adding a layer of clothing to restore comfort while minimising energy use.

Key findings:
- **Gradient Boosting + Differential Evolution** is the best-performing combination (99.1% comfort restoration, MATAW = 0.73°C)
- Adding clothing and window adjustments reduces heating-season temperature changes by **39–65%**
- **38% of cold-classroom discomfort** can be resolved by clothing adjustment alone: no HVAC needed

---

## Repository structure

```
thermal-comfort-inverse-optimisation/
│
├── thermal_comfort_study.py   # full reproducible pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # this file
│
└── outputs/                   # generated after running (not tracked by git)
    ├── fig1_dataset_overview.png
    ├── fig2_confusion.png
    ├── fig3_haf.png
    ├── fig4_temp_dist.png
    ├── fig5_success_rate.png
    ├── table_classification_metrics.csv
    ├── table_optimisation_results.csv
    └── table_statistical_tests.csv
```

---

## Dataset

## Dataset
This study uses the **Korsavi UK-schools dataset** from the
[ASHRAE Global Occupant Behavior Database](https://ashraeobdatabase.com/).

The CSV files are included in this repository (`korsavi_part1.csv`,
`korsavi_part2.csv`) for reproducibility. They are publicly available
to anyone who access the ASHRAE database portal.

To obtain the files directly from the source:
1. Go at https://ashraeobdatabase.com/
2. Under **Export**, select the following parameters:
   - Country: **UK: Coventry**
   - Building type: **Educational: Classroom**
   - Study: **Study 1**
3. Export the zipped folder containing both datasets and rename them as:
   `korsavi_part1.csv` and `korsavi_part2.csv`

Original dataset citation:
> Korsavi, S. S., Montazami, A., & Brusey, J. (2018). Developing a design
> framework to facilitate adaptive behaviors in UK schools.
> *Energy and Buildings*, 179, 360–373.
---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/thermal-comfort-inverse-optimisation.git
cd thermal-comfort-inverse-optimisation
pip install -r requirements.txt
```

Python 3.9+ recommended.

---

## Usage

```bash
python thermal_comfort_study.py
```

The script runs the full pipeline end-to-end:

| Step | What happens |
|------|-------------|
| 0 | Dataset download / verification |
| 1 | Data loading, PMV computation, comfort labelling |
| 2 | Classifier training with 10-fold CV (LR, MLP, Gradient Boosting) |
| 3 | Identification of optimisation candidates |
| 4 | Inverse optimisation - all model × case × solver combinations |
| 5 | Two-proportion z-tests on success rates |
| 6 | Prescriptive analytics (single-action restoration rates) |
| 7 | Figure generation and CSV export |

Expected runtime on a standard laptop CPU: **20–40 minutes**
(dominated by Gradient Boosting + Differential Evolution across 330 candidates).

---

## Methods summary

Three surrogate classifiers:
- Logistic Regression (LR)
- Multilayer Perceptron (MLP)
- Gradient Boosting (GB)

Two solvers:
- **COBYLA** - gradient-free local solver (baseline)
- **Differential Evolution (DE)**  population-based global solver (proposed)

Three feature-action spaces:
- Temperature only
- Temperature + window operation
- Temperature + window + clothing adjustment

All hyperparameters are fixed in the script for full reproducibility.
Random seed: `42`.

---

## Requirements

```
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
requests
```

See `requirements.txt` for pinned versions.

---

## Citation

If you use this code, please cite:

```bibtex
TBC
```

---

## Licence

MIT - free to use and adapt with attribution.


## Contact

Ismail Alarab - i.alarab@chi.ac.uk - University of Chichester
