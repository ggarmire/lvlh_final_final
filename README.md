## Overview 
Classical LV models with pairwise species interactions predict that large, complex communities are inherently unstable. This project incorporates demographic stage structure into the gLV model to evaluate how diversity across stages affects the asymptotic stability of the community. 

This codebase evaluates complex community matrices by computing the Jacobian of system dynamics and evaluating stability using the leading eigenvalue of J. 

## Structure 
```
lvlh_final_final/
│
├── pyproject.toml             # package build instructions
├── README.md                  # Project documentation
│
├── lv_lh_functions/           # custom python package for project 
│   ├── __init__.py            # exposes functions at package level
│   ├── matrices.py            # generates growth rates/demographic matrices (r/R), interaction matrices (A and B), initial conditions (x0)
│   ├── solvers.py             #
│   └── sweeps.py              # 
│
├── scripts/                   # Executable analysis and plotting scripts
│   └── [placeholder]               
├── data/                      # output directory for saved simulation data files (.npz)
│   └── R_dependence/
│

└── figures/                   # Output directory for generated plots
```

## Install 
1. Create and activate a virtual environment: 
```
python3 -m venv lvlh_env source 
lvlh_env/bin/activate
```
2. Install the package 
```
pip install -e .
```
