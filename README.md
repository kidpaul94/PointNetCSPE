# PointNetCSPE
Combination of Contact Surface Pair Estimator (CSPE) and [PointNetGPD](https://arxiv.org/pdf/1809.06267.pdf) for Offline 6DOF Grasp Detection. The framework takes a complete object point cloud file to generate possible contact surface pairs and their robustness (a.k.a grasp dictionaries) for a parallel jaw gripper using both analytical and data-driven models.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Download Process](#download-process)
- [How to Run](#how-to-run)
    - [CSPE_v2](#cspe_v2)
    - [Extra Processing](#extra-processing)
    - [Quality Estimation (ML)](#quality-estimation)
- [ToDo Lists](#todo-lists)

---

## Repository Structure

    ├── ML_modules        # DL modules 
    │   ├── engine.py
    │   ├── eval.py
    │   ├── train.py
    │   ├── transforms.py 
    │   ├── utils.py 
    │   ├── model
    │   │   └── pointnet.py
    │   ├── weights
    │   │   └── pointnetgpd_3class.model
    ├── dataset
    |   ├── train         # ML training data
    |   └── test          # ML test data
    ├── images
    ├── objects
    │   ├── dicts         # .txt grasp files
    │   └── pcds          # .pcd files
    ├── CSPE_utils.py
    ├── CSPE_v2.py
    ├── gripper_config.py 
    └── helper.py 

## Download Process

    git clone https://github.com/kidpaul94/PointNetCSPE.git
    cd PointNetCSPE/
    pip3 install -r requirements.txt
    pip3 install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

## How to Run

### CSPE_v2:

> **Note**
`CSPE_v2.py` receives several different arguments. Run the `--help` command to see everything it receives.

    python3 CSPE_v2.py --help

### Extra Processing:

> **Note**
`helper.py` receives several different arguments to convert a CAD model to a point cloud or visualize grasp configurations. Run the `--help` command to see everything it receives.

    python3 helper.py --help

### Quality Estimation:

> **Note**
`train.py` and `eval.py` receives several different arguments. Run the `--help` command to see everything it receives.

    cd ML_modules/
    python3 train.py --help
    python3 eval.py --help

## ToDo Lists

| **ML pipeline** | ![Progress](https://progress-bar.dev/100) |
| --- | --- |
| **Documentation** | ![Progress](https://progress-bar.dev/50) |
