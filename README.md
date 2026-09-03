# GAVOGO: A Leak-Free Benchmark for Cross-Domain Lane-Change Intention Prediction

Official code and scripts for **"On the Limits of Cross-Domain Lane-Change Prediction: A Leak-Free Benchmark Reveals Near-Chance Zero-Shot Transfer"**.

## Overview

Conventional evaluation protocols for cross-domain lane-change intention prediction suffer from vehicle-identity leakage: negative (lane-keeping) examples are sampled from vehicles that never change lanes, letting models learn *which vehicle it is* rather than *when a maneuver is imminent*. This repository provides:

- **Per-dataset adapters** that convert seven heterogeneous naturalistic driving datasets (highD, NGSIM, MiTra, ETRI, EMT, uniD, exiD) into a shared canonical schema
- **Within-vehicle negative sampling** and **road-frame canonicalization**, which remove the two leaks identified in the paper
- A **group–label purity diagnostic** for detecting this shortcut in any grouped dataset
- All scripts used to reproduce the tables and figures in the paper

## Datasets

This repository does **not** redistribute raw trajectory data. highD, exiD, uniD, NGSIM, and EMT are available from their original providers; MiTra and ETRI are governed by third-party licenses. Download pointers and preprocessing instructions are provided in `scripts/`.

## Repository structure

```
gavogo/
├── scripts/          # Data preparation, feature engineering, evaluation scripts
├── src/               # Core library: adapters, sampling, canonicalization, models
├── requirements.txt   # Python dependencies
└── .gitignore
```


## Installation

```bash
git clone https://github.com/qkdlsem-maker/gavogo.git
cd gavogo
pip install -r requirements.txt
```

## Citation

If you use this benchmark or code, please cite:

```
[논문 정식 출판 후 서지정보 업데이트 예정]
```
