# Hybrid GNN-RNN Framework for iPSC-Cardiomyocyte Differentiation Prediction

## Overview
This project implements a hybrid deep learning framework combining Graph Neural Networks (GNNs) and Recurrent Neural Networks (RNNs) to predict cardiomyocyte differentiation efficiency using spatial multi-omics data.

## Project Structure
```
├── data/                          # Large datasets (gitignored)
│   ├── selected_datasets/         # Core datasets for model training
│   │   ├── temporal_data/         # Time-series data for RNN training
│   │   ├── spatial_transcriptomics/ # Spatial RNA-seq for GNN training
│   │   ├── spatial_epigenomics/   # ATAC-seq and epigenomic data
│   │   ├── ipsc_cardiomyocyte/    # iPSC-CM differentiation data
│   │   └── validation_mi_datasets/ # MI data for validation
│   └── unused_datasets/           # Secondary datasets
├── data_catalog/                  # Dataset metadata and configurations
├── notebooks/                     # Jupyter notebooks for development
│   ├── 0_download/               # Data download and initial exploration
│   ├── 1_preprocess/             # Data preprocessing and QC
│   ├── 2_model_development/      # Model architecture development
│   ├── 3_training/               # Model training scripts
│   ├── 4_validation/             # Validation and testing
│   └── 5_explainability/         # XAI and interpretability
├── src/                          # Source code
│   ├── datasets/                 # PyTorch datasets and data loaders
│   ├── models/                   # Model architectures
│   ├── eval/                     # Evaluation metrics and utilities
│   └── xai/                      # Explainability tools
├── configs/                      # Hydra configuration files
├── experiments/                  # Experiment logs and checkpoints
└── reports/                      # Generated reports and markers database
    └── xai_reports/              # XAI analysis reports
```

## Key Features
- **Hybrid Architecture**: Combines spatial (GNN) and temporal (RNN) modeling
- **Multi-omics Integration**: Transcriptomics + Epigenomics
- **Memory Optimized**: Designed for M1 MacBook Pro (16GB RAM)
- **Explainable AI**: Comprehensive interpretability analysis
- **Reproducible**: Hydra configuration management

## Research Objectives
1. Develop hybrid GNN-RNN framework for differentiation prediction
2. Identify critical molecular markers and spatial relationships
3. Optimize iPSC-CM differentiation protocols
4. Predict functional maturation of iPSC-CMs
5. Provide explainable AI insights for biological understanding
6. Validate against myocardial infarction remodeling data

## Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage
Start with the notebooks in order:
1. `notebooks/0_download/` - Data acquisition and exploration
2. `notebooks/1_preprocess/` - Data preprocessing and QC
3. `notebooks/2_model_development/` - Model architecture development
4. `notebooks/3_training/` - Model training and optimization
5. `notebooks/4_validation/` - Validation and testing
6. `notebooks/5_explainability/` - XAI analysis and marker discovery

## Key Datasets
- **GSE175634**: Time-series iPSC-CM differentiation (7 timepoints, 230K+ cells)
- **Spatial Transcriptomics**: Multiple heart spatial datasets
- **Spatial Epigenomics**: ATAC-seq and regulatory data
- **MI Validation**: Myocardial infarction spatial multi-omics

## Citation
If you use this framework, please cite:
[Paper details to be added upon publication]

## License
[License to be determined]
