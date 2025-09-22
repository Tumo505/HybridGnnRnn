# Hybrid GNN-RNN Model for Cardiomyocyte Differentiation Prediction

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.4.0-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![Scikit-learn](https://img.shields.io/badge/sklearn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Accuracy](https://img.shields.io/badge/accuracy-96.67%25-brightgreen.svg)](https://github.com/Tumo505/HybridGnnRnn/blob/main/RESULTS.md)
[![Research](https://img.shields.io/badge/research-thesis%20project-blue.svg)](https://github.com/Tumo505/HybridGnnRnn)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Tumo505/HybridGnnRnn/issues)

![Hybrid Model Architecture](xai_visualizations/xai_dashboard.png)

A state-of-the-art deep learning framework combining Graph Neural Networks (GNN) and Recurrent Neural Networks (RNN) for predicting cardiomyocyte differentiation trajectories. Integrates spatial transcriptomics with temporal gene expression patterns to achieve **96.67% accuracy**.

## 🎯 Overview

This project combines spatial and temporal biological data analysis:

- **Spatial Analysis (GNN)**: Processes spatial transcriptomics data from human cardiac tissue
- **Temporal Analysis (RNN)**: Analyzes gene expression time series during cardiomyocyte differentiation  
- **Hybrid Fusion**: Combines both modalities for superior prediction performance
- **Explainable AI**: SHAP-based interpretability with cardiac gene pathway mapping

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Tumo505/HybridGnnRnn.git
cd HybridGnnRnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Train individual models
python scripts/training/train_enhanced_spatial_gnn.py
python scripts/training/train_enhanced_temporal_rnn.py

# Train hybrid model
python scripts/training/train_advanced_ensemble.py

# Generate XAI analysis
python scripts/xai/xai_real_data_analysis.py --embeddings-path analysis
```

## 📊 Results

| Model | Accuracy | F1-Score | Key Features |
|-------|----------|----------|--------------|
| **Hybrid Model** | **96.67%** | **96.45%** | Combined spatial + temporal |
| Temporal RNN | 96.88% | 96.67% | BiLSTM with attention |
| Spatial GNN | 65.29% | 64.12% | Graph attention networks |

## 🧬 Data Sources

- **Spatial Data**: Kuppe et al. (2022) - Human myocardial infarction spatial transcriptomics
- **Temporal Data**: Elorbany et al. (2022) - Cardiomyocyte differentiation time series

## 📚 Documentation

For comprehensive technical details, see:

- **[Technical Documentation](TECHNICAL_DOCUMENTATION.md)** - Complete model architecture, training procedures, and detailed analysis
- **[Results Analysis](RESULTS.md)** - Performance evaluation and statistical validation
- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions
- **[API Reference](docs/API.md)** - Code documentation and usage examples

## 🗂️ Project Structure

```text
HybridGnnRnn/
├── scripts/
│   ├── training/          # Model training scripts
│   ├── xai/              # Explainable AI analysis
│   └── data_processing/  # Data preprocessing
├── src/                  # Core model implementations
├── results/              # Experimental results
├── docs/                 # Documentation
└── requirements.txt      # Dependencies
```

## 🛠️ Core Technologies

- **PyTorch & PyTorch Geometric** - Deep learning and graph neural networks
- **Scikit-learn** - Machine learning utilities and metrics
- **SHAP** - Explainable AI and feature importance
- **Matplotlib & Seaborn** - Visualization and plotting

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{hybrid_gnn_rnn_2024,
  title={Hybrid GNN-RNN Model for Cardiomyocyte Differentiation Prediction},
  author={[Your Name]},
  year={2024},
  url={https://github.com/Tumo505/HybridGnnRnn}
}
```

## 🔗 References

1. **Kuppe, C., et al.** (2022). Spatial multi-omic map of human myocardial infarction. *Nature*, 608, 766–777.
2. **Elorbany, R., et al.** (2022). Single-cell sequencing reveals lineage-specific dynamic genetic regulation during human cardiomyocyte differentiation. *PLoS Genetics*, 18(1).

---

For questions or support, please [open an issue](https://github.com/Tumo505/HybridGnnRnn/issues).