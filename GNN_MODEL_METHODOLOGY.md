# Graph Neural Network (GNN) Model Methodology
## Cardiomyocyte Differentiation Classification

### Table of Contents
1. [Overview](#overview)
2. [Dataset and Data Processing](#dataset-and-data-processing)
3. [Model Architecture](#model-architecture)
4. [Training Methodology](#training-methodology)
5. [Implementation Details](#implementation-details)
6. [Running Instructions](#running-instructions)
7. [File Structure](#file-structure)
8. [Dependencies](#dependencies)

---

## Overview

This project implements a Graph Neural Network (GNN) for classifying cardiomyocyte differentiation stages using spatial transcriptomics data from 10X Genomics Visium technology. The model leverages the spatial relationships between cells to predict differentiation trajectories in cardiac tissue.

### Research Objectives
- **Primary Goal**: Classify authentic cardiomyocyte subtypes using spatial gene expression data with biological naming
- **Biological Relevance**: Understanding cardiac cell type diversity and spatial organization in tissue
- **Technical Innovation**: Advanced GNN architecture with biological naming system for interpretable cell type classification

### Key Features
- **Authentic Biological Data**: Uses real 10X Genomics Visium spatial transcriptomics data with biological cell type naming
- **Biological Classification**: Classifies 5 distinct cardiomyocyte subtypes with meaningful biological names
- **Graph Structure**: Constructs spatial graphs based on physical proximity of tissue spots
- **Advanced Architecture**: Enhanced GNN combining Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN)
- **Comprehensive Training**: Implements stratified sampling, class weighting, early stopping, and overfitting prevention
- **Interpretable Results**: Multi-metric assessment with biological cell type names and pathway analysis

---

## Dataset and Data Processing

### Data Source
- **Technology**: 10X Genomics Visium Spatial Gene Expression
- **Tissue Type**: Cardiac tissue samples
- **Data Format**: Spatial transcriptomics with coordinate information
- **Preprocessing**: Quality control, normalization, and feature selection

**Dataset Citation**: Kuppe, C., Ramirez Flores, R. O., Li, Z., Hayat, S., Levinson, R. T., Liao, X., Hannani, M. T., Tanevski, J., Wünnemann, F., Nagai, J. S., Halder, M., Schumacher, D., Menzel, S., Schäfer, G., Hoeft, K., Cheng, M., Ziegler, S., Zhang, X., Peisker, F., … Kramann, R. (2022). Spatial multi-omic map of human myocardial infarction. *Nature*, 608, 766–777. https://doi.org/10.1038/s41586-022-05060-x

### Data Processing Pipeline

#### Biological Cell Type Classification

The enhanced framework now uses biologically meaningful cell type names instead of generic numerical labels:

**Cardiomyocyte Subtypes:**
- **Atrial Cardiomyocytes**: Cells from the atrial chambers with specialized atrial gene expression
- **Ventricular Cardiomyocytes**: Cells from the ventricular chambers with contractile specialization
- **Pacemaker Cells**: Specialized cells responsible for cardiac rhythm generation
- **Conduction System Cells**: Cells specialized for electrical signal propagation
- **Immature Cardiomyocytes**: Developing or less differentiated cardiomyocyte cells

This biological naming system provides meaningful interpretation of model predictions and enables better correlation with cardiac biology literature.

#### 1. Raw Data Loading
```python
# Load 10X Genomics data with spatial coordinates
from src.data_processing.authentic_10x_processor import Authentic10XProcessor

processor = Authentic10XProcessor()
data = processor.load_cached_data()
```

#### 2. Quality Control
- **Cell Filtering**: Remove low-quality spots based on gene count and UMI thresholds
- **Gene Filtering**: Exclude genes with low expression across samples
- **Spatial Validation**: Ensure coordinate integrity and spatial consistency

#### 3. Normalization
- **Library Size Normalization**: Scale by total counts per spot
- **Log Transformation**: Apply log1p transformation for variance stabilization
- **Feature Scaling**: StandardScaler for model input preparation

#### 4. Graph Construction
```python
# Build spatial graph based on physical proximity
def build_spatial_graph(coordinates, k_neighbors=6):
    # K-nearest neighbors in spatial coordinates
    # Euclidean distance-based edge weights
    # Symmetric adjacency matrix construction
```

#### 5. Feature Engineering
- **Cardiac Gene Selection**: Prioritize known cardiac markers
- **Spatial Features**: Include coordinate information
- **Expression Profiles**: Aggregate neighborhood expression patterns

### Data Statistics
- **Total Spots**: ~2,000-5,000 depending on sample
- **Genes**: ~15,000-20,000 after filtering
- **Spatial Resolution**: 55μm spot diameter
- **Graph Connectivity**: Average 6 neighbors per node

---

## Model Architecture

### AdvancedCardiomyocyteGNN

The model implements a hybrid architecture combining multiple graph neural network layers with attention mechanisms and skip connections.

#### Architecture Components

##### 1. Input Processing
```python
# Input feature projection and normalization
self.input_projection = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.LayerNorm(hidden_dim)
)
```

##### 2. Graph Attention Network (GAT) Layers
```python
# Multi-head attention for spatial relationships
self.gat_layers = nn.ModuleList([
    GATConv(
        in_channels=layer_input_dim,
        out_channels=hidden_dim // num_heads,
        heads=num_heads,
        dropout=dropout,
        edge_dim=edge_dim
    ) for _ in range(num_gat_layers)
])
```

##### 3. Graph Convolutional Network (GCN) Layers
```python
# Message passing for feature aggregation
self.gcn_layers = nn.ModuleList([
    GCNConv(hidden_dim, hidden_dim)
    for _ in range(num_gcn_layers)
])
```

##### 4. Feature Fusion
```python
# Combine GAT and GCN representations
def feature_fusion(self, gat_out, gcn_out):
    # Attention-weighted combination
    attention_weights = self.fusion_attention(torch.cat([gat_out, gcn_out], dim=-1))
    fused = attention_weights * gat_out + (1 - attention_weights) * gcn_out
    return fused
```

##### 5. Skip Connections
```python
# Residual connections for gradient flow
def forward_with_skip(self, x, edge_index):
    residual = x
    x = self.graph_layers(x, edge_index)
    x = x + residual  # Skip connection
    return self.layer_norm(x)
```

##### 6. Classification Head
```python
# Final prediction layers with regularization
self.classifier = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim // 2, num_classes)
)
```

### Model Specifications
- **Parameters**: ~14.2M trainable parameters
- **Hidden Dimensions**: 256 (configurable)
- **GAT Heads**: 8 attention heads
- **GCN Layers**: 3 layers
- **Dropout**: 0.3 (adaptive)
- **Skip Connections**: Every 2 layers

---

## Training Methodology

### Training Pipeline

#### 1. Data Splitting Strategy
```python
# Stratified train/validation/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    features, labels, test_size=0.4, 
    stratify=labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, 
    stratify=y_temp, random_state=42
)
```

#### 2. Class Balancing
```python
# Compute class weights for imbalanced data
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
```

#### 3. Optimization Strategy
```python
# AdamW optimizer with weight decay
optimizer = optim.AdamW(
    model.parameters(), 
    lr=0.001, 
    weight_decay=1e-4
)

# Learning rate scheduling
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

#### 4. Training Loop Features
- **Early Stopping**: Prevent overfitting (patience=15)
- **Gradient Clipping**: Stabilize training (max_norm=1.0)
- **Mixed Precision**: Memory efficiency (optional)
- **Batch Processing**: Variable batch sizes based on graph size

#### 5. Regularization Techniques
- **Dropout**: Adaptive rates (0.1-0.5)
- **Weight Decay**: L2 regularization (1e-4)
- **Layer Normalization**: Stabilize activations
- **Data Augmentation**: Spatial jittering (optional)

### Training Configuration
```python
training_config = {
    'optimizer': 'AdamW',
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'scheduler': 'ReduceLROnPlateau',
    'max_epochs': 400,
    'patience': 60,
    'batch_size': 'full_graph',
    'gradient_clip': 1.0,
    'hidden_dim': 128,
    'dropout': 0.4
}
```

---

## Implementation Details

### Key Classes and Functions

#### 1. Data Processing
```python
class Authentic10XProcessor:
    """Handles 10X Genomics data loading and preprocessing"""
    
    def load_cached_data(self):
        """Load preprocessed spatial transcriptomics data"""
        
    def build_spatial_graph(self, coordinates, k=6):
        """Construct k-NN spatial graph"""
        
    def prepare_features(self, expression_data):
        """Normalize and scale features"""
```

#### 2. Model Definition
```python
class AdvancedCardiomyocyteGNN(nn.Module):
    """Advanced GNN for cardiomyocyte classification"""
    
    def __init__(self, input_dim, hidden_dim, num_classes):
        """Initialize model architecture"""
        
    def forward(self, x, edge_index, batch=None):
        """Forward pass with attention and fusion"""
        
    def get_model_info(self):
        """Return model architecture details"""
```

#### 3. Training Infrastructure
```python
class CardiomyocyteTrainer:
    """Comprehensive training pipeline"""
    
    def setup_training(self, model, data_loaders):
        """Configure optimization and scheduling"""
        
    def train_epoch(self, epoch):
        """Single epoch training with metrics"""
        
    def validate_epoch(self):
        """Validation with early stopping check"""
        
    def evaluate_model(self, test_loader):
        """Comprehensive model evaluation"""
```

### Performance Optimizations

#### 1. Memory Management
- **Gradient Checkpointing**: Reduce memory footprint
- **Dynamic Batching**: Adapt to graph sizes
- **Cache Management**: Efficient data loading

#### 2. Computational Efficiency
- **Sparse Operations**: Leverage graph sparsity
- **GPU Utilization**: CUDA optimization when available
- **Parallel Processing**: Multi-threaded data loading

#### 3. Numerical Stability
- **Layer Normalization**: Prevent activation explosion
- **Gradient Clipping**: Avoid gradient explosion
- **Learning Rate Scheduling**: Adaptive optimization

---

## Enhanced Capabilities and Visualization

### Comprehensive Visualization System

The framework now includes an extensive visualization suite for result analysis and model interpretation:

#### 1. GNN Result Visualizer
```python
class GNNVisualizer:
    """Comprehensive visualization for GNN model results"""
    
    def create_confusion_matrix(self, y_true, y_pred, class_names):
        """Generate interactive confusion matrix with heatmap"""
        
    def plot_training_curves(self, history):
        """Training/validation loss and accuracy curves"""
        
    def create_class_distribution(self, labels, predictions):
        """Distribution analysis with statistics"""
        
    def plot_per_class_metrics(self, metrics):
        """Precision, recall, F1-score bar charts"""
        
    def create_comprehensive_dashboard(self, results):
        """Complete analysis dashboard with all metrics"""
```

#### 2. Attention Visualization
```python
class AttentionVisualizer:
    """Visualize and analyze attention mechanisms"""
    
    def extract_attention_weights(self, model, data):
        """Extract attention weights from GAT layers"""
        
    def plot_attention_heatmap(self, attention_weights, coordinates):
        """Spatial attention pattern visualization"""
        
    def analyze_attention_distribution(self, attention_weights):
        """Statistical analysis of attention patterns"""
        
    def create_attention_flow_diagram(self, attention_weights):
        """Network flow diagram showing attention connections"""
```

### XAI (Explainable AI) Integration

#### 1. Model Explainability Tools
```python
class XAIExplainer:
    """Explainable AI tools for model interpretation"""
    
    def compute_grad_cam(self, model, data, target_class):
        """Gradient-based activation mapping"""
        
    def analyze_feature_importance(self, model, data):
        """Gene-level importance analysis"""
        
    def generate_counterfactual_explanations(self, model, data):
        """What-if analysis for predictions"""
        
    def create_lime_explanations(self, model, data, instance_idx):
        """Local interpretable model-agnostic explanations"""
```

#### 2. Biological Pathway Analysis
```python
class PathwayVisualizer:
    """Biological pathway analysis and visualization"""
    
    def analyze_cardiac_pathways(self, gene_importance, pathways):
        """Cardiac-specific pathway enrichment analysis"""
        
    def plot_pathway_networks(self, pathway_data):
        """Interactive pathway network visualization"""
        
    def create_gene_interaction_map(self, correlations, gene_names):
        """Gene-gene interaction network visualization"""
```

### Adaptive Graph Construction

#### 1. Dynamic Graph Learning
```python
class AdaptiveGraphConstructor:
    """Dynamic graph construction during training"""
    
    def adaptive_graph_update(self, data, attention_weights, epoch):
        """Update graph structure based on learned patterns"""
        
    def prune_edges_by_attention(self, edge_index, attention_weights):
        """Remove low-attention edges during training"""
        
    def add_edges_by_similarity(self, edge_index, similarity_matrix):
        """Add edges based on node similarity"""
        
    def biological_edge_filtering(self, edge_index, features, constraints):
        """Filter edges using biological knowledge"""
```

#### 2. Graph Evolution Tracking
- **Edge Statistics**: Track added/removed edges during training
- **Graph Metrics**: Monitor average degree, clustering coefficient
- **Attention Evolution**: Analyze how attention patterns change
- **Performance Correlation**: Link graph changes to model performance

### WandB Integration and Experiment Tracking

#### 1. Comprehensive Monitoring
```python
class WandBLogger:
    """Weights & Biases integration for experiment tracking"""
    
    def log_training_metrics(self, epoch, metrics):
        """Log training/validation metrics"""
        
    def log_visualizations(self, figures):
        """Upload plots and visualizations"""
        
    def log_model_artifacts(self, model, results):
        """Save model checkpoints and results"""
        
    def log_attention_analysis(self, attention_data):
        """Track attention pattern evolution"""
```

#### 2. Tracked Experiments
- **Model Architecture**: Hyperparameters and layer configurations
- **Training Dynamics**: Loss curves, learning rates, gradients
- **Attention Patterns**: Spatial attention evolution over epochs
- **Graph Evolution**: Edge statistics and structural changes
- **Performance Metrics**: Confusion matrices, per-class metrics
- **Biological Insights**: Pathway enrichment and gene importance

### Enhanced Training Pipeline

#### 1. Integrated Visualization Training
```python
class EnhancedCardiomyocyteTrainer:
    """Enhanced trainer with comprehensive monitoring"""
    
    def __init__(self, use_wandb=True, enable_visualization=True):
        self.visualizer = GNNVisualizer()
        self.attention_viz = AttentionVisualizer()
        self.xai_explainer = XAIExplainer()
        self.pathway_viz = PathwayVisualizer()
        self.adaptive_graph = AdaptiveGraphConstructor()
        
    def train_with_monitoring(self, model, data):
        """Training with real-time visualization and tracking"""
        
    def generate_comprehensive_report(self, results):
        """Create complete analysis report with all visualizations"""
```

#### 2. Real-time Analysis Features
- **Live Attention Tracking**: Monitor attention patterns during training
- **Dynamic Graph Visualization**: Real-time graph structure changes
- **Performance Dashboards**: Interactive training progress visualization
- **Biological Insight Generation**: Automatic pathway analysis
- **Model Interpretability**: Continuous XAI analysis

### Advanced Visualization Outputs

#### 1. Generated Visualizations
- **Confusion Matrix**: Interactive heatmap with class statistics
- **Training Curves**: Loss and accuracy progression with annotations
- **Attention Heatmaps**: Spatial attention pattern visualization
- **Feature Importance**: Gene-level contribution analysis
- **Pathway Networks**: Biological pathway interaction diagrams
- **Graph Evolution**: Dynamic graph structure changes over time
- **Performance Dashboards**: Comprehensive metric visualization

#### 2. Export Formats
- **Interactive HTML**: Plotly-based interactive visualizations
- **High-Resolution PNG**: Publication-quality static images
- **SVG Vector Graphics**: Scalable vector format for presentations
- **PDF Reports**: Complete analysis reports with all visualizations
- **JSON Data**: Exportable data for custom analysis

---

## Running Instructions

### Prerequisites
1. **Environment Setup**
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate.ps1  # Windows
source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

2. **Data Preparation**
```bash
# Ensure data directory exists
mkdir -p data/10x_visium
# Place your 10X Genomics data files in the data directory
```

### Training the Model

#### Enhanced Training with Visualization
```bash
# Run enhanced training with comprehensive monitoring
.\venv\Scripts\Activate.ps1
python enhanced_gnn_training.py

# With WandB experiment tracking (optional)
python enhanced_gnn_training.py --wandb --project "cardiomyocyte-gnn"
```

#### Training Configuration Options
```python
# Modify training parameters in enhanced_gnn_training.py
config = {
    # Model architecture
    'hidden_dim': 256,
    'num_gat_layers': 3,
    'num_gcn_layers': 2,
    'num_heads': 8,
    'dropout': 0.3,
    
    # Training settings
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    
    # Enhancement features
    'enable_visualization': True,
    'enable_attention_tracking': True,
    'adaptive_graph_construction': True,
    'use_wandb': True,
    
    # Visualization options
    'save_attention_maps': True,
    'generate_confusion_matrix': True,
    'create_pathway_analysis': True,
    'export_interactive_plots': True
}
```

#### Quick Start (Legacy)
```bash
# Basic training without enhancements
.\venv\Scripts\Activate.ps1
python train_cardiomyocyte_enhanced.py
```

#### Monitoring Training
```bash
# Enhanced training creates comprehensive outputs:
# - Real-time training curves and metrics
# - Attention heatmaps and evolution tracking
# - Interactive confusion matrices and class analysis
# - Biological pathway enrichment analysis
# - Graph structure evolution visualization

# Output directories:
# experiments_enhanced_cardiomyocyte/ - Training results and logs
# visualizations/ - Generated plots and interactive dashboards
# attention_analysis/ - Attention pattern analysis
# pathway_analysis/ - Biological pathway insights
# wandb_logs/ - WandB experiment tracking (if enabled)

# Monitor WandB dashboard (if enabled):
# Visit https://wandb.ai/your-username/cardiomyocyte-gnn
```

### Evaluation and Analysis

#### Model Evaluation
```python
# Comprehensive evaluation metrics
python -c "
from src.training.cardiomyocyte_trainer import CardiomyocyteTrainer
trainer = CardiomyocyteTrainer()
results = trainer.evaluate_saved_model('best_cardiomyocyte_model.pth')
print(f'Test Accuracy: {results[\"accuracy\"]:.4f}')
"
```

#### Result Analysis
```bash
# Generate detailed analysis reports
python analyze_gnn_results.py
```

### Advanced Usage

#### Hyperparameter Tuning
```python
# Grid search example
hyperparams = {
    'hidden_dim': [128, 256, 512],
    'dropout': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0005, 0.002]
}

# Run systematic hyperparameter optimization
python hyperparameter_search.py --config hyperparams.json
```

#### Custom Data Loading
```python
# Use your own spatial transcriptomics data
from src.data_processing.authentic_10x_processor import Authentic10XProcessor

processor = Authentic10XProcessor()
# Modify load_cached_data() to handle your data format
custom_data = processor.load_custom_data(your_data_path)
```

---

## File Structure

### Core Implementation
```
HybridGnnRnn/
├── src/
│   ├── models/
│   │   └── gnn_models/
│   │       └── cardiomyocyte_gnn.py      # Main GNN model
│   ├── data_processing/
│   │   └── authentic_10x_processor.py    # Data preprocessing
│   ├── training/
│   │   └── cardiomyocyte_trainer.py      # Training pipeline
│   └── utils/
│       └── visualization.py              # Result visualization
├── train_cardiomyocyte_enhanced.py       # Main training script
├── requirements.txt                      # Dependencies
└── GNN_MODEL_METHODOLOGY.md             # This document
```

### Output Directories
```
├── experiments_enhanced_cardiomyocyte/   # Training logs and metrics
├── cache/                                # Preprocessed data cache
├── models/                               # Saved model checkpoints
└── results/                              # Evaluation outputs
```

### Key Files Description

#### `src/models/gnn_models/cardiomyocyte_gnn.py`
- Contains the `AdvancedCardiomyocyteGNN` class
- Implements GAT+GCN hybrid architecture
- Includes attention mechanisms and skip connections

#### `src/data_processing/authentic_10x_processor.py`
- Handles 10X Genomics data loading
- Implements spatial graph construction
- Provides data caching and validation

#### `src/training/cardiomyocyte_trainer.py`
- Comprehensive training pipeline
- Includes evaluation metrics and visualization
- Handles model checkpointing and logging

#### `train_cardiomyocyte_enhanced.py`
- Main entry point for training
- Configurable parameters
- Clean interface with progress reporting

---

## Dependencies

### Core Requirements
```txt
torch>=2.0.0
torch-geometric>=2.3.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Spatial Analysis
```txt
scanpy>=1.9.0           # Single-cell analysis
anndata>=0.9.0          # Annotated data structures
scipy>=1.10.0           # Scientific computing
networkx>=3.1.0         # Graph algorithms
```

### Visualization and Logging
```txt
plotly>=5.15.0          # Interactive plots
wandb>=0.15.0           # Experiment tracking (optional)
tensorboard>=2.13.0     # TensorBoard logging (optional)
tqdm>=4.65.0            # Progress bars
```

### Development Tools
```txt
pytest>=7.4.0          # Testing framework
black>=23.0.0           # Code formatting
flake8>=6.0.0          # Code linting
jupyter>=1.0.0         # Notebook environment
```

### Installation Commands
```bash
# Essential dependencies
pip install torch torch-geometric scikit-learn pandas numpy matplotlib seaborn

# Spatial transcriptomics
pip install scanpy anndata scipy networkx

# Visualization
pip install plotly tqdm

# Optional: Experiment tracking
pip install wandb tensorboard

# Development tools
pip install pytest black flake8 jupyter
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Compatibility
```bash
# Check CUDA version compatibility
python -c "import torch; print(torch.cuda.is_available())"
# Install appropriate PyTorch version for your CUDA
```

#### 2. Memory Issues
```python
# Reduce batch size or model dimensions
config['hidden_dim'] = 128  # Instead of 256
config['batch_size'] = 16   # Instead of 32
```

#### 3. Data Loading Errors
```bash
# Clear cache and reload
rm -rf cache/
python train_cardiomyocyte_enhanced.py
```

#### 4. Graph Construction Issues
```python
# Adjust k-nearest neighbors parameter
processor.build_spatial_graph(coordinates, k=4)  # Reduce from 6
```

### Performance Tuning

#### GPU Optimization
```python
# Enable mixed precision training
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()
# Use in training loop for memory efficiency
```

#### Memory Management
```python
# Gradient checkpointing for large models
model.gradient_checkpointing_enable()
# Clear cache periodically
torch.cuda.empty_cache()
```

## References

1. **Graph Neural Networks**: Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks"
2. **Graph Attention Networks**: Veličković et al. (2018). "Graph Attention Networks"
3. **Spatial Transcriptomics**: Ståhl et al. (2016). "Visualization and analysis of gene expression in tissue sections by spatial transcriptomics"
4. **10X Genomics Visium**: Stickels et al. (2021). "Highly sensitive spatial transcriptomics at near-cellular resolution with Visium"
5. **Cardiomyocyte Development**: Paige et al. (2010). "A temporal chromatin signature in human embryonic stem cells identifies regulators of cardiac development"
6. **Dataset Source**: Kuppe, C., Ramirez Flores, R. O., Li, Z., Hayat, S., Levinson, R. T., Liao, X., Hannani, M. T., Tanevski, J., Wünnemann, F., Nagai, J. S., Halder, M., Schumacher, D., Menzel, S., Schäfer, G., Hoeft, K., Cheng, M., Ziegler, S., Zhang, X., Peisker, F., … Kramann, R. (2022). Spatial multi-omic map of human myocardial infarction. *Nature*, 608, 766–777. https://doi.org/10.1038/s41586-022-05060-x

---

*Last Updated: September 20, 2025*
*Version: 1.0*
*Authors: Tumo Kgabeng, lulu Wang, Harry Ngwangwa, Thanyani Pandelani*