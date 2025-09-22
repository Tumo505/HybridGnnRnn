# Contributing to Hybrid GNN-RNN Model

Thank you for your interest in contributing to this multimodal deep learning project! This guide will help you get started with contributing effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Code Organization](#code-organization)
4. [Contribution Guidelines](#contribution-guidelines)
5. [Testing Requirements](#testing-requirements)
6. [Model Development](#model-development)
7. [Data Processing](#data-processing)
8. [Documentation Standards](#documentation-standards)
9. [Pull Request Process](#pull-request-process)
10. [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- **Python**: 3.8+ (3.10 recommended)
- **CUDA**: 11.8+ for GPU acceleration
- **Git**: Latest version
- **Git LFS**: For handling large model files

### Quick Setup

1. **Fork and Clone**:

   ```bash
   git clone https://github.com/yourusername/HybridGnnRnn.git
   cd HybridGnnRnn
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   python -c "import torch_geometric; print('PyG installed successfully')"
   ```

## Development Environment

### Recommended Setup

- **IDE**: VS Code with Python, Jupyter, and GitLens extensions
- **GPU**: NVIDIA RTX 3060 or better (8GB+ VRAM)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ available space

### Environment Configuration

Create a development environment:

```bash
conda create -n hybrid-gnn-rnn python=3.10
conda activate hybrid-gnn-rnn
pip install -r requirements.txt
```

### Code Formatting

We use standardized formatting tools:

```bash
# Install formatting tools
pip install black isort flake8 mypy

# Format code
black src/
isort src/
flake8 src/
mypy src/
```

## Code Organization

### Project Structure

```text
src/
├── models/
│   ├── gnn/              # Graph Neural Network implementations
│   ├── rnn/              # Recurrent Neural Network implementations
│   ├── hybrid/           # Fusion strategies and hybrid models
│   └── base.py           # Base model classes and interfaces
├── data/
│   ├── loaders/          # Data loading utilities
│   ├── preprocessing/    # Data preprocessing pipelines
│   └── augmentation/     # Data augmentation techniques
├── training/
│   ├── trainers/         # Training loop implementations
│   ├── losses/           # Custom loss functions
│   └── optimizers/       # Optimization utilities
├── evaluation/
│   ├── metrics/          # Evaluation metrics
│   ├── visualization/    # Plotting and visualization
│   └── statistical/      # Statistical analysis tools
└── xai/
    ├── shap/             # SHAP explanations
    ├── attention/        # Attention visualizations
    └── biological/       # Biological interpretation
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Variables**: `snake_case`

## Contribution Guidelines

### Types of Contributions

1. **Bug Fixes**: Address issues in existing code
2. **New Features**: Add new model architectures or analysis tools
3. **Documentation**: Improve code documentation and guides
4. **Performance**: Optimize computational efficiency
5. **Testing**: Expand test coverage
6. **Examples**: Add tutorial notebooks and examples

### Coding Standards

#### Python Style

Follow PEP 8 with these specifications:

```python
# Good: Clear function documentation
def train_hybrid_model(
    spatial_data: torch.Tensor,
    temporal_data: torch.Tensor,
    fusion_strategy: str = "concatenation",
    epochs: int = 100
) -> Dict[str, float]:
    """Train hybrid GNN-RNN model with specified fusion strategy.
    
    Args:
        spatial_data: Spatial transcriptomics features [N, genes]
        temporal_data: Temporal gene expression [N, timepoints, genes]
        fusion_strategy: Method for combining modalities
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
    """
    pass
```

#### Type Hints

Use comprehensive type hints:

```python
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np

def process_spatial_data(
    adata: "AnnData",
    n_top_genes: int = 2000,
    normalize: bool = True
) -> Tuple[torch.Tensor, List[str]]:
    """Process spatial transcriptomics data."""
    pass
```

#### Error Handling

Implement robust error handling:

```python
def load_model_checkpoint(checkpoint_path: str) -> torch.nn.Module:
    """Load model from checkpoint with validation."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path)
        model = HybridGnnRnn(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
```

## Testing Requirements

### Test Structure

Create comprehensive tests for all contributions:

```python
# tests/test_models.py
import pytest
import torch
from src.models.hybrid import HybridGnnRnn

class TestHybridModel:
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        spatial_data = torch.randn(32, 2000)  # 32 samples, 2000 genes
        temporal_data = torch.randn(32, 10, 2000)  # 32 samples, 10 timepoints
        return spatial_data, temporal_data
    
    def test_forward_pass(self, sample_data):
        """Test model forward pass."""
        spatial_data, temporal_data = sample_data
        model = HybridGnnRnn(fusion_strategy="concatenation")
        
        output = model(spatial_data, temporal_data)
        
        assert output.shape == (32, 4)  # 4 classes
        assert not torch.isnan(output).any()
    
    def test_fusion_strategies(self, sample_data):
        """Test all fusion strategies work."""
        spatial_data, temporal_data = sample_data
        
        for strategy in ["concatenation", "attention", "ensemble"]:
            model = HybridGnnRnn(fusion_strategy=strategy)
            output = model(spatial_data, temporal_data)
            assert output.shape == (32, 4)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_models.py -v
```

### Statistical Testing

Include statistical validation:

```python
def test_model_performance_significance():
    """Test that model improvements are statistically significant."""
    from scipy.stats import ttest_rel
    
    baseline_scores = np.array([0.65, 0.63, 0.67, 0.64, 0.66])
    improved_scores = np.array([0.96, 0.97, 0.95, 0.98, 0.96])
    
    t_stat, p_value = ttest_rel(baseline_scores, improved_scores)
    assert p_value < 0.05, "Model improvement not statistically significant"
```

## Model Development

### Adding New Models

When contributing new model architectures:

1. **Inherit from Base Classes**:

   ```python
   from src.models.base import BaseGNN, BaseRNN
   
   class CustomGNN(BaseGNN):
       def __init__(self, input_dim: int, hidden_dim: int):
           super().__init__()
           # Your implementation
   ```

2. **Register Model**:

   ```python
   # In src/models/__init__.py
   from .custom_gnn import CustomGNN
   
   MODEL_REGISTRY = {
       'custom_gnn': CustomGNN,
       # ... other models
   }
   ```

3. **Add Configuration**:

   ```python
   # In configs/models/custom_gnn.yaml
   model:
     name: custom_gnn
     params:
       input_dim: 2000
       hidden_dim: 512
   ```

### Fusion Strategy Development

For new fusion methods:

```python
class CustomFusion(nn.Module):
    """Custom fusion strategy for multimodal data."""
    
    def __init__(self, gnn_dim: int, rnn_dim: int, output_dim: int):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            # Your fusion implementation
        )
    
    def forward(self, gnn_features: torch.Tensor, rnn_features: torch.Tensor) -> torch.Tensor:
        """Fuse GNN and RNN features."""
        # Implementation
        pass
```

## Data Processing

### Adding New Datasets

1. **Create Data Loader**:

   ```python
   class CustomDataLoader:
       def __init__(self, data_path: str, split: str = "train"):
           self.data_path = data_path
           self.split = split
       
       def load_spatial_data(self) -> torch.Tensor:
           """Load spatial transcriptomics data."""
           pass
       
       def load_temporal_data(self) -> torch.Tensor:
           """Load temporal gene expression data."""
           pass
   ```

2. **Add Preprocessing Pipeline**:

   ```python
   def preprocess_custom_data(raw_data: np.ndarray) -> torch.Tensor:
       """Preprocess custom dataset."""
       # Quality control
       # Normalization
       # Feature selection
       return processed_data
   ```

### Data Validation

Include data validation for new datasets:

```python
def validate_spatial_data(data: torch.Tensor) -> bool:
    """Validate spatial transcriptomics data format."""
    assert data.ndim == 2, "Spatial data must be 2D"
    assert data.shape[1] > 0, "Must have features"
    assert not torch.isnan(data).any(), "No NaN values allowed"
    return True
```

## Documentation Standards

### Code Documentation

Use comprehensive docstrings:

```python
def calculate_fusion_attention(
    gnn_features: torch.Tensor,
    rnn_features: torch.Tensor,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate attention weights for multimodal fusion.
    
    This function computes attention weights to combine GNN and RNN features
    based on their relevance for the prediction task.
    
    Args:
        gnn_features: Spatial graph features [batch_size, gnn_dim]
        rnn_features: Temporal sequence features [batch_size, rnn_dim]
        temperature: Temperature parameter for attention softmax
    
    Returns:
        Tuple containing:
            - Fused features [batch_size, output_dim]
            - Attention weights [batch_size, 2]
    
    Raises:
        ValueError: If input tensors have incompatible shapes
        
    Example:
        >>> gnn_feat = torch.randn(32, 128)
        >>> rnn_feat = torch.randn(32, 512)
        >>> fused, weights = calculate_fusion_attention(gnn_feat, rnn_feat)
        >>> print(f"Fused shape: {fused.shape}")
        Fused shape: torch.Size([32, 640])
    """
```

### README Updates

When adding new features, update relevant documentation:

- Main README.md for user-facing changes
- TECHNICAL_DOCUMENTATION.md for implementation details
- API documentation for new functions

### Jupyter Notebooks

For tutorial notebooks:

1. **Clear Structure**: Use markdown headers for organization
2. **Explanatory Text**: Explain each step thoroughly
3. **Reproducible**: Include random seeds and version info
4. **Clean Output**: Clear outputs before committing

## Pull Request Process

### Before Submitting

1. **Code Quality**:

   ```bash
   black src/
   isort src/
   flake8 src/
   mypy src/
   ```

2. **Run Tests**:

   ```bash
   pytest tests/ -v
   ```

3. **Update Documentation**:
   - Add docstrings to new functions
   - Update relevant README sections
   - Include usage examples

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new functionality
- [ ] Updated existing tests

## Documentation
- [ ] Updated docstrings
- [ ] Updated README if needed
- [ ] Added usage examples

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] No merge conflicts
- [ ] Descriptive commit messages
```

### Review Process

1. **Automated Checks**: CI/CD pipeline validates code quality
2. **Peer Review**: At least one maintainer review required
3. **Testing**: All tests must pass
4. **Documentation**: Documentation must be updated for new features

## Community Guidelines

### Communication

- **Be Respectful**: Treat all contributors with respect
- **Be Constructive**: Provide helpful feedback and suggestions
- **Be Patient**: Allow time for response and review
- **Ask Questions**: Don't hesitate to ask for clarification

### Issue Reporting

When reporting bugs:

1. **Search Existing Issues**: Check if issue already exists
2. **Provide Details**: Include system info, error messages, steps to reproduce
3. **Minimal Example**: Provide minimal code to reproduce the issue
4. **Expected Behavior**: Describe what you expected to happen

### Feature Requests

For new features:

1. **Clear Description**: Explain the feature and its benefits
2. **Use Cases**: Provide specific use cases
3. **Implementation Ideas**: Suggest potential implementation approaches
4. **Breaking Changes**: Note any potential breaking changes

## Development Tips

### Performance Optimization

- Profile code with `cProfile` for bottlenecks
- Use `torch.compile()` for model optimization (PyTorch 2.0+)
- Consider mixed precision training with `autocast`
- Monitor GPU memory usage with `torch.cuda.memory_summary()`

### Debugging

- Use `pdb` for interactive debugging
- Add logging with proper levels (DEBUG, INFO, WARNING, ERROR)
- Use `torch.autograd.set_detect_anomaly(True)` for gradient debugging
- Visualize intermediate outputs for model debugging

### Reproducibility

- Set random seeds for all random number generators
- Log all hyperparameters and configuration
- Version control data processing steps
- Use deterministic algorithms when possible

---

Thank you for contributing to the Hybrid GNN-RNN Model project! Your contributions help advance multimodal deep learning for biological data analysis.

For questions or additional guidance, please:

- Open an issue on GitHub
- Join our discussion forums
- Contact the maintainers directly

**Happy coding!** 🚀🧬🤖