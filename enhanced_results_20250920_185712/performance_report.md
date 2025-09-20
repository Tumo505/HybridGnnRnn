
# Enhanced Cardiomyocyte GNN - Performance Report
Generated: 2025-09-20 18:59:30

## Model Configuration
- Architecture: AdvancedCardiomyocyteGNN
- Hidden Dimensions: 128
- Dropout: 0.4
- Learning Rate: 0.001

## Performance Metrics
- Test Accuracy: 0.6769 (67.69%)
- Best Validation Accuracy: 0.6720 (67.20%)

## Per-Class Performance
- subtype_0:
  - F1-Score: 0.723
  - Precision: 0.773
  - Recall: 0.678
  - Support: 146 cells

- subtype_1:
  - F1-Score: 0.874
  - Precision: 0.888
  - Recall: 0.860
  - Support: 129 cells

- subtype_2:
  - F1-Score: 0.565
  - Precision: 0.561
  - Recall: 0.570
  - Support: 186 cells

- subtype_3:
  - F1-Score: 0.688
  - Precision: 0.592
  - Recall: 0.819
  - Support: 188 cells

- subtype_4:
  - F1-Score: 0.510
  - Precision: 0.780
  - Recall: 0.379
  - Support: 103 cells


## Dataset Information
- Total Samples: 4990
- Number of Classes: 5
- Feature Dimensions: 16986

## Generated Visualizations
- confusion_matrix.png - Model performance analysis
- class_distribution.png - Dataset distribution analysis
- training_curves.png - Training progression
- performance_report.md - This report

## Wandb Integration
Experiment logged to: enhanced-cardiomyocyte-gnn/enhanced_training_20250920_185708
