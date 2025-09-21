
# Enhanced Cardiomyocyte GNN - Performance Report
Generated: 2025-09-21 22:50:11

## Model Configuration
- Architecture: AdvancedCardiomyocyteGNN
- Hidden Dimensions: 128
- Dropout: 0.4
- Learning Rate: 0.001

## Performance Metrics
- Test Accuracy: 0.6303 (63.03%)
- Best Validation Accuracy: 0.6546 (65.46%)

## Per-Class Performance
- Atrial Cardiomyocytes:
  - F1-Score: 0.735
  - Precision: 0.656
  - Recall: 0.836
  - Support: 146 cells

- Ventricular Cardiomyocytes:
  - F1-Score: 0.830
  - Precision: 0.765
  - Recall: 0.907
  - Support: 129 cells

- Pacemaker Cells:
  - F1-Score: 0.454
  - Precision: 0.629
  - Recall: 0.355
  - Support: 186 cells

- Conduction System Cells:
  - F1-Score: 0.500
  - Precision: 0.670
  - Recall: 0.399
  - Support: 188 cells

- Immature Cardiomyocytes:
  - F1-Score: 0.629
  - Precision: 0.480
  - Recall: 0.913
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
Experiment logged to: enhanced-cardiomyocyte-gnn/enhanced_gnn_20250921_224823
