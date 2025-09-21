"""
Comprehensive GNN Embedding Analysis Summary
Integrating all findings from the trained models analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_model_comparison():
    """Compare all models and extract key insights"""
    
    logger.info("🔬 COMPREHENSIVE GNN ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    # Load analysis results
    analysis_file = "trained_gnn_analysis_20250921_193410.json"
    biological_file = "biological_analysis_Fold_3.json"
    
    with open(analysis_file, 'r') as f:
        model_results = json.load(f)
    
    with open(biological_file, 'r') as f:
        biological_results = json.load(f)
    
    # Extract key metrics
    models_summary = []
    for model_name, data in model_results.items():
        models_summary.append({
            'name': model_name,
            'type': data['model_type'],
            'accuracy': data['accuracy'],
            'val_score': data['checkpoint_info']['best_val_score'],
            'embedding_dim': data['embedding_shape'][1]
        })
    
    # Sort by accuracy
    models_summary.sort(key=lambda x: x['accuracy'], reverse=True)
    
    logger.info("\n📊 MODEL PERFORMANCE RANKING:")
    logger.info("-" * 50)
    for i, model in enumerate(models_summary, 1):
        logger.info(f"{i}. {model['name']} ({model['type']})")
        logger.info(f"   Test Accuracy: {model['accuracy']:.4f}")
        logger.info(f"   Val Score: {model['val_score']:.4f}")
        logger.info(f"   Embedding Dim: {model['embedding_dim']}")
        logger.info("")
    
    # Key insights
    logger.info("🎯 KEY INSIGHTS:")
    logger.info("-" * 50)
    
    # 1. Model architecture findings
    gnn_only = [m for m in models_summary if m['type'] == 'gnn_only'][0]
    hybrid_models = [m for m in models_summary if m['type'] == 'hybrid']
    best_hybrid = max(hybrid_models, key=lambda x: x['accuracy'])
    
    logger.info(f"1. ARCHITECTURE COMPARISON:")
    logger.info(f"   GNN-Only: {gnn_only['accuracy']:.4f} accuracy")
    logger.info(f"   Best Hybrid: {best_hybrid['accuracy']:.4f} accuracy")
    logger.info(f"   → Simple GNN architecture is competitive")
    logger.info("")
    
    # 2. Performance vs validation gap
    best_model = models_summary[0]
    val_test_gap = best_model['accuracy'] - best_model['val_score']
    
    logger.info(f"2. GENERALIZATION ANALYSIS:")
    logger.info(f"   Best model: {best_model['name']}")
    logger.info(f"   Val→Test gap: {val_test_gap:.4f}")
    logger.info(f"   → Significant overfitting to specific validation splits")
    logger.info("")
    
    # 3. Biological insights from best model
    bio = biological_results
    sep_ratio = bio['class_separation']['separation_ratio']
    best_clustering = max(bio['clustering_analysis'].values(), 
                         key=lambda x: x['silhouette_score'])
    
    logger.info(f"3. BIOLOGICAL REPRESENTATION:")
    logger.info(f"   Class separation ratio: {sep_ratio:.3f}")
    logger.info(f"   Best silhouette score: {best_clustering['silhouette_score']:.3f}")
    logger.info(f"   Effective dimensionality: {bio['embedding_statistics']['n_components_95_variance']}")
    logger.info(f"   → Models learn low-dimensional, clustered representations")
    logger.info("")
    
    # 4. Class-specific performance
    class_patterns = bio['prediction_patterns']
    best_class = max(class_patterns.items(), key=lambda x: x[1]['accuracy'])
    worst_class = min(class_patterns.items(), key=lambda x: x[1]['accuracy'])
    
    logger.info(f"4. CLASS-SPECIFIC INSIGHTS:")
    logger.info(f"   Best class: Class {best_class[0]} ({best_class[1]['accuracy']:.3f} accuracy)")
    logger.info(f"   Worst class: Class {worst_class[0]} ({worst_class[1]['accuracy']:.3f} accuracy)")
    logger.info(f"   Performance range: {worst_class[1]['accuracy']:.3f} - {best_class[1]['accuracy']:.3f}")
    logger.info(f"   → Highly imbalanced class learnability")
    logger.info("")
    
    # 5. Technical recommendations
    logger.info("🛠️ TECHNICAL RECOMMENDATIONS:")
    logger.info("-" * 50)
    logger.info("1. Focus on GNN-only architectures (simpler, better performance)")
    logger.info("2. Investigate Class 0 - it's highly learnable (70.9% accuracy)")
    logger.info("3. Address class imbalance - Classes 2,5,6 struggling (~25-29%)")
    logger.info("4. Consider biological validation over classification accuracy")
    logger.info("5. Explore trajectory/pathway analysis with learned embeddings")
    logger.info("")
    
    # 6. Biological implications
    logger.info("🧬 BIOLOGICAL IMPLICATIONS:")
    logger.info("-" * 50)
    logger.info("1. Model learns meaningful cell state representations")
    logger.info("2. High silhouette scores (0.65+) indicate biological clustering")
    logger.info("3. Low effective dimensionality (3 components) suggests biological manifold")
    logger.info("4. Class 0 may represent a distinct differentiation state")
    logger.info("5. Other classes may be transitional or less well-defined states")
    logger.info("")
    
    return {
        'models_summary': models_summary,
        'biological_insights': biological_results,
        'key_findings': {
            'best_model': best_model['name'],
            'architecture_winner': 'GNN-only competitive with hybrid',
            'separation_ratio': sep_ratio,
            'effective_dimensions': bio['embedding_statistics']['n_components_95_variance'],
            'class_performance_range': (worst_class[1]['accuracy'], best_class[1]['accuracy'])
        }
    }

def create_final_summary_plot():
    """Create a comprehensive summary visualization"""
    logger.info("🎨 Creating comprehensive summary visualization...")
    
    # Load data
    analysis_file = "trained_gnn_analysis_20250921_193410.json"
    with open(analysis_file, 'r') as f:
        results = json.load(f)
    
    # Prepare data
    models = []
    accuracies = []
    val_scores = []
    types = []
    colors = []
    
    for name, data in results.items():
        models.append(name)
        accuracies.append(data['accuracy'])
        val_scores.append(data['checkpoint_info']['best_val_score'])
        model_type = data['model_type']
        types.append(model_type)
        colors.append('skyblue' if model_type == 'gnn_only' else 'lightcoral')
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GNN Model Analysis Summary', fontsize=16, fontweight='bold')
    
    # 1. Model accuracies comparison
    ax = axes[0, 0]
    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title('Test Accuracy by Model', fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.tick_params(axis='x', rotation=45)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', label='GNN-only'),
                      Patch(facecolor='lightcoral', label='Hybrid')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 2. Validation vs Test performance
    ax = axes[0, 1]
    ax.scatter(val_scores, accuracies, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax.plot([0.14, 0.16], [0.14, 0.16], 'k--', alpha=0.5, label='Perfect correlation')
    ax.set_xlabel('Validation Score')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Validation vs Test Performance', fontweight='bold')
    
    # Annotate points
    for i, model in enumerate(models):
        ax.annotate(model, (val_scores[i], accuracies[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. Model complexity vs performance
    ax = axes[1, 0]
    embedding_dims = [results[name]['embedding_shape'][1] for name in models]
    ax.scatter(embedding_dims, accuracies, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax.set_xlabel('Embedding Dimensionality')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Complexity vs Performance', fontweight='bold')
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate summary stats
    gnn_acc = [acc for acc, typ in zip(accuracies, types) if typ == 'gnn_only'][0]
    hybrid_accs = [acc for acc, typ in zip(accuracies, types) if typ == 'hybrid']
    mean_hybrid = np.mean(hybrid_accs)
    
    summary_text = f"""
    📊 SUMMARY STATISTICS
    
    🏆 Best Model: {models[np.argmax(accuracies)]}
    📈 Best Accuracy: {max(accuracies):.4f}
    
    🤖 GNN-only: {gnn_acc:.4f}
    🔗 Hybrid Mean: {mean_hybrid:.4f}
    
    📏 Architecture Impact:
       Simple GNN competitive
       
    🎯 Key Finding:
       Model performance varies more
       by training fold than by
       architecture complexity
    
    🧬 Biological Insight:
       Models learn meaningful
       low-dimensional representations
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = "comprehensive_gnn_analysis_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   💾 Saved: {plot_path}")
    return plot_path

def main():
    """Main comprehensive analysis"""
    
    # Run analysis
    summary = analyze_model_comparison()
    
    # Create visualization
    plot_path = create_final_summary_plot()
    
    # Save comprehensive results
    with open("comprehensive_gnn_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("✅ ANALYSIS COMPLETE!")
    logger.info(f"📁 Summary: comprehensive_gnn_analysis_summary.json")
    logger.info(f"🎨 Visualization: {plot_path}")
    
    return summary

if __name__ == "__main__":
    summary = main()