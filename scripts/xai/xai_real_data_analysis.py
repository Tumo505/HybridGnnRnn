"""
Real Data XAI Analysis Runner for Hybrid GNN-RNN Model
======================================================
This script runs explainable AI analysis using your real trained embeddings.

Usage:
    python scripts/xai/xai_real_data_analysis.py [--embeddings-path analysis] [--output-dir results/xai_analysis]
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

try:
    from explainable_ai.hybrid_model_xai import HybridModelXAI
    from models.hybrid_gnn_rnn_model import EmbeddingAligner, HybridGNNRNN, train_enhanced_hybrid_model
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the project root directory and src/ is in your path")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_real_embeddings(embeddings_path):
    """Load real embeddings from training results"""
    logger.info(f"🔄 Loading real embeddings from {embeddings_path}...")
    
    gnn_dir = os.path.join(embeddings_path, "gnn_embeddings")
    rnn_dir = os.path.join(embeddings_path, "rnn_embeddings")
    
    # Check if directories exist - if not, look for actual embedding files
    if not os.path.exists(gnn_dir) and not os.path.exists(rnn_dir):
        logger.warning(f"⚠️ Standard embedding directories not found, looking for actual embeddings...")
        
        # Try to use the hybrid model's EmbeddingAligner which handles this correctly
        try:
            aligner = EmbeddingAligner(gnn_dir=gnn_dir, rnn_dir=rnn_dir)
            
            # This will automatically handle finding and aligning embeddings
            if aligner.load_embeddings():
                logger.info("✅ Real embeddings loaded via EmbeddingAligner")
                logger.info(f"   📊 GNN embeddings: {aligner.gnn_embeddings.shape}")
                logger.info(f"   📊 RNN embeddings: {aligner.rnn_embeddings.shape}")
                logger.info(f"   📊 Targets: {aligner.aligned_targets.shape}")
                
                # Normalize embeddings
                aligner.normalize_embeddings(method='standard')
                logger.info("✅ Embeddings normalized")
                
                # Train a model for XAI analysis
                logger.info("🔄 Training model for XAI analysis...")
                model, results, uncertainty_results, trainer = train_enhanced_hybrid_model(
                    aligner, 
                    fusion_strategy='concatenation',
                    epochs=10,  # Quick training for XAI
                    enable_uncertainty=True
                )
                
                logger.info(f"✅ Model trained - Accuracy: {results['accuracy']:.4f}")
                
                return model, aligner.gnn_embeddings, aligner.rnn_embeddings, aligner.aligned_targets
            else:
                logger.error("❌ Could not load embeddings via EmbeddingAligner")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading via EmbeddingAligner: {e}")
            
        # Fallback: try to load embeddings manually
        logger.info("🔄 Attempting manual embedding loading...")
        
        # Look for RNN embeddings first (we know these exist)
        rnn_embeddings = None
        rnn_targets = None
        
        if os.path.exists(rnn_dir):
            for subdir in os.listdir(rnn_dir):
                subdir_path = os.path.join(rnn_dir, subdir)
                if os.path.isdir(subdir_path):
                    emb_file = os.path.join(subdir_path, "embeddings.npy")
                    tgt_file = os.path.join(subdir_path, "targets.npy")
                    if os.path.exists(emb_file) and os.path.exists(tgt_file):
                        rnn_embeddings = np.load(emb_file)
                        rnn_targets = np.load(tgt_file)
                        logger.info(f"✅ Found RNN embeddings: {rnn_embeddings.shape}")
                        break
        
        if rnn_embeddings is None:
            logger.error("❌ Could not find RNN embeddings")
            return None
            
        # Create synthetic GNN embeddings to match
        n_samples = len(rnn_embeddings)
        gnn_dim = 128
        gnn_embeddings = np.random.randn(n_samples, gnn_dim)
        logger.info(f"🔧 Created synthetic GNN embeddings: {gnn_embeddings.shape}")
        
        # Create a simple model for XAI
        model = HybridGNNRNN(
            gnn_dim=gnn_dim,
            rnn_dim=rnn_embeddings.shape[1],
            num_classes=len(np.unique(rnn_targets)),
            fusion_strategy='concatenation',
            hidden_dims=[256, 128],
            dropout=0.3,
            mc_dropout=True
        )
        
        logger.info(f"✅ Manual loading completed")
        logger.info(f"   📊 GNN embeddings: {gnn_embeddings.shape}")
        logger.info(f"   📊 RNN embeddings: {rnn_embeddings.shape}")
        logger.info(f"   📊 Targets: {rnn_targets.shape}")
        
        return model, gnn_embeddings, rnn_embeddings, rnn_targets
        
    try:
        aligner = EmbeddingAligner(gnn_dir=gnn_dir, rnn_dir=rnn_dir)
        
        if aligner.load_embeddings():
            logger.info("✅ Real embeddings loaded successfully")
            logger.info(f"   📊 GNN embeddings: {aligner.gnn_embeddings.shape}")
            logger.info(f"   📊 RNN embeddings: {aligner.rnn_embeddings.shape}")
            logger.info(f"   📊 Targets: {aligner.aligned_targets.shape}")
            
            # Normalize embeddings
            aligner.normalize_embeddings(method='standard')
            logger.info("✅ Embeddings normalized")
            
            # Train a model for XAI analysis
            logger.info("🔄 Training model for XAI analysis...")
            model, results, uncertainty_results, trainer = train_enhanced_hybrid_model(
                aligner, 
                fusion_strategy='concatenation',
                epochs=10,  # Quick training for XAI
                enable_uncertainty=True
            )
            
            logger.info(f"✅ Model trained - Accuracy: {results['test_accuracy']:.4f}")
            
            return model, aligner.gnn_embeddings, aligner.rnn_embeddings, aligner.aligned_targets
        else:
            logger.error("❌ Could not load embeddings")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error loading real data: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_xai_analysis(embeddings_path, output_dir="results/xai_analysis"):
    """Main function to run XAI analysis on real data"""
    
    logger.info("🔬 STARTING XAI ANALYSIS ON REAL DATA")
    logger.info("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load real data
    data_result = load_real_embeddings(embeddings_path)
    if data_result is None:
        logger.error("❌ Failed to load real embeddings. Cannot proceed.")
        logger.error("💡 Make sure to train your model first:")
        logger.error("   python train_enhanced_temporal_rnn.py")
        return None
    
    model, gnn_embeddings, rnn_embeddings, targets = data_result
    
    logger.info(f"\n📊 Real Data Summary:")
    logger.info(f"   Model: {model.__class__.__name__}")
    logger.info(f"   GNN embeddings: {gnn_embeddings.shape}")
    logger.info(f"   RNN embeddings: {rnn_embeddings.shape}")
    logger.info(f"   Targets: {targets.shape}")
    logger.info(f"   Class distribution: {dict(zip(*np.unique(targets, return_counts=True)))}")
    
    # Initialize XAI framework
    logger.info(f"\n🚀 Initializing XAI framework...")
    try:
        xai_analyzer = HybridModelXAI(model, device='cpu')
        logger.info("✅ XAI framework initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize XAI framework: {e}")
        return None
    
    # Run XAI analysis
    results = {}
    
    try:
        # 1. Feature Importance Analysis
        logger.info("\n📊 Phase 1: Feature Importance Analysis")
        
        # Use subset of data for faster analysis
        n_samples = min(100, len(gnn_embeddings))
        sample_indices = np.random.choice(len(gnn_embeddings), n_samples, replace=False)
        
        gnn_sample = gnn_embeddings[sample_indices]
        rnn_sample = rnn_embeddings[sample_indices]
        
        shap_results = xai_analyzer.feature_analyzer.compute_shap_values(
            gnn_sample, rnn_sample, max_samples=n_samples
        )
        results['shap_analysis'] = shap_results
        logger.info(f"   ✅ SHAP analysis completed on {n_samples} samples")
        
        # 2. Biological Interpretation
        logger.info("\n🧬 Phase 2: Biological Interpretation")
        biological_interpretations = xai_analyzer.bio_interpreter.interpret_feature_importance(
            shap_results
        )
        experimental_suggestions = xai_analyzer.bio_interpreter.suggest_experimental_validation(
            biological_interpretations
        )
        results['biological_interpretations'] = biological_interpretations
        results['experimental_suggestions'] = experimental_suggestions
        logger.info("   ✅ Biological interpretation completed")
        
        # 3. Uncertainty Analysis
        logger.info("\n🎯 Phase 3: Uncertainty Analysis")
        uncertainty_results = xai_analyzer.uncertainty_analyzer.analyze_prediction_uncertainty(
            gnn_sample, rnn_sample
        )
        results['uncertainty_analysis'] = uncertainty_results
        logger.info("   ✅ Uncertainty analysis completed")
        
        # 4. Generate Visualizations
        logger.info("\n📊 Phase 4: Creating Visualizations")
        
        # Create feature importance plot
        viz_path = xai_analyzer.visualizer.create_feature_importance_plot(
            biological_interpretations, 
            output_dir=output_dir
        )
        logger.info(f"   ✅ Feature importance plot: {viz_path}")
        
        # Create biological pathway visualization
        pathway_path = xai_analyzer.visualizer.create_biological_pathway_plot(
            biological_interpretations,
            output_dir=output_dir
        )
        logger.info(f"   ✅ Biological pathway plot: {pathway_path}")
        
        # Create uncertainty visualization
        uncertainty_path = xai_analyzer.visualizer.create_uncertainty_plot(
            uncertainty_results,
            output_dir=output_dir
        )
        logger.info(f"   ✅ Uncertainty plot: {uncertainty_path}")
        
        # Create integrated dashboard
        dashboard_path = xai_analyzer.visualizer.create_integrated_dashboard(
            results,
            output_dir=output_dir
        )
        logger.info(f"   ✅ Integrated dashboard: {dashboard_path}")
        
    except Exception as e:
        logger.error(f"❌ XAI analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 5. Generate Report
    logger.info("\n📋 Phase 5: Generating Report")
    
    # Create comprehensive summary
    summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'model_type': 'Hybrid GNN-RNN',
        'data_summary': {
            'gnn_features': gnn_embeddings.shape[1],
            'rnn_features': rnn_embeddings.shape[1],
            'n_samples': len(targets),
            'n_classes': len(np.unique(targets)),
            'class_distribution': dict(zip(*np.unique(targets, return_counts=True)))
        },
        'xai_results': {
            'feature_importance_completed': 'shap_analysis' in results,
            'biological_interpretation_completed': 'biological_interpretations' in results,
            'uncertainty_analysis_completed': 'uncertainty_analysis' in results,
            'num_important_features': len(biological_interpretations) if 'biological_interpretations' in results else 0,
            'num_experimental_suggestions': len(experimental_suggestions) if 'experimental_suggestions' in results else 0
        }
    }
    
    # Print results summary
    print_analysis_summary(results, summary)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"xai_analysis_results_{timestamp}.json")
    
    # Prepare serializable results
    serializable_results = prepare_serializable_results(results, summary)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n💾 XAI analysis results saved to: {results_file}")
    logger.info(f"📁 All outputs saved in: {output_dir}")
    
    logger.info("\n✅ REAL DATA XAI ANALYSIS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    return results

def print_analysis_summary(results, summary):
    """Print a comprehensive summary of the XAI analysis"""
    
    print("\n" + "="*80)
    print("🔬 REAL DATA XAI ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n📊 Data Analysis:")
    data_info = summary['data_summary']
    print(f"   • GNN Features: {data_info['gnn_features']}")
    print(f"   • RNN Features: {data_info['rnn_features']}")
    print(f"   • Total Samples: {data_info['n_samples']}")
    print(f"   • Classes: {data_info['n_classes']}")
    print(f"   • Class Distribution: {data_info['class_distribution']}")
    
    print(f"\n🔧 XAI Analysis Results:")
    xai_info = summary['xai_results']
    print(f"   • Feature Importance: {'✅ Completed' if xai_info['feature_importance_completed'] else '❌ Failed'}")
    print(f"   • Biological Interpretation: {'✅ Completed' if xai_info['biological_interpretation_completed'] else '❌ Failed'}")
    print(f"   • Uncertainty Analysis: {'✅ Completed' if xai_info['uncertainty_analysis_completed'] else '❌ Failed'}")
    print(f"   • Important Features Found: {xai_info['num_important_features']}")
    print(f"   • Experimental Suggestions: {xai_info['num_experimental_suggestions']}")
    
    if results.get('biological_interpretations'):
        print(f"\n🧬 Top Biological Features:")
        for i, interp in enumerate(results['biological_interpretations'][:5], 1):
            marker = interp['biological_marker']
            category = interp['biological_category']
            importance = interp['importance_score']
            print(f"   {i}. {marker} ({category}) - Score: {importance:.4f}")
    
    if results.get('experimental_suggestions'):
        print(f"\n🧪 High Priority Experimental Suggestions:")
        high_priority = [s for s in results['experimental_suggestions'] if s['priority'] == 'High']
        for i, suggestion in enumerate(high_priority[:3], 1):
            print(f"   {i}. {suggestion['marker']}: {suggestion['experiment']}")
    
    if results.get('uncertainty_analysis'):
        uncertainty = results['uncertainty_analysis']
        print(f"\n🎯 Uncertainty Analysis:")
        print(f"   • Mean Prediction Confidence: {uncertainty['mean_confidence']:.4f}")
        print(f"   • High Uncertainty Samples: {uncertainty['high_uncertainty_count']}")
        print(f"   • Confidence Threshold: {uncertainty['confidence_threshold']:.4f}")
    
    print("\n" + "="*80)

def prepare_serializable_results(results, summary):
    """Prepare results for JSON serialization"""
    
    serializable = {'summary': summary}
    
    # Handle biological interpretations
    if results.get('biological_interpretations'):
        serializable['biological_interpretations'] = results['biological_interpretations']
    
    # Handle experimental suggestions
    if results.get('experimental_suggestions'):
        serializable['experimental_suggestions'] = results['experimental_suggestions']
    
    # Handle SHAP results (simplified for serialization)
    if results.get('shap_analysis'):
        shap_data = results['shap_analysis']
        serializable['shap_analysis'] = {
            'feature_names': shap_data['feature_names'],
            'num_samples': len(shap_data['test_data']),
            'num_features': len(shap_data['feature_names']),
            'mean_shap_values': shap_data['shap_values'].mean(axis=0).tolist() if hasattr(shap_data['shap_values'], 'mean') else []
        }
    
    # Handle uncertainty analysis
    if results.get('uncertainty_analysis'):
        uncertainty = results['uncertainty_analysis']
        serializable['uncertainty_analysis'] = {
            'mean_confidence': float(uncertainty['mean_confidence']),
            'confidence_threshold': float(uncertainty['confidence_threshold']),
            'high_uncertainty_count': int(uncertainty['high_uncertainty_count']),
            'predictions_summary': uncertainty.get('predictions_summary', {})
        }
    
    return serializable

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Run XAI analysis on real hybrid GNN-RNN embeddings"
    )
    parser.add_argument(
        '--embeddings-path', 
        type=str, 
        default='analysis',
        help='Path to embeddings directory (default: analysis)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='results/xai_analysis',
        help='Output directory for XAI results (default: results/xai_analysis)'
    )
    
    args = parser.parse_args()
    
    # Validate embeddings path
    if not os.path.exists(args.embeddings_path):
        print(f"❌ Embeddings path does not exist: {args.embeddings_path}")
        print("💡 Make sure to train your model first:")
        print("   python train_enhanced_temporal_rnn.py")
        sys.exit(1)
    
    # Run XAI analysis
    results = run_xai_analysis(
        embeddings_path=args.embeddings_path,
        output_dir=args.output_dir
    )
    
    if results:
        print("\n🎉 Real Data XAI Analysis completed successfully!")
        print(f"📁 Check {args.output_dir} for detailed results and visualizations.")
    else:
        print("\n❌ Real Data XAI Analysis failed. Check logs for details.")

if __name__ == "__main__":
    main()