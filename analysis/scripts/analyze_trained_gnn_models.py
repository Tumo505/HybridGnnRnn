"""
Analyze Existing Trained GNN Models and Extract Embeddings
This script loads saved model weights and analyzes their performance and embeddings
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add src to path and import GNN directly
import sys
import os
sys.path.append('src')
sys.path.append('src/models/gnn_models')

# Direct import of the GNN class
exec(open('src/models/gnn_models/cardiomyocyte_gnn.py').read())

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainedGNNAnalyzer:
    """Analyzer for trained GNN models"""
    
    def __init__(self):
        self.models = {}
        self.embeddings = {}
        self.data = None
        self.device = 'cpu'
        
    def find_trained_models(self):
        """Find all available trained model files"""
        model_paths = {}
        
        # Check main directory
        if os.path.exists('best_regularized_model.pth'):
            model_paths['best_regularized'] = 'best_regularized_model.pth'
        
        # Check experiment directories
        experiment_dirs = [
            'comprehensive_cpu_results_20250921_173835',
            'comprehensive_cpu_results_20250921_172426', 
            'comprehensive_cardiomyocyte_results_20250921_141228'
        ]
        
        for exp_dir in experiment_dirs:
            if os.path.exists(exp_dir):
                # Check for ablation models
                ablation_dirs = ['ablation_GNN_Only', 'ablation_Full_Hybrid_Model', 'ablation_RNN_Only']
                for ablation in ablation_dirs:
                    model_path = os.path.join(exp_dir, ablation, 'best_model.pth')
                    if os.path.exists(model_path):
                        model_paths[f'{exp_dir}_{ablation}'] = model_path
                
                # Check fold models
                for fold in range(1, 6):
                    model_path = os.path.join(exp_dir, f'fold_{fold}', 'best_model.pth')
                    if os.path.exists(model_path):
                        model_paths[f'{exp_dir}_fold_{fold}'] = model_path
        
        logger.info(f"🔍 Found {len(model_paths)} trained models:")
        for name, path in model_paths.items():
            logger.info(f"   📁 {name}: {path}")
        
        return model_paths
    
    def load_data(self):
        """Load the data that was used for training"""
        logger.info("📊 Loading training data...")
        
        # Try to load cached data first
        if os.path.exists('data/aligned_spatial_temporal_data.pt'):
            data = torch.load('data/aligned_spatial_temporal_data.pt', map_location=self.device)
            self.data = {
                'features': data['spatial']['features'],
                'targets': data['spatial']['targets'],
                'edge_index': data['spatial']['edge_index']
            }
            logger.info(f"   ✅ Data loaded: {self.data['features'].shape[0]} samples, {self.data['features'].shape[1]} features")
            return True
        else:
            logger.error("❌ Data file not found: data/aligned_spatial_temporal_data.pt")
            return False
    
    def load_model(self, model_path, model_name):
        """Load a specific trained model"""
        try:
            # Create model architecture (you may need to adjust parameters based on your training)
            if 'features' in self.data:
                num_features = self.data['features'].shape[1]
            else:
                num_features = 512  # Default
            
            num_classes = len(torch.unique(self.data['targets'])) if 'targets' in self.data else 7
            
            model = AdvancedCardiomyocyteGNN(
                num_features=num_features,
                num_classes=num_classes,
                hidden_dim=128,
                dropout=0.4
            )
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            model.to(self.device)
            
            self.models[model_name] = model
            logger.info(f"   ✅ Loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load {model_name}: {e}")
            return False
    
    def extract_embeddings(self, model_name):
        """Extract embeddings from a trained model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        model = self.models[model_name]
        
        # Create a modified forward pass to extract embeddings
        def get_embeddings(model, data):
            x, edge_index = data['features'], data['edge_index']
            
            # Follow the model's forward pass up to the final classification layer
            with torch.no_grad():
                # Input normalization
                if hasattr(model, 'input_norm'):
                    x_norm = model.input_norm(x)
                else:
                    x_norm = x
                
                # First GAT layer
                x_gat1 = F.relu(model.gat1(x_norm, edge_index))
                x_gat1 = F.dropout(x_gat1, p=model.dropout, training=False)
                
                # First GCN layer
                x_gcn1 = F.relu(model.gcn1(x_gat1, edge_index))
                
                # Second GAT layer
                x_gat2 = F.relu(model.gat2(x_gcn1, edge_index))
                x_gat2 = F.dropout(x_gat2, p=model.dropout, training=False)
                
                # Second GCN layer
                x_gcn2 = F.relu(model.gcn2(x_gat2, edge_index))
                
                # Skip connection
                if hasattr(model, 'skip_projection'):
                    x_skip = model.skip_projection(x_norm)
                    x_fused = torch.cat([x_gcn2, x_skip], dim=1)
                    if hasattr(model, 'feature_fusion'):
                        embeddings = model.feature_fusion(x_fused)
                    else:
                        embeddings = x_fused
                else:
                    embeddings = x_gcn2
                
                return embeddings
        
        embeddings = get_embeddings(model, self.data)
        self.embeddings[model_name] = embeddings.cpu().numpy()
        
        logger.info(f"   📊 Extracted embeddings for {model_name}: {embeddings.shape}")
        return embeddings.cpu().numpy()
    
    def evaluate_model_performance(self, model_name):
        """Evaluate model performance on the dataset"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        model = self.models[model_name]
        
        with torch.no_grad():
            # Get predictions
            from torch_geometric.data import Data
            data_obj = Data(x=self.data['features'], edge_index=self.data['edge_index'])
            logits = model(data_obj.x, data_obj.edge_index)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            targets = self.data['targets'].cpu().numpy()
            
            # Calculate metrics
            accuracy = accuracy_score(targets, predictions)
            report = classification_report(targets, predictions, output_dict=True)
            
            results = {
                'accuracy': accuracy,
                'classification_report': report,
                'predictions': predictions,
                'targets': targets
            }
            
            logger.info(f"   🎯 {model_name} Accuracy: {accuracy:.4f}")
            return results
    
    def visualize_embeddings(self, model_name, method='tsne'):
        """Create visualization of embeddings"""
        if model_name not in self.embeddings:
            logger.warning(f"No embeddings found for {model_name}")
            return None
        
        embeddings = self.embeddings[model_name]
        targets = self.data['targets'].cpu().numpy()
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            logger.error(f"Unknown visualization method: {method}")
            return None
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=targets, cmap='tab10', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Class')
        plt.title(f'{model_name} Embeddings ({method.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"embeddings_{model_name}_{method}_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   📊 Saved embedding visualization: {plot_path}")
        return embeddings_2d, plot_path
    
    def compare_models(self):
        """Compare performance across all loaded models"""
        logger.info("\n📊 MODEL COMPARISON")
        logger.info("=" * 50)
        
        comparison_results = {}
        
        for model_name in self.models.keys():
            # Get embeddings
            embeddings = self.extract_embeddings(model_name)
            
            # Evaluate performance  
            performance = self.evaluate_model_performance(model_name)
            
            # Create visualizations
            tsne_coords, tsne_path = self.visualize_embeddings(model_name, 'tsne')
            pca_coords, pca_path = self.visualize_embeddings(model_name, 'pca')
            
            comparison_results[model_name] = {
                'embeddings': embeddings,
                'performance': performance,
                'visualizations': {
                    'tsne_path': tsne_path,
                    'pca_path': pca_path
                }
            }
        
        # Summary comparison
        logger.info("\n🏆 PERFORMANCE SUMMARY:")
        for model_name, results in comparison_results.items():
            if results['performance']:
                acc = results['performance']['accuracy']
                logger.info(f"   {model_name}: {acc:.4f}")
        
        return comparison_results
    
    def save_analysis_results(self, results):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary = {}
        for model_name, model_results in results.items():
            summary[model_name] = {
                'accuracy': model_results['performance']['accuracy'] if model_results['performance'] else None,
                'embedding_shape': model_results['embeddings'].shape if model_results['embeddings'] is not None else None,
                'visualizations': model_results['visualizations']
            }
        
        summary_path = f"gnn_analysis_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📁 Analysis summary saved: {summary_path}")
        
        # Save embeddings as numpy arrays
        embeddings_dir = f"gnn_embeddings_{timestamp}"
        os.makedirs(embeddings_dir, exist_ok=True)
        
        for model_name, model_results in results.items():
            if model_results['embeddings'] is not None:
                embedding_path = os.path.join(embeddings_dir, f"{model_name}_embeddings.npy")
                np.save(embedding_path, model_results['embeddings'])
                logger.info(f"   💾 Saved embeddings: {embedding_path}")
        
        return summary_path, embeddings_dir

def main():
    """Main analysis function"""
    logger.info("🧬 GNN MODEL ANALYSIS")
    logger.info("=" * 60)
    
    analyzer = TrainedGNNAnalyzer()
    
    # 1. Find available models
    model_paths = analyzer.find_trained_models()
    
    if not model_paths:
        logger.error("❌ No trained models found!")
        return
    
    # 2. Load data
    if not analyzer.load_data():
        logger.error("❌ Failed to load data!")
        return
    
    # 3. Load models
    logger.info("\n🤖 Loading trained models...")
    loaded_count = 0
    for model_name, model_path in model_paths.items():
        if analyzer.load_model(model_path, model_name):
            loaded_count += 1
    
    if loaded_count == 0:
        logger.error("❌ No models could be loaded!")
        return
    
    logger.info(f"✅ Successfully loaded {loaded_count}/{len(model_paths)} models")
    
    # 4. Analyze models
    logger.info("\n🔍 Analyzing models...")
    results = analyzer.compare_models()
    
    # 5. Save results
    summary_path, embeddings_dir = analyzer.save_analysis_results(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 ANALYSIS COMPLETE")
    logger.info(f"📊 Summary: {summary_path}")
    logger.info(f"📁 Embeddings: {embeddings_dir}")
    logger.info("=" * 60)
    
    return results

if __name__ == "__main__":
    results = main()