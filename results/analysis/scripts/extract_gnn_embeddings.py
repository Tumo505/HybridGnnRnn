"""
Extract Embeddings from Trained GNN Models
This script recreates the correct model architectures and extracts embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import logging
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainedGNNEncoder(nn.Module):
    """Recreate the GNN encoder architecture from trained models"""
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        
        # GNN encoder layers (matching the saved model structure)
        self.gnn_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),     # 512 -> 256
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),    # 256 -> 128
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 7)  # 7 classes
        )
        
        # Additional components found in saved models
        self.efficiency_predictor = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        # Extract features through GNN encoder
        features = self.gnn_encoder(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits, features

class TrainedHybridModel(nn.Module):
    """Recreate the Hybrid GNN-RNN architecture from trained models"""
    
    def __init__(self, input_dim=512, hidden_dim=256, gnn_output_dim=128, rnn_hidden_dim=128):
        super().__init__()
        
        # GNN encoder (same as GNN-only model)
        self.gnn_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),     # 512 -> 256
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, gnn_output_dim),    # 256 -> 128
            nn.BatchNorm1d(gnn_output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Temporal RNN (BiLSTM with 2 layers)
        self.temporal_rnn = nn.LSTM(
            input_size=gnn_output_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=256,  # 2 * rnn_hidden_dim for bidirectional
            num_heads=8,
            dropout=0.3,
            batch_first=True
        )
        
        # Final classifier (takes concatenated features)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # RNN output size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)
        )
        
        # Additional components
        self.efficiency_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
    
    def forward(self, x, sequence_length=10):
        # GNN encoding
        gnn_features = self.gnn_encoder(x)  # [batch_size, 128]
        
        # Reshape for RNN (simulate temporal sequences)
        batch_size = gnn_features.size(0)
        temporal_features = gnn_features.unsqueeze(1).repeat(1, sequence_length, 1)  # [batch_size, seq_len, 128]
        
        # RNN processing
        rnn_output, _ = self.temporal_rnn(temporal_features)  # [batch_size, seq_len, 256]
        
        # Attention over temporal dimension
        attended_output, _ = self.temporal_attention(rnn_output, rnn_output, rnn_output)
        
        # Aggregate temporal information (mean pooling)
        final_features = attended_output.mean(dim=1)  # [batch_size, 256]
        
        # Classification
        logits = self.classifier(final_features)
        
        return logits, final_features

class TrainedModelAnalyzer:
    """Load and analyze trained GNN models"""
    
    def __init__(self):
        self.models = {}
        self.embeddings = {}
        self.data = None
        self.device = 'cpu'
        
    def load_data(self):
        """Load the training data"""
        logger.info("📊 Loading training data...")
        
        if not Path('data/aligned_spatial_temporal_data.pt').exists():
            logger.error("❌ Data file not found: data/aligned_spatial_temporal_data.pt")
            return False
        
        data = torch.load('data/aligned_spatial_temporal_data.pt', map_location=self.device)
        self.data = {
            'features': data['spatial']['features'],
            'targets': data['spatial']['targets'],
            'edge_index': data['spatial']['edge_index']
        }
        
        logger.info(f"   ✅ Data loaded: {self.data['features'].shape[0]} samples, {self.data['features'].shape[1]} features")
        return True
    
    def load_trained_model(self, checkpoint_path, model_name):
        """Load a specific trained model"""
        logger.info(f"🤖 Loading model: {model_name}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            
            # Determine model type based on layers present
            has_rnn = any('temporal_rnn' in key for key in state_dict.keys())
            
            if has_rnn:
                # Hybrid GNN-RNN model
                model = TrainedHybridModel()
                logger.info(f"   📊 Model type: Hybrid GNN-RNN")
            else:
                # GNN-only model
                model = TrainedGNNEncoder()
                logger.info(f"   📊 Model type: GNN-only")
            
            # Load weights
            model.load_state_dict(state_dict, strict=False)  # Use strict=False for partial loading
            model.eval()
            model.to(self.device)
            
            self.models[model_name] = {
                'model': model,
                'type': 'hybrid' if has_rnn else 'gnn_only',
                'checkpoint_info': {
                    'epoch': checkpoint.get('epoch', 'N/A'),
                    'best_val_score': checkpoint.get('best_val_score', 'N/A'),
                    'config': checkpoint.get('config', {})
                }
            }
            
            logger.info(f"   ✅ Successfully loaded {model_name}")
            logger.info(f"      Best val score: {checkpoint.get('best_val_score', 'N/A'):.4f}")
            
            return True
            
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to load {model_name}: {e}")
            return False
    
    def extract_embeddings(self, model_name):
        """Extract embeddings from a trained model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        logger.info(f"📊 Extracting embeddings from {model_name}")
        
        with torch.no_grad():
            x = self.data['features']
            
            if model_info['type'] == 'hybrid':
                logits, embeddings = model(x)
            else:
                logits, embeddings = model(x)
            
            # Store embeddings and predictions
            self.embeddings[model_name] = {
                'embeddings': embeddings.cpu().numpy(),
                'predictions': torch.argmax(logits, dim=1).cpu().numpy(),
                'logits': logits.cpu().numpy(),
                'targets': self.data['targets'].cpu().numpy()
            }
            
            # Calculate accuracy
            accuracy = accuracy_score(self.data['targets'].cpu().numpy(), 
                                    torch.argmax(logits, dim=1).cpu().numpy())
            
            logger.info(f"   📈 Accuracy: {accuracy:.4f}")
            logger.info(f"   📊 Embedding shape: {embeddings.shape}")
            
            return embeddings.cpu().numpy()
    
    def visualize_embeddings(self, model_name, method='tsne'):
        """Create embedding visualizations"""
        if model_name not in self.embeddings:
            logger.warning(f"No embeddings found for {model_name}")
            return None
        
        logger.info(f"🎨 Creating {method.upper()} visualization for {model_name}")
        
        embeddings = self.embeddings[model_name]['embeddings']
        targets = self.embeddings[model_name]['targets']
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            logger.error(f"Unknown method: {method}")
            return None
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Main scatter plot
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=targets, cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Class')
        plt.title(f'{model_name} - {method.upper()} by Class', fontweight='bold')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        # Predictions vs targets
        plt.subplot(2, 2, 2)
        predictions = self.embeddings[model_name]['predictions']
        scatter2 = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=predictions, cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter2, label='Predicted Class')
        plt.title(f'{model_name} - Predictions', fontweight='bold')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        # Confidence scores
        plt.subplot(2, 2, 3)
        logits = self.embeddings[model_name]['logits']
        confidence = np.max(F.softmax(torch.tensor(logits), dim=1).numpy(), axis=1)
        scatter3 = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=confidence, cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter3, label='Confidence')
        plt.title(f'{model_name} - Prediction Confidence', fontweight='bold')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        # Accuracy per class
        plt.subplot(2, 2, 4)
        correct = (predictions == targets)
        scatter4 = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=correct, cmap='RdYlGn', alpha=0.6, s=30)
        plt.colorbar(scatter4, label='Correct')
        plt.title(f'{model_name} - Correct/Incorrect', fontweight='bold')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"trained_gnn_embeddings_{model_name}_{method}_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   💾 Saved: {plot_path}")
        return embeddings_2d, plot_path
    
    def analyze_all_models(self):
        """Analyze all loaded models"""
        logger.info("\n🔍 ANALYZING ALL TRAINED MODELS")
        logger.info("=" * 50)
        
        analysis_results = {}
        
        for model_name, model_info in self.models.items():
            logger.info(f"\n📊 Analyzing: {model_name}")
            logger.info("-" * 30)
            
            # Extract embeddings
            embeddings = self.extract_embeddings(model_name)
            
            if embeddings is not None:
                # Create visualizations
                tsne_coords, tsne_path = self.visualize_embeddings(model_name, 'tsne')
                pca_coords, pca_path = self.visualize_embeddings(model_name, 'pca')
                
                # Performance metrics
                targets = self.embeddings[model_name]['targets']
                predictions = self.embeddings[model_name]['predictions']
                accuracy = accuracy_score(targets, predictions)
                
                # Store results
                analysis_results[model_name] = {
                    'model_type': model_info['type'],
                    'accuracy': accuracy,
                    'embedding_shape': embeddings.shape,
                    'checkpoint_info': model_info['checkpoint_info'],
                    'visualizations': {
                        'tsne': tsne_path,
                        'pca': pca_path
                    }
                }
                
                logger.info(f"   ✅ Analysis complete for {model_name}")
        
        return analysis_results
    
    def save_embeddings(self, results):
        """Save all embeddings and analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create embeddings directory
        embeddings_dir = f"trained_gnn_embeddings_{timestamp}"
        Path(embeddings_dir).mkdir(exist_ok=True)
        
        # Save embeddings as numpy arrays
        for model_name in self.embeddings.keys():
            model_dir = Path(embeddings_dir) / model_name
            model_dir.mkdir(exist_ok=True)
            
            embedding_data = self.embeddings[model_name]
            
            np.save(model_dir / "embeddings.npy", embedding_data['embeddings'])
            np.save(model_dir / "predictions.npy", embedding_data['predictions'])
            np.save(model_dir / "targets.npy", embedding_data['targets'])
            np.save(model_dir / "logits.npy", embedding_data['logits'])
            
            logger.info(f"   💾 Saved embeddings: {model_dir}")
        
        # Save analysis summary
        summary_path = f"trained_gnn_analysis_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Analysis summary: {summary_path}")
        logger.info(f"📁 Embeddings directory: {embeddings_dir}")
        
        return summary_path, embeddings_dir

def main():
    """Main analysis function"""
    logger.info("🧬 TRAINED GNN MODEL ANALYSIS")
    logger.info("=" * 60)
    
    analyzer = TrainedModelAnalyzer()
    
    # Load data
    if not analyzer.load_data():
        return
    
    # Define models to load
    models_to_load = {
        'GNN_Only': 'comprehensive_cpu_results_20250921_173835/ablation_GNN_Only/best_model.pth',
        'Full_Hybrid': 'comprehensive_cpu_results_20250921_173835/ablation_Full_Hybrid_Model/best_model.pth',
        'Fold_1': 'comprehensive_cpu_results_20250921_173835/fold_1/best_model.pth',
        'Fold_2': 'comprehensive_cpu_results_20250921_173835/fold_2/best_model.pth',
        'Fold_3': 'comprehensive_cpu_results_20250921_173835/fold_3/best_model.pth'
    }
    
    # Load models
    loaded_count = 0
    for model_name, model_path in models_to_load.items():
        if Path(model_path).exists():
            if analyzer.load_trained_model(model_path, model_name):
                loaded_count += 1
        else:
            logger.warning(f"   ⚠️ Model file not found: {model_path}")
    
    if loaded_count == 0:
        logger.error("❌ No models could be loaded!")
        return
    
    logger.info(f"\n✅ Successfully loaded {loaded_count} models")
    
    # Analyze all models
    results = analyzer.analyze_all_models()
    
    # Save results
    summary_path, embeddings_dir = analyzer.save_embeddings(results)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 ANALYSIS SUMMARY")
    logger.info("=" * 60)
    
    for model_name, result in results.items():
        logger.info(f"{model_name}:")
        logger.info(f"   Type: {result['model_type']}")
        logger.info(f"   Accuracy: {result['accuracy']:.4f}")
        logger.info(f"   Embeddings: {result['embedding_shape']}")
        logger.info(f"   Val Score: {result['checkpoint_info']['best_val_score']}")
    
    logger.info(f"\n📁 Results saved to: {summary_path}")
    logger.info(f"📁 Embeddings saved to: {embeddings_dir}")
    
    return results

if __name__ == "__main__":
    results = main()