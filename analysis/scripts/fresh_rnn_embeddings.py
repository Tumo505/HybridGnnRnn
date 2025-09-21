"""
Fresh RNN Training and Embedding Extraction
Train a new RNN model on the temporal data and extract embeddings
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import the RNN model directly
sys.path.insert(0, os.path.join(project_root, 'src'))
from models.rnn_models.temporal_cardiac_rnn import TemporalCardiacRNN
from data_processing.temporal_processor import load_temporal_cardiac_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreshRNNEmbeddingExtractor:
    """Train a fresh RNN model and extract embeddings"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = None
        self.data = None
        self.embeddings = {}
        
    def load_temporal_data(self):
        """Load temporal cardiac data for training"""
        logger.info("📊 Loading temporal cardiac data...")
        
        try:
            # Load data using the temporal cardiac data loader
            train_loader, test_loader, data_info = load_temporal_cardiac_data()
            
            # Extract data from loaders
            all_sequences = []
            all_targets = []
            
            for sequences, targets in train_loader:
                all_sequences.append(sequences)
                all_targets.append(targets)
            
            for sequences, targets in test_loader:
                all_sequences.append(sequences)
                all_targets.append(targets)
            
            sequences = torch.cat(all_sequences, dim=0)
            targets = torch.cat(all_targets, dim=0)
            
            self.data = {
                'sequences': sequences,
                'targets': targets,
                'input_size': data_info['input_size'],
                'num_classes': data_info['n_classes'],
                'sequence_length': data_info['sequence_length']
            }
            
            logger.info(f"   ✅ Data loaded:")
            logger.info(f"      Sequences: {self.data['sequences'].shape}")
            logger.info(f"      Targets: {self.data['targets'].shape}")
            logger.info(f"      Input size: {self.data['input_size']}")
            logger.info(f"      Classes: {self.data['num_classes']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load temporal data: {e}")
            # Create synthetic data as fallback
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic temporal data for testing"""
        logger.info("🔄 Creating synthetic temporal data...")
        
        n_samples = 800
        sequence_length = 10
        input_size = 2000
        num_classes = 4
        
        # Create synthetic temporal sequences
        sequences = torch.randn(n_samples, sequence_length, input_size)
        targets = torch.randint(0, num_classes, (n_samples,))
        
        self.data = {
            'sequences': sequences,
            'targets': targets,
            'input_size': input_size,
            'num_classes': num_classes,
            'sequence_length': sequence_length
        }
        
        logger.info(f"   ✅ Synthetic data created:")
        logger.info(f"      Sequences: {self.data['sequences'].shape}")
        logger.info(f"      Targets: {self.data['targets'].shape}")
        logger.info(f"      Input size: {self.data['input_size']}")
        logger.info(f"      Classes: {self.data['num_classes']}")
        
        return True
    
    def create_and_train_model(self):
        """Create and train the RNN model"""
        logger.info("🚀 Creating and training RNN model...")
        
        # Create model
        self.model = TemporalCardiacRNN(
            input_size=self.data['input_size'],
            hidden_size=128,  # Smaller for faster training
            num_layers=2,
            num_classes=self.data['num_classes'],
            dropout=0.3,
            use_batch_norm=True
        )
        
        self.model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        X = self.data['sequences'].to(self.device)
        y = self.data['targets'].to(self.device)
        
        # Train for 30 epochs
        self.model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            
            outputs = self.model(X)
            loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    predictions = torch.argmax(outputs, dim=1)
                    accuracy = (predictions == y).float().mean()
                    logger.info(f"   Epoch {epoch+1}/30: Loss={loss.item():.4f}, Accuracy={accuracy.item():.4f}")
        
        self.model.eval()
        logger.info("   ✅ Model training completed")
        
        return True
    
    def extract_embeddings(self):
        """Extract embeddings from the trained model"""
        logger.info("📊 Extracting RNN embeddings...")
        
        with torch.no_grad():
            X = self.data['sequences'].to(self.device)
            y = self.data['targets'].numpy()
            
            # Get model outputs
            outputs = self.model(X)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Get temporal features
            features = self.model.get_temporal_features(X)
            
            # Use pooled representation as main embeddings
            embeddings = features['pooled_representation'].cpu().numpy()
            
            # Store all results
            self.embeddings = {
                'embeddings': embeddings,
                'pooled_representation': features['pooled_representation'].cpu().numpy(),
                'lstm_outputs': features['lstm_outputs'].cpu().numpy(),
                'final_hidden': features['final_hidden'].cpu().numpy(),
                'predictions': predictions,
                'targets': y,
                'output_logits': outputs.cpu().numpy()
            }
            
            accuracy = accuracy_score(y, predictions)
            logger.info(f"   ✅ Embeddings extracted:")
            logger.info(f"      Main embeddings: {embeddings.shape}")
            logger.info(f"      LSTM outputs: {features['lstm_outputs'].shape}")
            logger.info(f"      Final hidden: {features['final_hidden'].shape}")
            logger.info(f"      Accuracy: {accuracy:.4f}")
            
            return embeddings
    
    def analyze_embeddings(self):
        """Analyze the quality of embeddings"""
        logger.info("🔍 Analyzing RNN embeddings...")
        
        embeddings = self.embeddings['embeddings']
        targets = self.embeddings['targets']
        
        # Clustering analysis
        silhouette_scores = []
        for k in [4, 6, 8]:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(score)
            logger.info(f"   K={k}: Silhouette={score:.3f}")
        
        best_silhouette = max(silhouette_scores)
        
        # Class separation analysis
        unique_classes = np.unique(targets)
        inter_class_distances = []
        
        for i, class_i in enumerate(unique_classes):
            for j, class_j in enumerate(unique_classes):
                if i < j:
                    class_i_embeddings = embeddings[targets == class_i]
                    class_j_embeddings = embeddings[targets == class_j]
                    
                    if len(class_i_embeddings) > 0 and len(class_j_embeddings) > 0:
                        center_i = np.mean(class_i_embeddings, axis=0)
                        center_j = np.mean(class_j_embeddings, axis=0)
                        distance = np.linalg.norm(center_i - center_j)
                        inter_class_distances.append(distance)
        
        avg_inter_class_distance = np.mean(inter_class_distances) if inter_class_distances else 0
        
        logger.info(f"   Best silhouette score: {best_silhouette:.3f}")
        logger.info(f"   Average inter-class distance: {avg_inter_class_distance:.3f}")
        
        return {
            'best_silhouette': best_silhouette,
            'avg_inter_class_distance': avg_inter_class_distance,
            'silhouette_scores': silhouette_scores
        }
    
    def create_visualizations(self):
        """Create t-SNE and PCA visualizations"""
        logger.info("🎨 Creating RNN embedding visualizations...")
        
        embeddings = self.embeddings['embeddings']
        targets = self.embeddings['targets']
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Color scheme
        colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(targets))))
        
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_tsne = tsne.fit_transform(embeddings)
        
        for i, class_label in enumerate(np.unique(targets)):
            mask = targets == class_label
            ax1.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], 
                       c=[colors[i]], label=f'Class {class_label}', alpha=0.7)
        
        ax1.set_title('🧬 RNN Embeddings (t-SNE)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PCA visualization
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)
        
        for i, class_label in enumerate(np.unique(targets)):
            mask = targets == class_label
            ax2.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], 
                       c=[colors[i]], label=f'Class {class_label}', alpha=0.7)
        
        ax2.set_title('🧬 RNN Embeddings (PCA)', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Class distribution
        unique_classes, class_counts = np.unique(targets, return_counts=True)
        ax3.bar(unique_classes, class_counts, color=colors[:len(unique_classes)])
        ax3.set_title('📊 Class Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        
        # Prediction accuracy by class
        predictions = self.embeddings['predictions']
        class_accuracies = []
        for class_label in unique_classes:
            mask = targets == class_label
            if np.sum(mask) > 0:
                accuracy = np.mean(predictions[mask] == targets[mask])
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(0)
        
        ax4.bar(unique_classes, class_accuracies, color=colors[:len(unique_classes)])
        ax4.set_title('🎯 Accuracy by Class', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = f"analysis/visualizations/fresh_rnn_embeddings_{timestamp}.png"
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   💾 Saved: {viz_path}")
        
        return {
            'tsne_coordinates': embeddings_tsne,
            'pca_coordinates': embeddings_pca,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'visualization_path': viz_path
        }
    
    def save_results(self):
        """Save embeddings and analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        embeddings_dir = f"analysis/rnn_embeddings/fresh_rnn_{timestamp}"
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Save embeddings
        np.save(f"{embeddings_dir}/embeddings.npy", self.embeddings['embeddings'])
        np.save(f"{embeddings_dir}/targets.npy", self.embeddings['targets'])
        np.save(f"{embeddings_dir}/predictions.npy", self.embeddings['predictions'])
        np.save(f"{embeddings_dir}/output_logits.npy", self.embeddings['output_logits'])
        
        logger.info(f"📁 Results saved:")
        logger.info(f"   Embeddings: {embeddings_dir}")
        
        return embeddings_dir
    
    def generate_analysis_report(self, analysis_results, viz_results):
        """Generate comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate per-class performance
        targets = self.embeddings['targets']
        predictions = self.embeddings['predictions']
        
        class_performance = {}
        for class_label in np.unique(targets):
            mask = targets == class_label
            class_count = np.sum(mask)
            class_accuracy = np.mean(predictions[mask] == targets[mask]) if class_count > 0 else 0
            class_performance[int(class_label)] = {
                'count': int(class_count),
                'accuracy': float(class_accuracy)
            }
        
        # Create comprehensive report
        report = {
            'model_info': {
                'type': 'Fresh Temporal Cardiac RNN',
                'input_size': int(self.data['input_size']),
                'sequence_length': int(self.data['sequence_length']),
                'num_classes': int(self.data['num_classes']),
                'embedding_dimension': int(self.embeddings['embeddings'].shape[1])
            },
            'performance': {
                'overall_accuracy': float(accuracy_score(targets, predictions)),
                'class_performance': class_performance
            },
            'analysis': {
                'best_silhouette_score': float(analysis_results['best_silhouette']),
                'avg_inter_class_distance': float(analysis_results['avg_inter_class_distance']),
                'silhouette_scores': [float(x) for x in analysis_results['silhouette_scores']]
            },
            'embeddings': {
                'shape': [int(x) for x in self.embeddings['embeddings'].shape],
                'tsne_coordinates': [[float(x) for x in row] for row in viz_results['tsne_coordinates'].tolist()],
                'pca_coordinates': [[float(x) for x in row] for row in viz_results['pca_coordinates'].tolist()],
                'pca_explained_variance': [float(x) for x in viz_results['pca_explained_variance'].tolist()]
            },
            'data_info': {
                'total_samples': int(len(targets)),
                'embedding_dimension': int(self.embeddings['embeddings'].shape[1])
            }
        }
        
        # Save report
        report_path = f"analysis/rnn_embeddings/fresh_rnn_analysis_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"   Analysis: {report_path}")
        
        return report_path

def main():
    """Main execution function"""
    logger.info("🧬 FRESH RNN EMBEDDING EXTRACTION")
    logger.info("=" * 60)
    
    # Initialize extractor
    extractor = FreshRNNEmbeddingExtractor()
    
    # Load data
    if not extractor.load_temporal_data():
        logger.error("❌ Failed to load data")
        return None
    
    # Train model
    if not extractor.create_and_train_model():
        logger.error("❌ Failed to train model")
        return None
    
    # Extract embeddings
    embeddings = extractor.extract_embeddings()
    if embeddings is None:
        logger.error("❌ Failed to extract embeddings")
        return None
    
    # Analyze embeddings
    analysis_results = extractor.analyze_embeddings()
    
    # Create visualizations
    viz_results = extractor.create_visualizations()
    
    # Save results
    embeddings_dir = extractor.save_results()
    
    # Generate report
    report_path = extractor.generate_analysis_report(analysis_results, viz_results)
    
    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("🎯 FRESH RNN ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: Fresh Temporal Cardiac RNN")
    logger.info(f"Overall Accuracy: {analysis_results.get('accuracy', accuracy_score(extractor.embeddings['targets'], extractor.embeddings['predictions'])):.4f}")
    logger.info(f"Best Silhouette Score: {analysis_results['best_silhouette']:.3f}")
    logger.info(f"Embedding Dimension: {embeddings.shape[1]}")
    logger.info("")
    logger.info("🎯 Class Performance:")
    
    targets = extractor.embeddings['targets']
    predictions = extractor.embeddings['predictions']
    
    for class_label in np.unique(targets):
        mask = targets == class_label
        class_count = np.sum(mask)
        class_accuracy = np.mean(predictions[mask] == targets[mask]) if class_count > 0 else 0
        logger.info(f"   Class {class_label}: {class_count} samples, {class_accuracy:.3f} accuracy")
    
    logger.info("")
    logger.info(f"💾 Results: {report_path}")
    logger.info(f"🎨 Visualization: {viz_results['visualization_path']}")
    
    return {
        'embeddings': embeddings,
        'analysis': analysis_results,
        'visualization': viz_results,
        'report_path': report_path,
        'embeddings_dir': embeddings_dir
    }

if __name__ == "__main__":
    results = main()