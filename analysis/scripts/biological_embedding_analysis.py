"""
Biological Analysis of GNN Embeddings
Focus on the best-performing model (Fold_3) for biological insights
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import logging
from pathlib import Path
import json

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BiologicalEmbeddingAnalyzer:
    """Analyze GNN embeddings from biological perspective"""
    
    def __init__(self, embeddings_dir="trained_gnn_embeddings_20250921_193410"):
        self.embeddings_dir = Path(embeddings_dir)
        self.models_data = {}
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load all model embeddings"""
        logger.info("📊 Loading model embeddings...")
        
        for model_dir in self.embeddings_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                try:
                    embeddings = np.load(model_dir / "embeddings.npy")
                    predictions = np.load(model_dir / "predictions.npy")
                    targets = np.load(model_dir / "targets.npy")
                    logits = np.load(model_dir / "logits.npy")
                    
                    self.models_data[model_name] = {
                        'embeddings': embeddings,
                        'predictions': predictions,
                        'targets': targets,
                        'logits': logits,
                        'accuracy': np.mean(predictions == targets)
                    }
                    
                    logger.info(f"   ✅ Loaded {model_name}: {embeddings.shape}")
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to load {model_name}: {e}")
    
    def analyze_best_model(self):
        """Focus analysis on the best-performing model"""
        # Find best model by accuracy
        best_model = max(self.models_data.keys(), 
                        key=lambda k: self.models_data[k]['accuracy'])
        
        logger.info(f"🏆 Best model: {best_model} (Accuracy: {self.models_data[best_model]['accuracy']:.4f})")
        
        data = self.models_data[best_model]
        embeddings = data['embeddings']
        targets = data['targets']
        predictions = data['predictions']
        
        # Biological analysis
        results = {
            'model_name': best_model,
            'accuracy': data['accuracy'],
            'clustering_analysis': self.analyze_clustering(embeddings, targets),
            'class_separation': self.analyze_class_separation(embeddings, targets),
            'prediction_patterns': self.analyze_prediction_patterns(embeddings, targets, predictions),
            'embedding_statistics': self.compute_embedding_statistics(embeddings, targets)
        }
        
        return results, embeddings, targets, predictions
    
    def analyze_clustering(self, embeddings, targets):
        """Analyze natural clustering in embedding space"""
        logger.info("🔍 Analyzing clustering patterns...")
        
        results = {}
        
        # Test different numbers of clusters
        for n_clusters in [7, 10, 15, 20]:  # Including true number of classes (7)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Silhouette score (higher is better)
            silhouette = silhouette_score(embeddings, cluster_labels)
            
            # ARI with true labels (higher is better)
            ari_true = adjusted_rand_score(targets, cluster_labels)
            
            results[f'k_{n_clusters}'] = {
                'silhouette_score': silhouette,
                'ari_with_true_labels': ari_true,
                'cluster_labels': cluster_labels
            }
            
            logger.info(f"   K={n_clusters}: Silhouette={silhouette:.3f}, ARI={ari_true:.3f}")
        
        return results
    
    def analyze_class_separation(self, embeddings, targets):
        """Analyze how well classes are separated in embedding space"""
        logger.info("📏 Analyzing class separation...")
        
        # Calculate pairwise distances within and between classes
        unique_classes = np.unique(targets)
        n_classes = len(unique_classes)
        
        within_class_distances = []
        between_class_distances = []
        
        for class_i in unique_classes:
            mask_i = targets == class_i
            embeddings_i = embeddings[mask_i]
            
            if len(embeddings_i) > 1:
                # Within-class distances
                within_dist = pdist(embeddings_i)
                within_class_distances.extend(within_dist)
            
            # Between-class distances
            for class_j in unique_classes:
                if class_j > class_i:  # Avoid duplicates
                    mask_j = targets == class_j
                    embeddings_j = embeddings[mask_j]
                    
                    # Calculate all pairwise distances between classes
                    for emb_i in embeddings_i:
                        for emb_j in embeddings_j:
                            between_class_distances.append(np.linalg.norm(emb_i - emb_j))
        
        within_mean = np.mean(within_class_distances)
        between_mean = np.mean(between_class_distances)
        separation_ratio = between_mean / within_mean if within_mean > 0 else np.inf
        
        logger.info(f"   Within-class distance: {within_mean:.3f}")
        logger.info(f"   Between-class distance: {between_mean:.3f}")
        logger.info(f"   Separation ratio: {separation_ratio:.3f}")
        
        return {
            'within_class_mean': within_mean,
            'between_class_mean': between_mean,
            'separation_ratio': separation_ratio,
            'within_class_std': np.std(within_class_distances),
            'between_class_std': np.std(between_class_distances)
        }
    
    def analyze_prediction_patterns(self, embeddings, targets, predictions):
        """Analyze what the model learned about each class"""
        logger.info("🎯 Analyzing prediction patterns...")
        
        unique_classes = np.unique(targets)
        class_analysis = {}
        
        for class_id in unique_classes:
            mask = targets == class_id
            class_embeddings = embeddings[mask]
            class_predictions = predictions[mask]
            
            # Class accuracy
            accuracy = np.mean(class_predictions == class_id)
            
            # Most common misclassifications
            misclassified = class_predictions[class_predictions != class_id]
            if len(misclassified) > 0:
                unique, counts = np.unique(misclassified, return_counts=True)
                most_common_error = unique[np.argmax(counts)]
            else:
                most_common_error = None
            
            # Embedding statistics for this class
            centroid = np.mean(class_embeddings, axis=0)
            spread = np.mean(np.linalg.norm(class_embeddings - centroid, axis=1))
            
            class_analysis[int(class_id)] = {
                'sample_count': int(np.sum(mask)),
                'accuracy': float(accuracy),
                'most_common_error': int(most_common_error) if most_common_error is not None else None,
                'embedding_spread': float(spread),
                'centroid': centroid.tolist()
            }
            
            logger.info(f"   Class {class_id}: {np.sum(mask)} samples, {accuracy:.3f} accuracy")
        
        return class_analysis
    
    def compute_embedding_statistics(self, embeddings, targets):
        """Compute overall embedding space statistics"""
        logger.info("📊 Computing embedding statistics...")
        
        # Overall statistics
        mean_embedding = np.mean(embeddings, axis=0)
        std_embedding = np.std(embeddings, axis=0)
        
        # Dimensionality analysis
        explained_variance_pca = PCA().fit(embeddings).explained_variance_ratio_
        cumsum_variance = np.cumsum(explained_variance_pca)
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
        
        # Embedding density
        pairwise_distances = pdist(embeddings)
        
        return {
            'embedding_dim': embeddings.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
            'n_components_95_variance': int(n_components_95),
            'total_variance_in_top10': float(np.sum(explained_variance_pca[:10])),
            'mean_pairwise_distance': float(np.mean(pairwise_distances)),
            'std_pairwise_distance': float(np.std(pairwise_distances))
        }
    
    def create_biological_visualization(self, embeddings, targets, predictions, model_name):
        """Create comprehensive biological visualization"""
        logger.info("🎨 Creating biological visualization...")
        
        # Apply t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Biological Analysis: {model_name}', fontsize=16, fontweight='bold')
        
        # 1. True classes
        ax = axes[0, 0]
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=targets, cmap='tab10', alpha=0.6, s=20)
        ax.set_title('True Cell States', fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax, label='True Class')
        
        # 2. Predictions
        ax = axes[0, 1]
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=predictions, cmap='tab10', alpha=0.6, s=20)
        ax.set_title('Predicted Cell States', fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax, label='Predicted Class')
        
        # 3. Correct vs Incorrect
        ax = axes[0, 2]
        correct = (predictions == targets)
        colors = ['red' if not c else 'green' for c in correct]
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                  c=colors, alpha=0.6, s=20)
        ax.set_title(f'Prediction Accuracy: {np.mean(correct):.3f}', fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        
        # 4. Class centroids and connections
        ax = axes[1, 0]
        unique_classes = np.unique(targets)
        centroids_2d = []
        
        for class_id in unique_classes:
            mask = targets == class_id
            if np.sum(mask) > 0:
                centroid_2d = np.mean(embeddings_2d[mask], axis=0)
                centroids_2d.append(centroid_2d)
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          alpha=0.3, s=10, label=f'Class {class_id}')
                ax.scatter(centroid_2d[0], centroid_2d[1], 
                          s=200, c='black', marker='x')
                ax.annotate(f'C{class_id}', centroid_2d, fontweight='bold')
        
        ax.set_title('Class Centroids', fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        
        # 5. Embedding norms distribution
        ax = axes[1, 1]
        norms = np.linalg.norm(embeddings, axis=1)
        ax.hist(norms, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title('Embedding Magnitude Distribution', fontweight='bold')
        ax.set_xlabel('L2 Norm')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(norms), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(norms):.2f}')
        ax.legend()
        
        # 6. Class-wise accuracy
        ax = axes[1, 2]
        class_accuracies = []
        class_labels = []
        
        for class_id in unique_classes:
            mask = targets == class_id
            if np.sum(mask) > 0:
                accuracy = np.mean(predictions[mask] == class_id)
                class_accuracies.append(accuracy)
                class_labels.append(f'Class {class_id}')
        
        bars = ax.bar(class_labels, class_accuracies, alpha=0.7)
        ax.set_title('Per-Class Accuracy', fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=45)
        
        # Color bars by accuracy
        for bar, acc in zip(bars, class_accuracies):
            bar.set_color('green' if acc > 0.5 else 'orange' if acc > 0.3 else 'red')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"biological_analysis_{model_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   💾 Saved: {plot_path}")
        return plot_path

def main():
    """Main biological analysis"""
    logger.info("🧬 BIOLOGICAL EMBEDDING ANALYSIS")
    logger.info("=" * 60)
    
    analyzer = BiologicalEmbeddingAnalyzer()
    
    if not analyzer.models_data:
        logger.error("❌ No embedding data found!")
        return
    
    # Analyze the best model
    results, embeddings, targets, predictions = analyzer.analyze_best_model()
    
    # Create biological visualization
    plot_path = analyzer.create_biological_visualization(
        embeddings, targets, predictions, results['model_name']
    )
    
    # Save analysis results
    output_path = f"biological_analysis_{results['model_name']}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 BIOLOGICAL INSIGHTS SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Best Model: {results['model_name']}")
    logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
    
    sep = results['class_separation']
    logger.info(f"\n📏 Class Separation:")
    logger.info(f"   Within-class distance: {sep['within_class_mean']:.3f}")
    logger.info(f"   Between-class distance: {sep['between_class_mean']:.3f}")
    logger.info(f"   Separation ratio: {sep['separation_ratio']:.3f}")
    
    clustering = results['clustering_analysis']
    logger.info(f"\n🔍 Clustering Analysis:")
    for k, metrics in clustering.items():
        logger.info(f"   {k}: Silhouette={metrics['silhouette_score']:.3f}, ARI={metrics['ari_with_true_labels']:.3f}")
    
    stats = results['embedding_statistics']
    logger.info(f"\n📊 Embedding Statistics:")
    logger.info(f"   Dimensionality: {stats['embedding_dim']}")
    logger.info(f"   Components for 95% variance: {stats['n_components_95_variance']}")
    logger.info(f"   Mean embedding norm: {stats['mean_norm']:.3f}")
    
    class_patterns = results['prediction_patterns']
    logger.info(f"\n🎯 Class Performance:")
    for class_id, data in class_patterns.items():
        logger.info(f"   Class {class_id}: {data['sample_count']} samples, {data['accuracy']:.3f} accuracy")
    
    logger.info(f"\n💾 Results saved to: {output_path}")
    logger.info(f"🎨 Visualization: {plot_path}")
    
    return results

if __name__ == "__main__":
    results = main()