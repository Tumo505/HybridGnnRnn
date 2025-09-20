"""
Authentic 10X Genomics Data Processor for Cardiomyocyte Analysis

This module handles loading, processing, and caching of 10X Genomics spatial
transcriptomics data for cardiomyocyte subtype classification.
"""
import pickle
import logging
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class Authentic10XProcessor:
    """Processor for authentic 10X Genomics spatial transcriptomics data."""
    
    def __init__(self, cache_dir="cache"):
        """Initialize the processor.
        
        Args:
            cache_dir (str): Directory for caching processed data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "improved_authentic_10x_processed.pkl"
        
    def load_cached_data(self, device='cpu'):
        """Load cached processed data.
        
        Args:
            device (str): Device to load data onto ('cpu' or 'cuda')
            
        Returns:
            torch_geometric.data.Data: Processed graph data
            
        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        if not self.cache_file.exists():
            raise FileNotFoundError(
                f"No cached data found at {self.cache_file}. "
                "Run the data processing pipeline first."
            )
        
        logger.info("Loading cached cardiomyocyte subtype data...")
        with open(self.cache_file, 'rb') as f:
            data = pickle.load(f)
        
        data = data.to(device)
        
        logger.info(f"Dataset loaded: {data.x.shape[0]} cardiomyocytes, {data.x.shape[1]} genes")
        logger.info(f"Cardiomyocyte subtypes: {data.num_classes}")
        
        # Log class distribution
        class_counts = torch.bincount(data.y)
        logger.info(f"Subtype distribution: {class_counts.tolist()}")
        for i, count in enumerate(class_counts):
            percentage = 100 * count / len(data.y)
            logger.info(f"  Subtype {i}: {count} cells ({percentage:.1f}%)")
            
        return data
    
    def cache_exists(self):
        """Check if cached data exists.
        
        Returns:
            bool: True if cache file exists
        """
        return self.cache_file.exists()
    
    def clear_cache(self):
        """Clear the cached data."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            logger.info(f"Cache cleared: {self.cache_file}")
    
    def get_cache_info(self):
        """Get information about the cached data.
        
        Returns:
            dict: Cache information
        """
        if not self.cache_file.exists():
            return {"exists": False, "size": 0}
        
        size = self.cache_file.stat().st_size
        return {
            "exists": True,
            "path": str(self.cache_file),
            "size_bytes": size,
            "size_mb": size / (1024 * 1024)
        }