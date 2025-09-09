"""
Quick Start Script for Advanced RNN Training
Optimized for NVIDIA RTX 5070

This script provides an easy way to start training the advanced temporal RNN
with optimal settings for your GPU.
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import torch


def check_environment():
    """Check if the environment is ready for training."""
    print("🔍 Environment Check")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check PyTorch installation
    try:
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed!")
        return False
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ CUDA available: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        
        # Check if RTX 5070
        if "5070" in gpu_name:
            print("🚀 RTX 5070 detected - optimal configuration will be used!")
        elif "RTX" in gpu_name or "GTX" in gpu_name:
            print("⚠️  Non-RTX 5070 GPU detected - configuration may need adjustment")
        
    else:
        print("⚠️  CUDA not available - training will use CPU (much slower)")
    
    # Check data availability
    temporal_data_path = Path("data/GSE175634_temporal_data")
    if temporal_data_path.exists():
        print(f"✅ Temporal data found: {temporal_data_path}")
    else:
        print(f"❌ Temporal data not found: {temporal_data_path}")
        print("   Please organize your data first!")
        return False
    
    print("✅ Environment check passed!")
    return True


def create_optimal_config():
    """Create optimal configuration for RTX 5070."""
    
    # Detect GPU memory to adjust batch size
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        
        if gpu_memory_gb >= 12:  # RTX 5070 or better
            batch_size = 64
            hidden_dim = 1024
            num_attention_heads = 16
        elif gpu_memory_gb >= 8:  # RTX 4060 Ti or similar
            batch_size = 48
            hidden_dim = 768
            num_attention_heads = 12
        else:  # Smaller GPUs
            batch_size = 32
            hidden_dim = 512
            num_attention_heads = 8
    else:
        # CPU configuration
        batch_size = 16
        hidden_dim = 256
        num_attention_heads = 4
    
    config = {
        # Model architecture optimized for detected hardware
        'model': {
            'hidden_dim': hidden_dim,
            'num_layers': 4,
            'num_attention_heads': num_attention_heads,
            'embedding_dim': hidden_dim // 2,
            'dropout': 0.15,
            'bidirectional': True,
            'use_attention': True,
            'use_residual': True
        },
        
        # Training parameters
        'num_epochs': 200,
        'batch_size': batch_size,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'optimizer': 'adamw',
        
        # Scheduler
        'scheduler': 'cosine',
        'scheduler_params': {
            'T_0': 20,
            'T_mult': 2,
            'eta_min': 1e-6,
            'factor': 0.5,
            'patience': 10
        },
        
        # Loss weights
        'loss_weights': {
            'differentiation': 1.0,
            'cell_type': 0.3,
            'smoothness': 0.1
        },
        
        # Regularization
        'max_grad_norm': 1.0,
        'mixed_precision': torch.cuda.is_available(),  # Enable if CUDA available
        'early_stopping_patience': 30,
        
        # Data
        'n_top_genes': 3000,
        'num_workers': min(8, os.cpu_count()),  # Use available CPU cores
        'pin_memory': torch.cuda.is_available(),
        
        # Logging
        'log_interval': 50,
        'save_interval': 10,
        'log_dir': 'experiments'
    }
    
    return config


def start_training():
    """Start the training process."""
    print("\\n🚀 Starting Advanced RNN Training")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Please fix issues before training.")
        return
    
    # Create optimal configuration
    config = create_optimal_config()
    
    # Save configuration
    config_path = Path("optimal_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📋 Configuration saved to: {config_path}")
    print(f"🔧 Batch size: {config['batch_size']}")
    print(f"🔧 Hidden dimension: {config['model']['hidden_dim']}")
    print(f"🔧 Attention heads: {config['model']['num_attention_heads']}")
    print(f"🔧 Mixed precision: {config['mixed_precision']}")
    
    # Construct training command
    data_path = "data/GSE175634_temporal_data"
    
    # Check if the data exists
    if not Path(data_path).exists():
        print(f"❌ Data not found at: {data_path}")
        print("Please ensure your temporal data is available.")
        return
    
    cmd = [
        sys.executable,  # Use current Python interpreter
        "train_advanced_rnn.py",
        "--data_path", data_path,
        "--config_path", str(config_path),
        "--gpu_id", "0"
    ]
    
    print(f"\\n🏃 Running command:")
    print(" ".join(cmd))
    
    # Ask for confirmation
    response = input("\\n❓ Start training? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Training cancelled.")
        return
    
    # Start training
    try:
        print("\\n🎯 Training started! Monitor progress in the terminal...")
        print("📊 Tensorboard logs will be saved in experiments/")
        print("💾 Model checkpoints will be saved automatically")
        print("\\n" + "=" * 50)
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code {e.returncode}")
        print("Check the error messages above for details.")
    except KeyboardInterrupt:
        print("\\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def prepare_data_first():
    """Prepare data before training."""
    print("\\n📋 Data Preparation")
    print("=" * 50)
    
    # Check if data preparation is needed
    temporal_data_path = Path("data/GSE175634_temporal_data")
    
    if temporal_data_path.exists():
        data_path = temporal_data_path
    else:
        print("❌ No temporal data found!")
        print("Please ensure you have downloaded and organized the GSE175634 data.")
        return False
    
    # Run data preparation
    cmd = [
        sys.executable,
        "prepare_temporal_data.py",
        "--data_path", str(data_path),
        "--output_summary", "data_summary.md"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Data preparation completed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Data preparation failed!")
        return False


def main():
    """Main function."""
    print("🧬 Advanced Temporal RNN Training - Quick Start")
    print("🚀 Optimized for NVIDIA RTX 5070")
    print("=" * 60)
    
    # Menu
    while True:
        print("\\n📋 What would you like to do?")
        print("1. 🔍 Check data and environment")
        print("2. 🚀 Start training immediately")
        print("3. 📊 Prepare data first (recommended)")
        print("4. ❌ Exit")
        
        choice = input("\\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            check_environment()
            
        elif choice == '2':
            start_training()
            break
            
        elif choice == '3':
            if prepare_data_first():
                response = input("\\n🚀 Data ready! Start training now? (y/N): ").strip().lower()
                if response in ['y', 'yes']:
                    start_training()
            break
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
