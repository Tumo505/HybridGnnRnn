"""
Inspect Model Checkpoints and Extract Architecture Information
"""

import torch
import json
from datetime import datetime

def inspect_checkpoint(checkpoint_path):
    """Inspect a model checkpoint to understand its structure"""
    print(f"\n🔍 Inspecting: {checkpoint_path}")
    print("=" * 50)
    
    try:
        # Load with weights_only=False to handle older checkpoints
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            print("📋 Checkpoint Structure:")
            for key in checkpoint.keys():
                if key == 'model_state_dict' or key == 'state_dict':
                    print(f"   {key}: Model weights")
                    state_dict = checkpoint[key]
                elif isinstance(checkpoint[key], (int, float, str)):
                    print(f"   {key}: {checkpoint[key]}")
                else:
                    print(f"   {key}: {type(checkpoint[key])}")
            
            # Get state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        print("\n🏗️ Model Architecture (Layer shapes):")
        for name, param in state_dict.items():
            if hasattr(param, 'shape'):
                print(f"   {name}: {param.shape}")
            else:
                print(f"   {name}: {type(param)}")
                
        return state_dict
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def main():
    """Inspect available model checkpoints"""
    
    print("🧬 MODEL CHECKPOINT INSPECTION")
    print("=" * 60)
    
    # List of checkpoints to inspect
    checkpoints = [
        "best_regularized_model.pth",
        "comprehensive_cpu_results_20250921_173835/ablation_GNN_Only/best_model.pth",
        "comprehensive_cpu_results_20250921_173835/ablation_Full_Hybrid_Model/best_model.pth",
        "comprehensive_cpu_results_20250921_173835/fold_1/best_model.pth"
    ]
    
    inspection_results = {}
    
    for checkpoint_path in checkpoints:
        try:
            state_dict = inspect_checkpoint(checkpoint_path)
            if state_dict:
                # Extract key information
                layer_info = {}
                for name, param in state_dict.items():
                    if hasattr(param, 'shape'):
                        layer_info[name] = list(param.shape)
                
                inspection_results[checkpoint_path] = {
                    'layer_shapes': layer_info,
                    'total_parameters': sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel')),
                    'layer_count': len(state_dict)
                }
                
        except Exception as e:
            print(f"⚠️ Failed to inspect {checkpoint_path}: {e}")
    
    # Save inspection results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"checkpoint_inspection_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(inspection_results, f, indent=2)
    
    print(f"\n📁 Inspection results saved to: {results_file}")
    
    # Summary
    print("\n📊 SUMMARY:")
    for checkpoint_path, info in inspection_results.items():
        print(f"   {checkpoint_path}:")
        print(f"      Parameters: {info['total_parameters']:,}")
        print(f"      Layers: {info['layer_count']}")

if __name__ == "__main__":
    main()