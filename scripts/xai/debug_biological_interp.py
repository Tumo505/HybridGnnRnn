#!/usr/bin/env python3
"""
Debug script to check biological interpretations
"""

import numpy as np
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from explainable_ai.hybrid_model_xai import BiologicalInterpreter

def debug_biological_interpretations():
    """Debug why biological interpretations are empty"""
    
    # Load the SHAP results from JSON
    with open('results/xai_analysis/xai_analysis_results_20250921_232550.json', 'r') as f:
        data = json.load(f)
    
    print("🔍 Debugging biological interpretations...")
    
    # Check if biological interpretations exist in results
    if 'biological_interpretations' in data:
        print(f"✅ Biological interpretations found: {len(data['biological_interpretations'])} items")
        for i, interp in enumerate(data['biological_interpretations'][:3]):
            print(f"   {i+1}. {interp}")
    else:
        print("❌ No biological interpretations found in results")
    
    # Check the actual SHAP structure
    shap_data = data['shap_analysis']
    print(f"\n📊 SHAP data structure:")
    for key, value in shap_data.items():
        if isinstance(value, list):
            print(f"   {key}: list with {len(value)} items")
            if key == 'mean_shap_values' and len(value) > 0:
                print(f"     First item: {value[0]}")
                print(f"     Type of first item: {type(value[0])}")
                if isinstance(value[0], list):
                    print(f"     Classes per feature: {len(value[0])}")
        else:
            print(f"   {key}: {type(value)} = {value}")
    
    # The mean_shap_values is stored as [feature0_classes, feature1_classes, ...]
    feature_names = shap_data['feature_names'] 
    shap_per_feature = shap_data['mean_shap_values']  # List of [class0, class1, class2, class3] per feature
    
    print(f"\n📊 Processed analysis:")
    print(f"   Features: {len(feature_names)}")
    print(f"   SHAP values per feature: {len(shap_per_feature)}")
    
    # Calculate average absolute importance across all classes for each feature
    feature_importance = []
    for feature_shaps in shap_per_feature:
        if isinstance(feature_shaps, list):
            # Average absolute value across classes
            avg_abs_importance = np.mean(np.abs(feature_shaps))
        else:
            avg_abs_importance = abs(feature_shaps)
        feature_importance.append(avg_abs_importance)
    
    feature_importance = np.array(feature_importance)
    print(f"   Importance range: {feature_importance.min():.6f} to {feature_importance.max():.6f}")
    
    # Get top features
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    
    print(f"\n🔝 Top 10 features by absolute importance:")
    for i, idx in enumerate(top_indices):
        print(f"   {i+1}. {feature_names[idx]}: {feature_importance[idx]:.6f}")
    
    # Test biological interpretation with mean_shap_values directly
    print(f"\n🧬 Testing biological interpretation...")
    bio_interpreter = BiologicalInterpreter()
    
    # Use the mean_shap_values directly (features x classes format)
    shap_results = {
        'feature_names': feature_names,
        'shap_values': shap_data['mean_shap_values']  # Pass the mean values directly
    }
    
    interpretations = bio_interpreter.interpret_feature_importance(shap_results, top_k=10)
    
    print(f"   Generated interpretations: {len(interpretations)}")
    for i, interp in enumerate(interpretations[:10]):
        print(f"   {i+1}. {interp['feature_name']}: {interp['importance_score']:.6f} - {interp['biological_marker']}")

if __name__ == "__main__":
    debug_biological_interpretations()