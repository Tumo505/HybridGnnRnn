"""
Test XAI Visualizations Content
"""

import json
import matplotlib.pyplot as plt
import os

# Load the latest results - get path relative to script location
results_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'xai_analysis', 'xai_analysis_results_20250921_234737.json')
with open(results_path, 'r') as f:
    results = json.load(f)

print("🔍 Testing XAI visualization data...")

# Check biological interpretations
bio_interps = results.get('biological_interpretations', [])
print(f"✅ Biological interpretations: {len(bio_interps)} items")

if bio_interps:
    print("\n🔬 Top 5 biological interpretations:")
    for i, interp in enumerate(bio_interps[:5]):
        print(f"   {i+1}. {interp['feature_name']}: {interp['importance_score']:.6f} - {interp['biological_marker']}")
    
    # Test creating a simple plot
    feature_names = [interp['feature_name'] for interp in bio_interps[:10]]
    importance_scores = [interp['importance_score'] for interp in bio_interps[:10]]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importance_scores)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Importance Score')
    plt.title('Top 10 Features by Importance')
    plt.tight_layout()
    plt.savefig('test_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Test visualization saved as test_feature_importance.png")
    print(f"   Data ranges from {min(importance_scores):.6f} to {max(importance_scores):.6f}")
else:
    print("❌ No biological interpretations found!")

# Check other data
print(f"\n📊 Other XAI components:")
print(f"   SHAP analysis: {'✅' if 'shap_analysis' in results else '❌'}")
print(f"   Uncertainty analysis: {'✅' if 'uncertainty_analysis' in results else '❌'}")
print(f"   Experimental suggestions: {len(results.get('experimental_suggestions', []))} items")

print("\n🎉 Visualization test completed!")