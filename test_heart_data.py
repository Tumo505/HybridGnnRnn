import sys
sys.path.append('src')

from data.process_heart_data import create_optimized_heart_dataset

print('🫀 Processing heart datasets...')

try:
    heart_data = create_optimized_heart_dataset()
    
    if heart_data is not None:
        print('\n🎉 SUCCESS! Heart dataset processed.')
        print(f'Shape: {heart_data.shape}')
        print(f'Spatial: {"Yes" if "spatial" in heart_data.obsm else "No"}')
        print(f'Cardiac markers: {heart_data.uns.get("n_cardiac_markers", 0)}')
        
        # Show cardiac genes
        if hasattr(heart_data, 'var') and 'cardiac_marker' in heart_data.var.columns:
            cardiac_genes = heart_data.var_names[heart_data.var['cardiac_marker']].tolist()
            if cardiac_genes:
                print(f'Cardiac genes: {cardiac_genes}')
        
        print('\n🎯 HEART-SPECIFIC dataset ready for training!')
        
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
