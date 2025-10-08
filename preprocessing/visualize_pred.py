import numpy as np
import matplotlib.pyplot as plt
import os
# Load the data
id = 124
num = 100
ct_scan = np.load(f'/home/r.vt.hull/data/Semantic_Segm/totalseg/val/ct/s0{id}_{num}.npy')
predicted_masks = np.load(f'/home/r.vt.hull/data/Semantic_Segm/totalseg/test_arrays/predicted_masks/s0{id}_{num}_pred.npy')
actual_masks = np.load(f'/home/r.vt.hull/data/Semantic_Segm/totalseg/test_arrays/gt_masks/s0{id}_{num}_pred.npy')
actual_masks2 = np.load(f'/home/r.vt.hull/data/Semantic_Segm/totalseg/val/combined_seg/s0{id}_{num}_combined.npy')

# Visualize a sample slice
# slice_idx = 50  # Change this index to visualize different slices
# Create a color map for masks

print(np.unique(predicted_masks))
print(np.unique(actual_masks))
print(np.unique(actual_masks2))

# Visualize a sample slice
plt.figure(figsize=(15, 5))

# CT Scan
plt.subplot(131)
plt.imshow(ct_scan, cmap='bone')
plt.title('CT Scan')

# Predicted Mask
plt.subplot(132)
plt.imshow(predicted_masks, cmap='viridis')
plt.title('Predicted Mask')

# Actual Mask
plt.subplot(133)
plt.imshow(actual_masks, cmap='viridis')
plt.title('Actual Mask')

# plt.subplot(133)
# plt.imshow(actual_masks2, cmap='viridis')
# plt.title('Actual Mask 2')

plt.tight_layout()

# Save the figure
plt.savefig('visualization.png')  # Change the format if needed (e.g., .jpg, .pdf)
plt.show()



folder_path = '/home/r.vt.hull/data/Semantic_Segm/totalseg/val/combined_seg'

# List all files in the folder
files = os.listdir(folder_path)

print(files)

# Iterate through each file in the folder
# Iterate through each file in the folder
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    
    # Check if the file is an npy file
    if file_name.endswith('.npy'):
        # Load the array from the npy file
        array = np.load(file_path)
        
        # Get unique values from the array
        unique_values = np.unique(array)
        
        # Print unique values for the current array
        print(f"Unique values in '{file_name}': {unique_values}")
