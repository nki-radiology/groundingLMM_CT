import os
import numpy as np

def transform_labels(label):
    if 3 <= label <= 8:
        return 1
    elif label > 8:
        return label - 6
    else:
        return label

def process_arrays_in_directory(directory, new_directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".npy"):
                # Load the array
                old_array_path = os.path.join(root, filename)
                array = np.load(old_array_path)
                
                # Transform the labels
                vectorized_transform = np.vectorize(transform_labels)
                transformed_array = vectorized_transform(array)
                
                # Determine the new path
                relative_path = os.path.relpath(root, directory)
                new_dir_path = os.path.join(new_directory, relative_path)
                new_array_path = os.path.join(new_dir_path, filename)
                
                # Create the new directory if it doesn't exist
                os.makedirs(new_dir_path, exist_ok=True)
                
                # Save the transformed array to the new path
                np.save(new_array_path, transformed_array)
                print(f"Processed {filename}")

# Directory containing the arrays
directory = "/data/groups/beets-tan/r.vt.hull/Semantic_Segm/pancancer_totalseg/train/segmentations"
save_dir = "/data/groups/beets-tan/r.vt.hull/Semantic_Segm/pancancer_totalseg/train/segmentations2"

# Process the arrays
process_arrays_in_directory(directory, save_dir)
