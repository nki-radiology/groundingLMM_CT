import SimpleITK as sitk
import numpy as np
import os
import json

# data path and all patient ids
data_path = "/data/groups/beets-tan/_public/Totalsegmentator_v201/"
ids = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

# number of patients to process
num_test_patients = 100
num_val_patients = 10

with open('totalseg_grouped.json', 'r') as file:
    totalseg_grouped = json.load(file)

for i in range(len(ids)):
    id = ids[i]
    id_path = data_path + str(id) + "/segmentations/"
    
    # Create a new directory for grouped segmentations
    grouped_path = os.path.join(data_path, str(id), "grouped_segmentations")
    os.makedirs(grouped_path, exist_ok=True)

    # Concatenate files for each organ and store as a total organ name
    for organ_name, organ_files_list in totalseg_grouped.items():
        try:
            concat_organ = sitk.ReadImage(os.path.join(id_path, f"{organ_files_list[0]}.nii.gz"))
            for organ_part in organ_files_list[1:]:
                concat_organ = concat_organ + sitk.ReadImage(os.path.join(id_path, f"{organ_part}.nii.gz"))
            
            # Save the combined segmentation in the grouped directory
            sitk.WriteImage(concat_organ, os.path.join(grouped_path, f"{organ_name}.nii.gz"))
        
        # Sometimes a file's metadata is corrupted
        except Exception as e:
            print(f"Error processing patient {id}")
            continue
