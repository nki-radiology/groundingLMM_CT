import SimpleITK as sitk
import numpy as np
import os
import json

# data path and all patient ids
input_path = "/data/groups/beets-tan/r.vt.hull/pancancer_totalseg2"
output_path = "/data/groups/beets-tan/r.vt.hull/pancancer_totalseg_grouped"
ids = sorted([d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))])

# number of patients to process
num_cts = 10

with open('totalseg_grouped.json', 'r') as file:
    totalseg_grouped = json.load(file)

for i in range(110, len(ids)):
    id = ids[i]
    print(id)
    id_path = os.path.join(input_path, id)
    

    # Create a new directory for grouped segmentations
    grouped_path = os.path.join(output_path, id)
    if not os.path.exists(grouped_path):
        os.makedirs(grouped_path)
    else:
        continue  # Skip to the next iteration if directory already exists


    # Concatenate files for each organ and store as a total organ name
    for organ_name, organ_files_list in totalseg_grouped.items():
        try:
            concat_organ = sitk.ReadImage(os.path.join(id_path, f"{organ_files_list[0]}.nii.gz"))
            for organ_part in organ_files_list[1:]:
                concat_organ = concat_organ + sitk.ReadImage(os.path.join(id_path, f"{organ_part}.nii.gz"))
        
            # Save the combined segmentation in the grouped directory
            sitk.WriteImage(concat_organ, os.path.join(grouped_path, f"{organ_name}.nii.gz"))
        except:
            print(f"organ {organ_part} is loessoe")