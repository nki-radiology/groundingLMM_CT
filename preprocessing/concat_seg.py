#    Concatenates the lung segmentations into a single segmentations.
#    Other organs could be added.


import SimpleITK as sitk
import numpy as np
import os

# data path and all patient ids
data_path = "/data/groups/beets-tan/_public/Totalsegmentator_v201/"
ids = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])


# number of patients to process
num_patients = 10

# all files to concatenate, per organ
to_concat = [["lung_lower_lobe_left.nii.gz", "lung_lower_lobe_right.nii.gz", "lung_middle_lobe_right.nii.gz", "lung_upper_lobe_left.nii.gz", "lung_upper_lobe_right.nii.gz"]]

# names of organs to concatenate
concat_names = ["lungs.nii.gz"]

for i in range(num_patients,200):
    id = ids[i]
    id_path = data_path + str(id) + "/segmentations/"

    # concatenate all organ files of patient and store as total organ name
    for organ_name, organ in zip(concat_names, to_concat):
        concat_organ = sitk.ReadImage(id_path + organ[0])
        for organ_part in organ[1:]:
            concat_organ = concat_organ + sitk.ReadImage(id_path + organ_part)
        # Save the combined segmentation
        sitk.WriteImage(concat_organ, id_path + organ_name)