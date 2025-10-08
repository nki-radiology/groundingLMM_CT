# Saves each slice of a nifty CT scan in array form
# Saves segmentations of organs seperately

import SimpleITK as sitk
import numpy as np
import os
import cv2

data_path = "/data/groups/beets-tan/_public/Totalsegmentator_v201/"
ids = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

segs_to_save = ["heart", "lungs", "liver", "brain"]

num_patients = 10
num_ids = range(num_patients)

for i in num_ids:
    id = ids[i]
    try:
        ct_segs_path = data_path + str(id) + "/segmentations/"
        ct = sitk.ReadImage(data_path + str(id) + "/ct.nii.gz")
    except:
        print(f"id {id} not valid")
        continue
        
    ct_array = sitk.GetArrayFromImage(ct)

    segs = [sitk.ReadImage(ct_segs_path + seg + ".nii.gz") for seg in segs_to_save]
    seg_arrays = [sitk.GetArrayFromImage(seg) for seg in segs]
    
    for j in range(ct_array.shape[0]):
        ct_array_slice =  np.rot90(ct_array[j, :, :], k=2)
        np.save(f"ct/{str(id)}_{j}.npy", ct_array_slice)

        for seg_name, seg_array in zip(segs_to_save, seg_arrays):
            seg_array_slice = np.rot90(seg_array[j, :, :], k=2)
            np.save(f"seg/{str(id)}_{j}_{seg_name}.npy", seg_array_slice)


