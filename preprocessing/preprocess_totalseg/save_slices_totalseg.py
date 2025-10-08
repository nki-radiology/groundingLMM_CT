# Saves each slice of a nifty CT scan in array form
# Combines segmentations of different organs

import SimpleITK as sitk
import numpy as np
import os
import json

data_path = "/data/groups/beets-tan/_public/Totalsegmentator_v201/"
ids = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

with open('totalseg_grouped.json') as f:
    organ_groups = json.load(f)

# Extract keys from the JSON file
organs = list(organ_groups.keys())


for i in range(len(ids)):
    id = ids[i]
    try:
        ct_segs_path = os.path.join(data_path, str(id), "grouped_segmentations/")
        ct = sitk.ReadImage(os.path.join(data_path, str(id), "ct.nii.gz"))
    except:
        print(f"id {id} not valid")
        continue
        
    ct_array = sitk.GetArrayFromImage(ct)
    combined_seg = np.zeros_like(ct_array, dtype=np.uint8)

    for idx, organ in enumerate(organs, start=1):
        try:
            seg = sitk.ReadImage(os.path.join(ct_segs_path, f"{organ}.nii.gz"))
            seg_array = sitk.GetArrayFromImage(seg)
            combined_seg[seg_array > 0] = idx
        except:
            print(f"{organ} does not exist")
    
    for j in range(ct_array.shape[0]):
        combined_seg_slice = np.rot90(combined_seg[j, :, :], k=2)

        # np.save(f"/data/groups/beets-tan/r.vt.hull/Semantic_Segm/totalseg/{mode}/combined_seg/{str(id)}_{j}_combined.npy", combined_seg_slice)
        np.save(f"/data/groups/beets-tan/_public/totalsegmentator_slices/segmentations/{str(id)}_seg_{j}.npy", combined_seg_slice)

        ct_array_slice =  np.rot90(ct_array[j, :, :], k=2)
        np.save(f"/data/groups/beets-tan/_public/totalsegmentator_slices/ct/{str(id)}_{j}.npy", ct_array_slice)