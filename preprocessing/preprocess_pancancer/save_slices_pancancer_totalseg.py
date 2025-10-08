import SimpleITK as sitk
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image


# Directory with CT nifty's
imagesTs_folder = "/data/groups/beets-tan/l.estacio/pancancer-nnunet/nnUNet_raw/Dataset105_PANC/imagesTs"
imagesTr_folder = "/data/groups/beets-tan/l.estacio/pancancer-nnunet/nnUNet_raw/Dataset105_PANC/imagesTr"

# Function file in in CT nifty folder that matches to segmentation file
def find_matching_file(id):
    # Search in imagesTs folder
    for root, dirs, files in os.walk(imagesTs_folder):
        for file in files:
            if file.startswith(id) and file.endswith("_0000.nii.gz"):
                return os.path.join(root, file)
    
    # Search in imagesTr folder
    for root, dirs, files in os.walk(imagesTr_folder):
        for file in files:
            if file.startswith(id) and file.endswith("_0000.nii.gz"):
                return os.path.join(root, file)
    
    # Return None if no matching file is found
    return None

# Directory with segmentations of tumors (NRRD files)
seg_dir = "/data/groups/beets-tan/k.groot.lipman/pancancer_seg_nrrd"
seg_files = sorted([file for file in os.listdir(seg_dir) if file.endswith(".nrrd")])


# Load totalsegmentator organ grouping
with open('totalseg_grouped.json') as f:
    organ_groups = json.load(f)
# Extract keys from the JSON file
organs = list(organ_groups.keys())

# Process all files in the pancancer dataset and store as slices. For each slice, store 2 arrays:
# 1) the CT scan and 2) the segmentations of tumors and organs (totalsegmentator output)
for seg_file in seg_files[110:]:
    print(seg_file)

    seg_id = seg_file.split('.')[0]
    seg_number = int(seg_id.split('_')[-1])
    seg_path = os.path.join(seg_dir, seg_file)
    seg_img = sitk.ReadImage(seg_path)

    # Find matching CT scan for segmentation file
    ct_path = find_matching_file(seg_id)
    try:
        ct_img = sitk.ReadImage(ct_path)
    # If file doesn't exist, continue
    except:
        with open('bad_files.txt', 'a') as file:
            message = f"Nifty file does not exist for {seg_id}\n"
            file.write(message)
            print("No nifty for", seg_file)
            continue

    # Find all labels in file
    hit1 = 'Segment._Name'
    hit2 = 'Segment.._Name'
    regex1 = re.compile(hit1)
    regex2 = re.compile(hit2)

    segment_names = [item for item in seg_img.GetMetaDataKeys() if (re.match(regex1, item)) and ('Auto' not in item)]
    segment_names2 = [item for item in seg_img.GetMetaDataKeys() if (re.match(regex2, item)) and ('Auto' not in item)]
    segment_names.extend(segment_names2)

    label_dict = {}
    seg_labels = []

    # Construct mapping of labels in the array to the actual tumor labels
    for item in segment_names:
        seg_label = int(seg_img.GetMetaData(item.replace('Name', 'LabelValue')))
        seg_labels.append(seg_label)
        med_label = int(float(seg_img.GetMetaData(item))) + 1
        # Label 1 and 9 both refer to a tumor that is not assigned to an organ
        if med_label == 9:
            med_label = 1
        label_dict[seg_label] = med_label

    # Load array with tumor segmentations and replace labels with actual tumor labels
    seg_array = sitk.GetArrayFromImage(seg_img)
    for old_value, new_value in label_dict.items():
        seg_array = np.where(seg_array == old_value, new_value, seg_array)
    
    # Remove extra dimension if layer of labels addeds
    if len(seg_array.shape) > 3:
        print("Shape mismatch so changed ", seg_file)
        seg_array = seg_array[:, :, :, 0]

    # Load the ct array
    ct_array = sitk.GetArrayFromImage(ct_img)

    # Construct segmentation array with totalseg labels and cancer labels
    # Labels 1 to 8 are for tumors, from then on for organs
    totalseg_path = "/data/groups/beets-tan/r.vt.hull/pancancer_totalseg_grouped"
    combined_seg = seg_array
    for idx, organ in enumerate(organs, start=9):
        try:
            totalseg_seg = sitk.ReadImage(os.path.join(totalseg_path, seg_id + "_0000", f"{organ}.nii.gz"))
            totalseg_array = sitk.GetArrayFromImage(totalseg_seg)
            combined_seg[(combined_seg == 0) & (totalseg_array > 0)] = idx
        except:
            print(f"{organ} does not exist")

    if ct_array.shape[0] == combined_seg.shape[0]:
        # Save each CT and segmentation slice as numpy array
        for j in range(ct_array.shape[0]):
            ct_slice = ct_array[j, :, :]
            seg_slice = combined_seg[j, :, :]
            np.save(f"/data/groups/beets-tan/r.vt.hull/Semantic_Segm/pancancer_totalseg/train/ct/{seg_id}_{j}.npy", ct_slice)
            np.save(f"/data/groups/beets-tan/r.vt.hull/Semantic_Segm/pancancer_totalseg/train/segmentations/{seg_id}_{j}.npy", seg_slice)
            # combined_seg_image = Image.fromarray(seg_slice.astype('uint8'))
            # combined_seg_image.save(f'test_img2/{j}_seg.png')
            # combined_seg_image = Image.fromarray(ct_slice.astype('uint8'))
            # combined_seg_image.save(f'test_img2/{j}_ct.png')
            # np.save(f"/processing/r.vt.hull/Semantic_Segm/pancancer/segmentations/{seg_id}_{j}.npy", ct_slice)
            # np.save(f"/processing/r.vt.hull/Semantic_Segm/pancancer/ct/{seg_id}_{j}.npy", seg_slice)
    
    # Don't save files if shapes of CT and segmentation don't match
    else:
        with open('bad_files.txt', "a") as output_file:
            print("Shape mismatch", seg_file)
            output_file.write(f"Could not save {seg_file} due to size mismatch in ct {ct_array.shape} and seg {seg_array.shape}\n")