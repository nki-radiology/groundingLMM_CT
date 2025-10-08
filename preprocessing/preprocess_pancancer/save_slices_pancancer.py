import SimpleITK as sitk
import re
import os
import numpy as np
import matplotlib.pyplot as plt


# Direcotory with 
imagesTs_folder = "/data/groups/beets-tan/l.estacio/pancancer-nnunet/nnUNet_raw/Dataset105_PANC/imagesTs"
imagesTr_folder = "/data/groups/beets-tan/l.estacio/pancancer-nnunet/nnUNet_raw/Dataset105_PANC/imagesTr"

# Function to find matching files in both folders
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

# Directory with segmentations
seg_dir = "/data/groups/beets-tan/k.groot.lipman/pancancer_seg_nrrd"
# ct_dir = "/data/groups/beets-tan/l.estacio/pancancer-nnunet/nnUNet_raw/Dataset105_PANC"

seg_files = sorted([file for file in os.listdir(seg_dir) if file.endswith(".nrrd")])


for seg_file in seg_files:
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

    # Replace labels in file with actual labels
    for item in segment_names:
        seg_label = int(seg_img.GetMetaData(item.replace('Name', 'LabelValue')))
        seg_labels.append(seg_label)
        med_label = int(float(seg_img.GetMetaData(item)))
        if med_label == 8:
            med_label = 0

        label_dict[seg_label] = med_label
    
    seg_array = sitk.GetArrayFromImage(seg_img)
    
    # Remove extra dimension if layer of labels addeds
    if len(seg_array.shape) > 3:
        print("Shape mismatch so changed ", seg_file)
        seg_array = seg_array[:, :, :, 0]

    ct_array = sitk.GetArrayFromImage(ct_img)

    if ct_array.shape[0] == seg_array.shape[0]:
        # Save each CT and segmentation slice as numpy array
        for j in range(ct_array.shape[0]):
            ct_slice = ct_array[j, :, :]
            seg_slice = seg_array[j, :, :]
            for old_value, new_value in label_dict.items():
                seg_slice = np.where(seg_slice == old_value, new_value, seg_slice)
            np.save(f"/data/groups/beets-tan/r.vt.hull/Semantic_Segm/pancancer/ct/{seg_id}_{j}.npy", ct_slice)
            np.save(f"/data/groups/beets-tan/r.vt.hull/Semantic_Segm/pancancer/segmentations/{seg_id}_{j}.npy", seg_slice)
            # np.save(f"/processing/r.vt.hull/Semantic_Segm/pancancer/segmentations/{seg_id}_{j}.npy", ct_slice)
            # np.save(f"/processing/r.vt.hull/Semantic_Segm/pancancer/ct/{seg_id}_{j}.npy", seg_slice)
    
    # Don't save files if shapes of CT and segmentation don't match
    else:
        with open('bad_files.txt', "a") as output_file:
            print("Shape mismatch", seg_file)
            output_file.write(f"Could not save {seg_file} due to size mismatch in ct {ct_array.shape} and seg {seg_array.shape}\n")