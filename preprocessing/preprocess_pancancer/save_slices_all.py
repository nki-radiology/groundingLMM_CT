import SimpleITK as sitk
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

with open('/home/r.vt.hull/save_slices.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()

# Strip any leading/trailing whitespace characters and create a list
file_names = [line.strip() for line in lines]

organs = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "kidney_right",
    "kidney_left",
    "rib_left_1",
    "rib_left_2",
    "rib_left_3",
    "rib_left_4",
    "rib_left_5",
    "rib_left_6",
    "rib_left_7",
    "rib_left_8",
    "rib_left_9",
    "rib_left_10",
    "rib_left_11",
    "rib_left_12",
    "rib_right_1",
    "rib_right_2",
    "rib_right_3",
    "rib_right_4",
    "rib_right_5",
    "rib_right_6",
    "rib_right_7",
    "rib_right_8",
    "rib_right_9",
    "rib_right_10",
    "rib_right_11",
    "rib_right_12",
    "vertebrae_S1",
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
    "vertebrae_T12",
    "vertebrae_T11",
    "vertebrae_T10",
    "vertebrae_T9",
    "vertebrae_T8",
    "vertebrae_T7",
    "vertebrae_T6",
    "vertebrae_T5",
    "vertebrae_T4",
    "vertebrae_T3",
    "vertebrae_T2",
    "vertebrae_T1",
    "vertebrae_C7",
    "vertebrae_C6",
    "vertebrae_C5",
    "vertebrae_C4",
    "vertebrae_C3",
    "vertebrae_C2",
    "vertebrae_C1",
    "spleen",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "esophagus",
    "trachea",
    "thyroid_gland",
    "small_bowel",
    "duodenum",
    "colon",
    "urinary_bladder",
    "prostate",
    "kidney_cyst_left",
    "kidney_cyst_right",
    "sacrum",
    "heart",
    "aorta",
    "pulmonary_vein",
    "brachiocephalic_trunk",
    "subclavian_artery_right",
    "subclavian_artery_left",
    "common_carotid_artery_right",
    "common_carotid_artery_left",
    "brachiocephalic_vein_left",
    "brachiocephalic_vein_right",
    "atrial_appendage_left",
    "superior_vena_cava",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
    "humerus_left",
    "humerus_right",
    "scapula_left",
    "scapula_right",
    "clavicula_left",
    "clavicula_right",
    "femur_left",
    "femur_right",
    "hip_left",
    "hip_right",
    "spinal_cord",
    "gluteus_maximus_left",
    "gluteus_maximus_right",
    "gluteus_medius_left",
    "gluteus_medius_right",
    "gluteus_minimus_left",
    "gluteus_minimus_right",
    "autochthon_left",
    "autochthon_right",
    "iliopsoas_left",
    "iliopsoas_right",
    "brain",
    "skull",
    "sternum",
    "costal_cartilages"
]


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
report_dir = "/data/groups/beets-tan/r.vt.hull/llama3_labels_short"
reports= os.listdir(report_dir)
seg_files = sorted([file for file in os.listdir(seg_dir) if file.endswith(".nrrd")])

# Load totalsegmentator organ grouping
# with open('totalseg_grouped.json') as f:
#     organ_groups = json.load(f)
# Extract keys from the JSON file
# organs = list(organ_groups.keys())

# Process all files in the pancancer dataset and store as slices. For each slice, store 2 arrays:
# 1) the CT scan and 2) the segmentations of tumors and organs (totalsegmentator output)

todo_files = [file for file in seg_files if file not in file_names and file.rsplit('.', 1)[0] + ".txt" in reports]

for seg_file in todo_files:
    print(seg_file, flush=True)

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
    totalseg_path = "/data/groups/beets-tan/r.vt.hull/pancancer_totalseg2"
    combined_seg = seg_array
    for idx, organ in enumerate(organs, start=10):
        try:
            totalseg_seg = sitk.ReadImage(os.path.join(totalseg_path, seg_id + "_0000", f"{organ}.nii.gz"))
            totalseg_array = sitk.GetArrayFromImage(totalseg_seg)
            combined_seg[(combined_seg == 0) & (totalseg_array > 0)] = idx
        except:
            print(f"{organ} does not exist")

    if ct_array.shape[0] == combined_seg.shape[0]:
        # Save each CT and segmentation slice as numpy array
        for j in range(ct_array.shape[0]):
            # ct_slice = ct_array[j, :, :]
            seg_slice = combined_seg[j, :, :]
            # np.save(f"/data/groups/beets-tan/r.vt.hull/Semantic_Segm/pancancer_totalseg/train/ct/{seg_id}_{j}.npy", ct_slice)
            np.save(f"/data/groups/beets-tan/r.vt.hull/Semantic_Segm/pancancer_totalseg/train/segmentations_full/{seg_id}_{j}.npy", seg_slice)
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