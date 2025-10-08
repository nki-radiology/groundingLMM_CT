import SimpleITK as sitk
import re
import os
import numpy as np
import matplotlib.pyplot as plt


# Path to the folders
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

# Assuming you have a file named 'your_array.npy'
path = '/data/groups/beets-tan/k.groot.lipman/pancancer_seg_nrrd/'

file_names = ['PANCANCER_0052.nrrd']
# file_names = [
#     "PANCANCER_0008.nrrd",
#     "PANCANCER_0030.nrrd",
#     "PANCANCER_0047.nrrd",
#     "PANCANCER_0184.nrrd",
#     "PANCANCER_0190.nrrd",
#     "PANCANCER_0198.nrrd",
#     "PANCANCER_0219.nrrd",
#     "PANCANCER_0239_1.nrrd",
#     "PANCANCER_0246.nrrd",
#     "PANCANCER_0264.nrrd",
#     "PANCANCER_0328.nrrd",
#     "PANCANCER_0447.nrrd",
#     "PANCANCER_0584.nrrd",
#     "PANCANCER_0650.nrrd",
#     "PANCANCER_0890.nrrd",
#     "PANCANCER_0891.nrrd",
#     "PANCANCER_0893.nrrd",
#     "PANCANCER_0895.nrrd",
#     "PANCANCER_0896.nrrd",
#     "PANCANCER_0914.nrrd",
#     "PANCANCER_0924.nrrd",
#     "PANCANCER_0955.nrrd",
#     "PANCANCER_1032.nrrd",
#     "PANCANCER_1070.nrrd",
#     "PANCANCER_1159.nrrd",
#     "PANCANCER_1193.nrrd",
#     "PANCANCER_1277.nrrd"
# ]
file_names=[
    "PANCANCER_0052.nrrd",
    "PANCANCER_0067.nrrd",
    "PANCANCER_0080.nrrd",
    "PANCANCER_0112.nrrd",
    "PANCANCER_0116.nrrd",
    "PANCANCER_0121.nrrd",
    "PANCANCER_0132.nrrd",
    "PANCANCER_0198.nrrd",
    "PANCANCER_0008.nrrd",
    "PANCANCER_0030.nrrd",
    "PANCANCER_0047.nrrd",
    "PANCANCER_0052.nrrd",
    "PANCANCER_0067.nrrd",
    "PANCANCER_0080.nrrd",
    "PANCANCER_0112.nrrd",
    "PANCANCER_0116.nrrd",
    "PANCANCER_0121.nrrd",
    "PANCANCER_0132.nrrd",
    "PANCANCER_0184.nrrd",
    "PANCANCER_0190.nrrd",
    "PANCANCER_0198.nrrd",
    "PANCANCER_0219.nrrd"
]

for seg_file in file_names:
    # Load the array from the file
    seg_path = os.path.join(path, seg_file)
    seg_img = sitk.ReadImage(seg_path)
    seg_array = sitk.GetArrayFromImage(seg_img)

    seg_id = seg_file.split('.')[0]
    ct_path = find_matching_file(seg_id)
    ct_img = sitk.ReadImage(ct_path)
    ct_array = sitk.GetArrayFromImage(ct_img)

    hit1 = 'Segment._Name'
    hit2 = 'Segment.._Name'
    regex1 = re.compile(hit1)
    regex2 = re.compile(hit2)


    # Accessing a slice along the fourth dimension
    print(seg_array.shape)
    print(ct_array.shape)
    # slice1 = seg_array[:, :, :, 0]  # Replace 0 with the index you want
    # slice2 = seg_array[:, :, :, 1]
    # # slice3 = seg_array[:, :, :, 2]

    # print(np.unique(slice1))
    # print(np.unique(slice2))
    # # print(np.unique(slice3))

    # # # count_of_nrs = np.count_nonzero(slice1==1)
    # # for num in range(11):
    # #     print(num, np.count_nonzero(slice1==num))
    # count_of_ones = np.count_nonzero(slice2 == 1)
    # # total_length = slice2.size

    # # print("Total length of the array:", total_length)

    # # # print("Number of occurrences of nrs:", count_of_nrs)
    # print("Number of occurrences of 1:", count_of_ones)