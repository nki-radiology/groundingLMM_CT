import os

ct_folder = '/home/r.vt.hull/data/Semantic_Segm/totalseg/val/ct'
test_arrays_folder = '/home/r.vt.hull/data/Semantic_Segm/totalseg/test_arrays/gt_masks'

ct_files = sorted(os.listdir(ct_folder))
test_arrays_files = sorted(os.listdir(test_arrays_folder))

for ct_file, test_file in zip(ct_files, test_arrays_files):
    if ct_file.endswith('.npy') and test_file.endswith('.npy'):
        new_test_name = f"{ct_file.split('.')[0]}_pred.npy"
        os.rename(os.path.join(test_arrays_folder, test_file), os.path.join(test_arrays_folder, new_test_name))
        print(f"Renamed {test_file} to {new_test_name}")