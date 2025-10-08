import os
import json

# Load the JSON file
with open('totalseg_grouped.json', 'r') as json_file:
    data = json.load(json_file)

# Define the folder path

data_path = "/data/groups/beets-tan/_public/Totalsegmentator_v201/"
ids = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])


# Iterate through the keys in the dictionary
for i in range(len(ids)):
    id = ids[i]
    for key, values in data.items():
        # Check if the length of values is greater than 1
        if len(values) > 1:
            # Delete corresponding files in the folder
            for value in values:
                file_path = os.path.join(data_path, id, 'segmentations', f'{key}.nii.gz')
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f'Deleted: {file_path}')
                else:
                    print(f'File not found: {file_path}')
