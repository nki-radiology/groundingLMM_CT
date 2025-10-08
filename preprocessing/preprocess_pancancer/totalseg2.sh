#!/bin/bash

#SBATCH -t 0-20:00:00
#SBATCH --partition=p6000
#SBATCH --job-name="run_totalseg2"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH -o /home/r.vt.hull/joblogs/totalseg_Tr_%A_%a.out     # output file (what gets printed normally)
#SBATCH -e /home/r.vt.hull/joblogs/totalseg_Tr_%A_%a.err    # error file (check this for warnings/errors)

# Activate conda environment
source /home/r.vt.hull/miniconda3/bin/activate totalseg_env

# Set the directory containing NIfTI files
nifti_directory="/data/groups/beets-tan/l.estacio/pancancer-nnunet/nnUNet_raw/Dataset105_PANC/imagesTr"

# Move to the directory with NIfTI files
cd "$nifti_directory"

# Create output base directory for segmentations in the home directory
output_base_directory="/data/groups/beets-tan/r.vt.hull/pancancer_totalseg2"
mkdir -p "$output_base_directory"

# Loop through all NIfTI files in the current directory
for input_file in *_0000.nii.gz; do
    # Extract the file name without extension
    file_name=$(basename "${input_file%.nii.gz}")
    
    # Create a new directory for each file in the segmentations folder if it doesn't already exist
    output_directory="$output_base_directory/${file_name}"
    if [ -d "$output_directory" ]; then
        echo "Output directory already exists, skipping: $file_name"
        continue
    fi
    
    mkdir -p "$output_directory"
    
    # Run TotalSegmentator for each file
    TotalSegmentator -i "$input_file" -o "$output_directory"
done
