#!/bin/bash


#SBATCH -t 0-20:00:00
#SBATCH --partition=p6000
#SBATCH --job-name="run_totalseg2"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH -o /home/r.vt.hull/joblogs/Z_%A_%a.out
#SBATCH -e /home/r.vt.hull/joblogs/Z_%A_%a.err

# Activate conda environment

source /home/r.vt.hull/miniconda3/bin/activate llava



# # Set the directory containing NIfTI files
# nifti_directory="/data/groups/beets-tan/l.estacio/pancancer-nnunet/nnUNet_raw/Dataset105_PANC/imagesTs"

# # Move to the directory with NIfTI files
# cd "$nifti_directory"

# Create output base directory for segmentations in the home directory
output_base_directory="/data/groups/beets-tan/r.vt.hull/pancancer_totalseg2"
# mkdir -p "$output_base_directory"

start_processing=false

# Loop through all NIfTI files in the current directory
for input_file in *_0000.nii.gz; do
    echo "$input_file"
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

TotalSegmentator -i /data/groups/public/archive/radiology/Totalsegmentator_v201/s0747/ct.nii.gz -o /data/groups/beets-tan/r.vt.hull/totalseg_testing/s0747