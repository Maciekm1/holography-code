#!/bin/bash

# Exit on error
set -e

module purge

module load Python/3.10.4-GCCcore-11.3.0
module load tqdm/4.64.0-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
module load scikit-image/0.19.3-foss-2022a
module load OpenCV/4.6.0-foss-2022a-contrib
module load openpyxl/3.0.10-GCCcore-11.3.0

echo "Modules loaded successfully..."
echo ""

# Capture the output of the Python script
output_folder_frames=$(python csv_to_parameter_file.py $1 $2)

# Echo the folder path to confirm it
echo "Python script returned: $output_folder_frames"

# Use the captured output as a parameter for the next command
if [[ -d "$output_folder_frames" ]]; then
    echo "Directory exists: $output_folder_frames"
    # Add additional commands using $output_folder_frames here
    ./update_job_array.sh "$output_folder_frames"
else
    echo "Error: Directory does not exist."
    exit 1
fi
