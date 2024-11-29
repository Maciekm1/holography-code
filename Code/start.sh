#!/bin/bash

# Exit on error
set -e

# Purge existing modules and load required ones
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load tqdm/4.64.0-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
module load scikit-image/0.19.3-foss-2022a
module load OpenCV/4.6.0-foss-2022a-contrib
module load openpyxl/3.0.10-GCCcore-11.3.0

echo "Modules loaded successfully..."
echo ""

# Capture the output of the Python script (directory path)
output_folder_frames=$(python csv_to_parameter_file.py "$1" "$2")

# Echo the folder path to confirm it
echo "Python script returned: $output_folder_frames"

# Use the captured output as a parameter for the next command
if [[ -d "$output_folder_frames" ]]; then
    echo "Directory exists: $output_folder_frames"
    
    # Run the update_job_array.sh script with the output directory path
    ./update_job_array.sh "$output_folder_frames"
    
    # Ask the user if they want to monitor the job queue
    read -p "Do you want to monitor the job queue with 'watch'? (y/n): " user_input
    if [[ "$user_input" == "y" || "$user_input" == "Y" ]]; then
        # Run 'watch' to monitor the job queue
        echo "Monitoring job queue..."
        watch -n 10 squeue --me
    else
        echo "Skipping job queue monitoring."
    fi

else
    echo "Error: Directory does not exist."
    exit 1
fi

