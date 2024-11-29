#!/bin/bash

# Get the input directory as the first argument
dir=$1

# Count the number of files in the directory
file_count=$(ls "$dir" | wc -l)
echo "File count: $file_count"
echo ""

# Update the #SBATCH --array line with the correct range
sed -i "s/^#SBATCH --array=.*/#SBATCH --array=0-$(($file_count - 1))/g" my_job_array.job

# Remove the /inputs part from the directory path
base_dir="${dir%/inputs}"

# Update the input_dir variable in the .job file
sed -i "s|^input_dir=.*|input_dir=\"./$dir\"|g" my_job_array.job

# Update the param_file variable in the .job file
# Ensure we only append parameters.json to the directory path
sed -i "s|^param_file=.*|param_file=\"./$base_dir/parameters.json\"|g" my_job_array.job

# Optionally, print the updated .job file for verification
echo "▼▼▼ MAKE SURE JOB ARRAY AND INPUT VARIABLES ARE CORRECT ▼▼▼"
cat my_job_array.job
echo ""  # Blank line for readability

# Ask for confirmation before submitting the job
read -p "Are you sure you want to submit this job? [y/n]: " confirmation

if [[ "$confirmation" == "y" || "$confirmation" == "Y" ]]; then
    sbatch my_job_array.job
    echo "Job submitted successfully."
else
    echo "Job submission canceled."
fi

