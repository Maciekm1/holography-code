import os
import sys
import json
from pathlib import Path
import csv
import numpy as np
from param import output

import functions as func


# A python script that takes in a 2D grayscale array and a parameter file and outputs the x, y, z positions for that frame.
def process_frame_to_csv(frame_path, params_path):
    """
    Process a single frame and save the results as a CSV file.

    Parameters:
    -----------
    frame_path : str
        Path to the frame file (numpy .npy format).
    params_path : str
        Path to the parameter file (JSON format).

    Output:
    -------
    CSV file with columns X, Y, Z, I_FS, I_GS.
    """
    # Load parameters
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Load frame
    frame = np.load(frame_path)

    # Process frame using positions_batch function
    results = func.positions_batch(
        (
            frame,
            None,  # Pass median frame if needed
            params["refractive_index"],
            params["wavelength"],
            params["magnification"],
            params["refocus_start"],
            params["step_size"],
            params["num_steps"],
            params["bp_large"],
            params["bp_small"],
            params["grad_threshold"],
            params["peak_min_dist"],
            func.bgPathToArray(params["bg_image_path"]),
            params["use_bg_image"],
        )
    )

    # Unpack results
    x_coords, y_coords, z_coords, intensity_fs, intensity_gs = results

    # Flatten arrays for CSV output
    rows = zip(
        np.ravel(x_coords), np.ravel(y_coords), np.ravel(z_coords),
        np.ravel(intensity_fs), np.ravel(intensity_gs)
    )

    # Output to CSV
    output_path = Path(str(frame_path).replace("inputs", "outputs")).with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["X", "Y", "Z", "I_FS", "I_GS"])  # Write header
        csv_writer.writerows(rows)  # Write data rows

    print(f"Processed frame saved to {output_path}")


if __name__ == '__main__':
    frame_path = sys.argv[1]
    params_path = sys.argv[2]
    process_frame_to_csv(frame_path, params_path)
