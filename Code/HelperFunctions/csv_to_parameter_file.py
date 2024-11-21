import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import functions as func

SPREADSHEET_PATH = r'C:\Users\mz1794\Downloads\Python and Viking DHM port\CSV\2024-01-24 - part 1 - videos to frames_subset_Maciek.xlsx'

def extract_params_and_frames(spreadsheet_path, row_index):
    # Load spreadsheet
    data = pd.read_excel(spreadsheet_path, header=1)

    # Select row
    if row_index >= len(data):
        raise ValueError("Row index out of range.")

    row = data.iloc[row_index]

    # Extract parameters
    parameters = {
        "bg_image_path": row['BG image path (if none, just copy video path)'],
        "refractive_index": float(row['refractive index']),
        "wavelength": float(row['Wavelength (um)']),
        "magnification": (float(row['pixels/micron']) / 0.711) * 10,
        "refocus_start": int(row['start refocus value (um)']),
        "step_size": float(row['step size (um)']),
        "num_steps": int(row['Number of steps']),
        "bp_large": int(row['bp largest (px)']),
        "bp_small": int(row['bp smallest (px)']),
        "grad_threshold": float(row['threshold for gradient stack (raw, not 0-255)']),
        "peak_min_dist": 2,
        "frame_rate": 50,
        "use_bg_image": bool(row['use bg image (1/0)']),
        "invert_video": False,
        "export_csv": True,
        "frame_count": 50,
        "scheme": "-RS-",
    }

    video_path = row['Video Path']

    # Create output folder
    video_name = Path(video_path).stem
    output_folder = Path(f"./{video_name}_frames")
    output_folder.mkdir(exist_ok=True)
    output_folder_frames = Path(f"./{video_name}_frames/inputs")
    output_folder_frames.mkdir(exist_ok=True)

    # Save parameters as JSON
    json_path = output_folder / "parameters.json"
    with open(json_path, 'w') as f:
        json.dump(parameters, f, indent=4)
    #print(f"Parameters saved to {json_path}")

    # Extract frames
    video_data = func.videoImport(video_path, 0)

    for i in range(video_data.shape[0]):
        frame_path = output_folder_frames / f"frame_{i:05d}.npy"
        np.save(frame_path, video_data[i])
    #print(f"Frames extracted to {output_folder_frames}")
    print(output_folder_frames)

if __name__ == '__main__':
    spreadsheet_path = sys.argv[1] if len(sys.argv) > 1 else SPREADSHEET_PATH
    row_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # Default to the first row
    extract_params_and_frames(spreadsheet_path, row_index)
