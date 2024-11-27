import os
import sys
from multiprocessing import Pool, cpu_count
from time import time

import functions as func
import numpy as np
import pandas as pd
from tqdm import tqdm

PEAK_MIN_DISTANCE = 2
RAYLEIGH_SOMMERFELD_PROPAGATOR = True
FRAME_COUNT = 50
EXPORT_CSV = True
INVERT_VIDEO = False

# Default path - used if not provided as an argument
SPREADSHEET_PATH = r"C:\Users\mz1794\Downloads\Python and Viking DHM port\CSV\2024-01-24 - part 1 - videos to frames_subset_Maciek.xlsx"


def main():
    """Interface for combining parameters from spreadsheet and sending them to processing function."""
    # Use the provided spreadsheet path or the default path above
    spreadsheet_path = sys.argv[1] if len(sys.argv) > 1 else SPREADSHEET_PATH

    processing_params = []
    video_paths, bg_image_paths, params = load_params_from_spreadsheet(
        spreadsheet_path
    )

    # TODO - SOME PARAMS ARE NOT USED (area lower + upper, flip_z_gradient, median_frames and num_loops) even though they are loaded in from spreadsheet.
    for param in params:
        scheme = "-RS-" if RAYLEIGH_SOMMERFELD_PROPAGATOR else "-MOD-"
        processing_params.append(
            {
                "refractive_index": float(param["refractive_index"]),
                "wavelength": float(param["wavelength"]),
                "magnification": (float(param["pixels_per_micron"]) / 0.711)
                * 10,
                "refocus_start": int(param["refocus_start"]),
                "step_size": float(param["step_size"]),
                "num_steps": int(param["num_steps"]),
                "bp_large": int(param["bp_large"]),
                "bp_small": int(param["bp_small"]),
                "grad_threshold": float(param["grad_threshold"]),
                "peak_min_dist": PEAK_MIN_DISTANCE,
                "frame_rate": FRAME_COUNT,
                "use_bg_image": param["use_bg_image"],
                "invert_video": INVERT_VIDEO,
                "export_csv": EXPORT_CSV,
                "frame_count": FRAME_COUNT,
                "scheme": scheme,
            }
        )

    main_processing(video_paths, bg_image_paths, processing_params)


def main_processing(video_path, bg_image_path, processing_params):
    """Process video."""
    # Process each video file that was added
    video_data = func.videoImport(video_path, 0)
    bg_image = func.bgPathToArray(bg_image_path)
    total_frames = video_data.shape

    if processing_params["invert_video"]:
        video_data = video_data.max() - video_data

    # Currently only using BG_Image normalization -> Median frame is not used
    median_frame = func.medianImage(video_data, 20)

    frames_to_process = processing_params["frame_count"] or total_frames

    # Create a list of {framerate} items. Each item is a 2D array of x, y coordinates of that frame.
    frame_data = [video_data[i, :, :] for i in range(frames_to_process)]
    median_data = [median_frame] * frames_to_process

    params = processing_params

    # Select function to run for frame processing, based on GUI Input.
    scheme_function = (
        func.positions_batch
        if params["scheme"] == "-RS-"
        else func.positions_batch_modified
    )

    ### Parallel Processing Start
    with Pool(cpu_count()) as pool:
        t_start = time()

        # Execute batch processing in parallel with progress bar
        # This runs scheme_function on each frame with corresponding parameters.
        results = []
        with tqdm(
            total=frames_to_process, desc="Processing Frames", unit="frame"
        ) as progress_bar:
            for res in pool.imap_unordered(
                scheme_function,
                zip(
                    frame_data,
                    median_data,
                    [params["refractive_index"]] * frames_to_process,
                    [params["wavelength"]] * frames_to_process,
                    [params["magnification"]] * frames_to_process,
                    [params["refocus_start"]] * frames_to_process,
                    [params["step_size"]] * frames_to_process,
                    [params["num_steps"]] * frames_to_process,
                    [params["bp_large"]] * frames_to_process,
                    [params["bp_small"]] * frames_to_process,
                    [params["grad_threshold"]] * frames_to_process,
                    [params["peak_min_dist"]] * frames_to_process,
                    [bg_image] * frames_to_process,
                    [params["use_bg_image"]] * frames_to_process,
                ),
            ):
                results.append(res)
                progress_bar.update(1)

        print("Processing completed in", time() - t_start, "seconds.")

    positions = pd.DataFrame(
        columns=["X", "Y", "Z", "I_FS", "I_GS", "FRAME", "TIME"]
    )

    ### Collect Results

    for i in range(len(results)):
        x = results[i][0][0]
        y = results[i][1][0]
        z = results[i][2][0]
        i_fs = results[i][3][0]
        i_gs = results[i][4][0]

        frame = np.full_like(x, i)
        # TODO - FIX THIS - READ IN FRAME COUNT FROM PARAMS
        time_values = frame / FRAME_COUNT

        ### Organize Results

        data_row = np.column_stack((x, y, z, i_fs, i_gs, frame, time_values))
        positions = pd.concat(
            [
                positions,
                pd.DataFrame(
                    data_row,
                    columns=["X", "Y", "Z", "I_FS", "I_GS", "FRAME", "TIME"],
                ),
            ],
            ignore_index=True,
        )

    positions = positions.astype("float")
    positions["TIME"] = positions["TIME"].round(3)

    ### Output results

    # Export CSV if required
    if params["export_csv"]:
        export_path = os.path.join(
            os.path.split(video_path)[0],
            f"{os.path.splitext(os.path.split(video_path)[-1])[0]}_{params['scheme']}_TEST.csv",
        )
        positions.to_csv(export_path, index=False)
        print(f"Exported to: {export_path}")

    print("Done.")


def load_params_from_spreadsheet(spreadsheet_path):
    """Load parameters from spreadsheet."""
    # Load spreadsheet
    data = pd.read_excel(spreadsheet_path, header=1)

    video_paths, bg_image_paths, processing_params = [], [], []

    # Iterate over each row in the spreadsheet and gather parameters
    for _, row in data.iterrows():
        video_paths.append(row.iloc[0])
        bg_image_paths.append(row.iloc[1])

        processing_params.append(
            {
                "wavelength": float(row.iloc[2]),
                "refractive_index": float(row.iloc[3]),
                "pixels_per_micron": float(row.iloc[4]),
                "refocus_start": float(row.iloc[5]),
                "step_size": float(row.iloc[6]),
                "flip_z_gradient": bool(row.iloc[7]),
                "num_steps": int(row.iloc[8]),
                "bp_large": int(row.iloc[9]),
                "bp_small": int(row.iloc[10]),
                "use_bg_image": bool(row.iloc[11]),
                "median_frames": int(row.iloc[12]),
                "num_loops": int(row.iloc[13]),
                "grad_threshold": float(row.iloc[14]),
                "area_upper_limit": int(row.iloc[15]),
                "area_lower_limit": int(row.iloc[16]),
            }
        )

    ### Print out Params for debugging ---
    # for i, params in enumerate(processing_params):
    #     print(f"Video Path: {video_paths[i]}")
    #     print(f"Background Image Path: {bg_image_paths[i]}")
    #     print("Parameters:", params)
    # ---

    return video_paths, bg_image_paths, processing_params


if __name__ == "__main__":
    main()
