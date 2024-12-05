#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Holographic Coordinate Detection Script with Progress Bar

Original Author: Erick
Modified by: Maciek (October 31, 2024)
"""

import os
from multiprocessing import Pool, cpu_count
from time import time

import functions as func
import numpy as np
import pandas as pd
import PySimpleGUI as sg
from tqdm import tqdm

# Flag to automatically open file after export
OPEN_CSV_AFTER_EXPORT = True


# GUI configuration and setup
def main():
    layout = [
        [
            sg.Text("Video Path (.avi)", size=(35, 1)),
            sg.FileBrowse(key="-VIDEO-"),
        ],
        [
            sg.Text("Background Image Path (.png)", size=(35, 1)),
            sg.FileBrowse(key="-BG_IMAGE-"),
        ],
        [
            sg.Text("Propagator Scheme", size=(35, 1)),
            sg.Radio(
                "Rayleigh-Sommerfeld", "SCHEME", key="-RS-", default=True
            ),
            sg.Radio("Modified", "SCHEME", key="-MOD-"),
        ],
        [
            sg.Text("Wavelength (μm)", size=(35, 1)),
            sg.InputText(default_text=0.642, key="-WAVELENGTH-"),
        ],
        [
            sg.Text("Refractive Index (Water = 1.3226)", size=(35, 1)),
            sg.InputText(default_text=1.3326, key="-REFRACTIVE_INDEX-"),
        ],
        [
            sg.Text("Magnification (10, 20, etc.)", size=(35, 1)),
            sg.InputText(default_text=40, key="-MAGNIFICATION-"),
        ],
        [
            sg.Text("Start Refocus Value (μm)", size=(35, 1)),
            sg.InputText(default_text=0, key="-REFOCUS_START-"),
        ],
        [
            sg.Text("Step Size (μm)", size=(35, 1)),
            sg.InputText(default_text=3, key="-STEP_SIZE-"),
        ],
        [
            sg.Text("Number of Steps", size=(35, 1)),
            sg.InputText(default_text=30, key="-NUM_STEPS-"),
        ],
        [
            sg.Text("Largest Bandpass Filter (px)", size=(35, 1)),
            sg.InputText(default_text=60, key="-BP_LARGE-"),
        ],
        [
            sg.Text("Smallest Bandpass Filter (px)", size=(35, 1)),
            sg.InputText(default_text=4, key="-BP_SMALL-"),
        ],
        [
            sg.Text("Gradient Stack Threshold (~0.1)", size=(35, 1)),
            sg.InputText(default_text=0.002, key="-GRAD_THRESHOLD-"),
        ],
        [
            sg.Text("Peak Minimum Distance (px)", size=(35, 1)),
            sg.InputText(default_text=2, key="-PEAK_MIN_DIST-"),
        ],
        [
            sg.Text("Frame Rate (fps)", size=(35, 1)),
            sg.InputText(default_text=50, key="-FRAME_RATE-"),
        ],
        [sg.Checkbox("Use Background Image", default=True, key="-USE_BG-")],
        [sg.Checkbox("Invert Video", default=False, key="-INVERT_VIDEO-")],
        [sg.Checkbox("Export as CSV", default=True, key="-EXPORT_CSV-")],
        [
            sg.Text("Frames for Calculation", size=(35, 1)),
            sg.InputText(default_text=50, key="-FRAME_COUNT-"),
        ],
        [sg.Button("Add Video"), sg.Button("Start"), sg.Cancel()],
    ]

    window = sg.Window("Holographic Video Input", layout)
    video_paths, bg_image_paths, processing_params = [], [], []

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Cancel"):
            break
        elif event == "Start":
            break
        elif event == "Add Video":
            print(f'Added {os.path.split(values["-VIDEO-"])[-1]} to queue.')

            # --- TEMP FOR DEBUGGING
            # Uncomment the two lines below to be able to select them with the UI
            # paths.append(values['-FILE-'])
            # bg_paths.append(values['-BGIMAGE-'])
            # If you uncommented the lines above, comment out the two lines below
            video_paths.append(
                "C:/Users/mz1794/Downloads/Python and Viking DHM port/40x_100Hz_1081_CHO_1_T5_detrend_frame0-50.avi"
            )
            bg_image_paths.append(
                r"C:\Users\mz1794\Downloads\Python and Viking DHM port\1024bg.png"
            )
            # ---

            scheme = "-RS-" if values["-RS-"] else "-MOD-"
            processing_params.append(
                {
                    "refractive_index": float(values["-REFRACTIVE_INDEX-"]),
                    "wavelength": float(values["-WAVELENGTH-"]),
                    "magnification": int(values["-MAGNIFICATION-"]),
                    "refocus_start": int(values["-REFOCUS_START-"]),
                    "step_size": float(values["-STEP_SIZE-"]),
                    "num_steps": int(values["-NUM_STEPS-"]),
                    "bp_large": int(values["-BP_LARGE-"]),
                    "bp_small": int(values["-BP_SMALL-"]),
                    "grad_threshold": float(values["-GRAD_THRESHOLD-"]),
                    "peak_min_dist": int(values["-PEAK_MIN_DIST-"]),
                    "frame_rate": int(values["-FRAME_RATE-"]),
                    "use_bg_image": values["-USE_BG-"],
                    "invert_video": values["-INVERT_VIDEO-"],
                    "export_csv": values["-EXPORT_CSV-"],
                    "frame_count": (
                        int(values["-FRAME_COUNT-"])
                        if values["-FRAME_COUNT-"]
                        else None
                    ),
                    "scheme": scheme,
                }
            )

    window.close()

    ### Processing Start

    # Process each video file that was added through GUI
    for idx, video_path in enumerate(video_paths):
        video_data = func.videoImport(video_path, 0)
        bg_image = func.bgPathToArray(bg_image_paths[idx])
        ni, nj, total_frames = video_data.shape

        if processing_params[idx]["invert_video"]:
            video_data = video_data.max() - video_data

        # Currently only using BG_Image normalization -> Median frame is not used
        # median_frame = func.medianImage(video_data, 20)

        frames_to_process = (
            processing_params[idx]["frame_count"] or total_frames
        )

        # Create a list of {framerate} items. Each item is a 2D array of x, y coordinates of that frame.
        frame_data = [video_data[i, :, :] for i in range(frames_to_process)]
        # median_data = [median_frame] * frames_to_process
        median_data = np.empty((1024, 1024, 50))  # Placeholder

        params = processing_params[idx]

        # Select function to run for frame processing, based on GUI Input.
        scheme_function = (
            func.positions_batch
            if params["scheme"] == "-RS-"
            else func.positions_batch_modified
        )

        ### --- Single core for debugging
        # Run in single-core mode if needed for debugging
        if False:  # Set to False to skip single-core execution
            results = []
            for i in range(frames_to_process):
                res = scheme_function(
                    tuple(
                        [
                            frame_data[i],
                            median_data,
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
                            bg_image,
                            params["use_bg_image"],
                        ]
                    )
                )
                results.append(res)
                print(
                    f"Processed frame {i + 1}/{frames_to_process} in single-core mode"
                )
            # Use results here if you want to inspect them after each frame is processed
        ### --- END

        ### Parallel Processing Start
        with Pool(cpu_count()) as pool:
            t_start = time()
            print("-" * 150)
            print(
                f"Processing file {idx + 1}/{len(video_paths)}: {os.path.split(video_path)[-1]}"
            )

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
            time_values = frame / float(values["-FRAME_COUNT-"])

            ### Organize Results

            data_row = np.column_stack(
                (x, y, z, i_fs, i_gs, frame, time_values)
            )
            positions = pd.concat(
                [
                    positions,
                    pd.DataFrame(
                        data_row,
                        columns=[
                            "X",
                            "Y",
                            "Z",
                            "I_FS",
                            "I_GS",
                            "FRAME",
                            "TIME",
                        ],
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
                f"{os.path.splitext(os.path.split(video_path)[-1])[0]}_{params['scheme']}.csv",
            )
            positions.to_csv(export_path, index=False)
            print(f"Exported to: {export_path}")

            if OPEN_CSV_AFTER_EXPORT:
                os.startfile(export_path)

    print("Done.")


if __name__ == "__main__":
    main()
