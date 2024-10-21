import os
import cv2
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplot

import functions as f
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    # test bgPathToArray
    #  print(f.bgPathToArray(r'C:\Users\mz1794\Downloads\Python and Viking DHM port-20241010T094616Z-001\Python and Viking DHM port\1024bg.png'))

    fig = plt.figure(figsize=(12, 6))
    plot_3d_positions(fig, 121)
    plot_expected_3d_positions_from_folder(fig ,r"C:\Users\mz1794\Downloads\Python and Viking DHM port-20241010T094616Z-001\Python and Viking DHM port\frames", 122)

    plt.show()

def plot_3d_positions(fig, subplot_pos):
    data = pd.read_csv(
        'C:/Users/mz1794/Downloads/Python and Viking DHM port-20241010T094616Z-001/Python and Viking DHM port/Python and Viking DHM _TH0.002_PMD40_SZ3.0_NUMSTEPS30_RS.csv')
    data_mod = pd.read_csv('C:/Users/mz1794/Downloads/Python and Viking DHM port-20241010T094616Z-001/Python and Viking DHM port/Python and Viking DHM _TH0.002_PMD20_SZ3.0_NUMSTEPS30_MOD.csv')
    #data = data_mod

    # Extract x, y, z coordinates
    x = data['X']
    y = data['Y']
    z = data['Z']

    # Create a 3D plot
    #fig = plt.figure()
    ax = fig.add_subplot(subplot_pos, projection='3d')

    # Scatter plot of the 3D points
    ax.scatter(x, y, z, c='r', marker='o')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    ax.view_init(elev=20.,azim=120)
    #plt.show()


def plot_expected_3d_positions_from_folder(fig, folder_path, subplot_pos):
    all_data = []

    # Loop through all files in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.startswith("40x_100Hz_1081_CHO_1_T5_detrend_frame0-50_frame") and filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            # Assuming the text file contains X, Y, Z columns
            data = pd.read_csv(file_path, sep=r'\s+', header=None, names=['I' ,'X', 'Y', 'Z'])
            all_data.append(data)

    # Combine all the data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    # Extract x, y, z coordinates
    x = combined_data['X']
    y = combined_data['Y']
    z = combined_data['Z']

    # Create a 3D plot
    #fig = plt.figure()
    ax = fig.add_subplot(subplot_pos, projection='3d')

    # Scatter plot of the 3D points
    ax.scatter(x, y, z, c='g', marker='o')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    ax.view_init(elev=20., azim=120)
    #plt.show()


if __name__ == '__main__':
    main()