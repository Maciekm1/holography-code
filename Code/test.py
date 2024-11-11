import os
import matplotlib.pyplot as plt
import pandas as pd

# TODO - LabVIEW doesn't process/ save last frame??? frame 49 exists in python output but not in labVIEW
# -1 for all frames, a number for specific frame
FRAME = 0
PYTHON_3D_DATA_PATH = r'C:/Users/mz1794/Downloads/Python and Viking DHM port\40x_100Hz_1081_CHO_1_T5_detrend_frame0-50_-RS-.csv'

# 'frames' folder
LABVIEW_3D_DATA_FOLDER_PATH = r"C:\Users\mz1794\Downloads\Python and Viking DHM port\frames"

# which file to choose in 'frames' folder, without .txt or frame number (i.e. without 00000/00001...)
LABVIEW_FILE_NAME = "40x_100Hz_1081_CHO_1_T5_detrend_frame0-50_frame"


def main():
    # test bgPathToArray
    #  print(f.bgPathToArray(r'C:\Users\mz1794\Downloads\Python and Viking DHM port-20241010T094616Z-001\Python and Viking DHM port\1024bg.png'))

    # Figure to show both plots on
    fig = plt.figure(figsize=(12, 6))

    print('-' * 150)
    plot_3d_positions(fig, 121, FRAME)
    print('-' * 150)
    plot_expected_3d_positions_from_folder(fig ,LABVIEW_3D_DATA_FOLDER_PATH,122, FRAME)

    plt.show()


def plot_3d_positions(fig, subplot_pos, frame=-1):
    data = pd.read_csv(
        PYTHON_3D_DATA_PATH)
    #data_mod = pd.read_csv('C:/Users/mz1794/Downloads/Python and Viking DHM port-20241010T094616Z-001/Python and Viking DHM port/Python and Viking DHM _TH0.002_PMD20_SZ3.0_NUMSTEPS30_MOD.csv')
    #data = data_mod

    if frame != -1:
        data = data[data['FRAME'] == frame]

    print('python datapoints: ' + str(len(data)))

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


def plot_expected_3d_positions_from_folder(fig, folder_path, subplot_pos, frame=-1):
    all_data = []

    if frame == -1:
        # Loop through all files in the folder
        for filename in sorted(os.listdir(folder_path)):
            if filename.startswith(LABVIEW_FILE_NAME) and filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                # Assuming the text file contains X, Y, Z columns
                data = pd.read_csv(file_path, sep=r'\s+', header=None, names=['I' ,'X', 'Y', 'Z'])
                all_data.append(data)

        # Combine all the data into a single DataFrame
        combined_data = pd.concat(all_data, ignore_index=True)
        data = combined_data
    else:
        filename = f"{LABVIEW_FILE_NAME}{str(frame).zfill(5)}.txt"
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path, sep=r'\s+', header=None, names=['I', 'X', 'Y', 'Z'])

    print('labVIEW datapoints: ' + str(len(data)))

    # Extract x, y, z coordinates
    x = data['X']
    y = data['Y']
    z = data['Z']

    # Create a 3D plot
    #fig = plt.figure()
    ax = fig.add_subplot(subplot_pos, projection='3d')

    # Scatter plot of the 3D points
    ax.scatter(x, y, z, c='g', marker='o')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the viewing angle
    ax.view_init(elev=20., azim=120)
    #plt.show()


if __name__ == '__main__':
    main()
