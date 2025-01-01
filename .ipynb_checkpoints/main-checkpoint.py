import os
import numpy as np
from scipy.io import loadmat, savemat

# Define the base directory
BASE_DIR = "/home/jovyan/Projeto_PIV/data/1.1/CTownAirport_1.1"
RESULTS_DIR = os.path.join(BASE_DIR, "results_1.1")

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def estimate_homography(matches):
    num_points = matches.shape[0]
    A = []

    for i in range(num_points):
        x1, y1, x2, y2 = matches[i]
        A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))
    return H / H[2, 2]  # Normalize

def transform_coordinates(H, coords):
    transformed_coords = []
    for bbox in coords:
        x1, y1, x2, y2 = bbox
        blc = np.array([x1, y1, 1]).T
        trc = np.array([x2, y2, 1]).T

        blc_transformed = np.dot(H, blc)
        trc_transformed = np.dot(H, trc)

        blc_transformed /= blc_transformed[2]
        trc_transformed /= trc_transformed[2]

        transformed_coords.append([blc_transformed[0], blc_transformed[1], trc_transformed[0], trc_transformed[1]])

    return np.array(transformed_coords)

def process_video_frames():
    # Load keypoint matches
    kp_matches_path = os.path.join(BASE_DIR, 'kp_gmaps.mat')
    kp_matches = loadmat(kp_matches_path)['kp_gmaps']

    # Estimate the homography matrix
    H = estimate_homography(kp_matches)
    savemat(os.path.join(RESULTS_DIR, 'homography.mat'), {'H': H})

    frame_idx = 1
    yolo_dir = os.path.join(BASE_DIR, 'yolo')

    while True:
        try:
            # Load YOLO detections for the current frame
            yolo_file = os.path.join(yolo_dir, f'yolo_{frame_idx:04d}.mat')
            yolo_data = loadmat(yolo_file)

            # Transform the bounding box coordinates
            transformed_coords = transform_coordinates(H, yolo_data['xyxy'])

            # Save transformed YOLO data
            transformed_yolo_data = {
                'xyxy': transformed_coords,
                'id': yolo_data['id'],
                'class': yolo_data['class']
            }
            output_file = os.path.join(RESULTS_DIR, f'yolooutput_{frame_idx:04d}.mat')
            savemat(output_file, transformed_yolo_data)

            frame_idx += 1

        except FileNotFoundError:
            break  # No more frames to process

if __name__ == '__main__':
    process_video_frames()
