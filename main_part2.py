import os
import numpy as np
from scipy.io import loadmat, savemat

def compute_homography(src_points, dst_points):
    """
    Compute the homography matrix using the Direct Linear Transform (DLT) algorithm.
    src_points: Nx2 array of keypoints in the source image.
    dst_points: Nx2 array of keypoints in the destination image.
    """
    num_points = src_points.shape[0]
    A = []
    for i in range(num_points):
        x1, y1 = src_points[i]
        x2, y2 = dst_points[i]
        A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))
    return H / H[2, 2]

def match_descriptors(desc1, desc2):
    """
    Match descriptors between two sets of keypoints using nearest neighbor matching.
    desc1: MxD array of descriptors from the first image.
    desc2: NxD array of descriptors from the second image.
    Returns:
        matches: List of index pairs [(i, j)] where i is the index in desc1 and j is the index in desc2.
    """
    matches = []
    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        j = np.argmin(distances)
        matches.append((i, j))
    return matches

def get_matched_keypoints(kp1, kp2, matches):
    """
    Extract matched keypoints from two sets of keypoints based on descriptor matches.
    kp1: Keypoints from the first image (Mx2).
    kp2: Keypoints from the second image (Nx2).
    matches: List of index pairs [(i, j)] matching keypoints.
    Returns:
        matched_kp1: Matched keypoints in the first image.
        matched_kp2: Matched keypoints in the second image.
    """
    matched_kp1 = np.array([kp1[i] for i, _ in matches])
    matched_kp2 = np.array([kp2[j] for _, j in matches])
    return matched_kp1, matched_kp2

def transform_bounding_boxes(H, bounding_boxes):
    """
    Transform YOLO bounding boxes using the homography matrix.
    H: 3x3 homography matrix.
    bounding_boxes: Mx4 array of bounding boxes (x_blc, y_blc, x_trc, y_trc).
    """
    transformed_boxes = []
    for box in bounding_boxes:
        x_blc, y_blc, x_trc, y_trc = box
        blc = np.array([x_blc, y_blc, 1]).T
        trc = np.array([x_trc, y_trc, 1]).T

        blc_transformed = np.dot(H, blc)
        trc_transformed = np.dot(H, trc)

        blc_transformed /= blc_transformed[2]
        trc_transformed /= trc_transformed[2]

        transformed_boxes.append([blc_transformed[0], blc_transformed[1], trc_transformed[0], trc_transformed[1]])

    return np.array(transformed_boxes)

def find_best_matching_frame(reference_kp, reference_desc, input_dir):
    """
    Find the frame in the video that has the most matches with the reference image.
    reference_kp: Keypoints of the reference image.
    reference_desc: Descriptors of the reference image.
    input_dir: Directory containing keypoints and descriptors for all frames.
    Returns:
        best_frame_idx: Index of the best matching frame.
        best_matches: List of matches between the reference and the best frame.
    """
    best_frame_idx = -1
    best_matches = []
    max_matches = 0

    frames = sorted([f for f in os.listdir(input_dir) if f.startswith('kp_') and f.endswith('.mat')])

    for i, frame in enumerate(frames):
        frame_path = os.path.join(input_dir, frame)
        kp_data = loadmat(frame_path)
        frame_kp = kp_data['kp']
        frame_desc = kp_data['desc']

        matches = match_descriptors(reference_desc, frame_desc)

        if len(matches) > max_matches:
            max_matches = len(matches)
            best_frame_idx = i
            best_matches = matches

    return best_frame_idx, best_matches

def process_video(reference_kp, reference_desc, input_dir, output_dir):
    """
    Process a single video directory, computing homographies and transforming YOLO detections.
    reference_kp: Keypoints of the reference image.
    reference_desc: Descriptors of the reference image.
    input_dir: Directory containing video input data.
    output_dir: Directory to save the output.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find the best matching frame to the reference
    best_frame_idx, best_matches = find_best_matching_frame(reference_kp, reference_desc, input_dir)
    frames = sorted([f for f in os.listdir(input_dir) if f.startswith('kp_') and f.endswith('.mat')])

    if best_frame_idx == -1:
        raise ValueError("No frame matches the reference image.")

    # Compute homography for the best matching frame
    #best_frame_path = os.path.join(input_dir, frames[best_frame_idx])
    #best_frame_data = loadmat(best_frame_path)
    #best_frame_kp = best_frame_data['kp']
    #best_frame_desc = best_frame_data['desc']

    #matched_best_kp, matched_ref_kp = get_matched_keypoints(best_frame_kp, reference_kp, best_matches)
    #H_best_to_ref = compute_homography(matched_best_kp, matched_ref_kp)
    H_best_to_ref = [np.eye(3)]

    # Propagate homographies to other frames
    homographies = [None] * len(frames)
    homographies[best_frame_idx] = H_best_to_ref

    for i in range(best_frame_idx - 1, -1, -1):
        # Backward propagation
        frame_data = loadmat(os.path.join(input_dir, frames[i]))
        frame_kp = frame_data['kp']
        frame_desc = frame_data['desc']

        next_frame_data = loadmat(os.path.join(input_dir, frames[i + 1]))
        next_frame_kp = next_frame_data['kp']
        next_frame_desc = next_frame_data['desc']

        matches = match_descriptors(frame_desc, next_frame_desc)
        matched_kp, matched_next_kp = get_matched_keypoints(frame_kp, next_frame_kp, matches)

        H_frame_to_next = compute_homography(matched_kp, matched_next_kp)
        homographies[i] = np.dot(homographies[i + 1], np.linalg.inv(H_frame_to_next))

    for i in range(best_frame_idx + 1, len(frames)):
        # Forward propagation
        frame_data = loadmat(os.path.join(input_dir, frames[i]))
        frame_kp = frame_data['kp']
        frame_desc = frame_data['desc']

        prev_frame_data = loadmat(os.path.join(input_dir, frames[i - 1]))
        prev_frame_kp = prev_frame_data['kp']
        prev_frame_desc = prev_frame_data['desc']

        matches = match_descriptors(prev_frame_desc, frame_desc)
        matched_prev_kp, matched_kp = get_matched_keypoints(prev_frame_kp, frame_kp, matches)

        H_prev_to_frame = compute_homography(matched_prev_kp, matched_kp)
        homographies[i] = np.dot(homographies[i - 1], H_prev_to_frame)

    # Transform YOLO detections
    for i, frame in enumerate(frames):
        yolo_path = os.path.join(input_dir, f"yolo_{i + 1:04d}.mat")
        if os.path.exists(yolo_path):
            yolo_data = loadmat(yolo_path)
            if 'xyxy' in yolo_data and homographies[i] is not None:
                transformed_boxes = transform_bounding_boxes(homographies[i], yolo_data['xyxy'])
                transformed_yolo = {
                    'xyxy': transformed_boxes,
                    'id': yolo_data['id'],
                    'class': yolo_data['class']
                }
                savemat(os.path.join(output_dir, f"yolooutput_{i + 1:04d}.mat"), transformed_yolo)

    # Save all homographies
    homographies_stack = np.stack([H if H is not None else np.eye(3) for H in homographies], axis=2)
    savemat(os.path.join(output_dir, 'homographies.mat'), {'H': homographies_stack})

def main():
    import sys
    args = sys.argv[1:]

    # Directories
    ref_dir = args[0]
    input_dir = args[1]
    output_dir = args[2]

    # Load reference image keypoints and descriptors
    ref_kp_data = loadmat(os.path.join(ref_dir, 'kp_ref.mat'))
    reference_kp = ref_kp_data['kp']
    reference_desc = ref_kp_data['desc']

    # Process the input directory
    process_video(reference_kp, reference_desc, input_dir, output_dir)

if __name__ == '__main__':
    main()
