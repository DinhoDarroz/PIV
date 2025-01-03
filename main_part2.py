import os
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class HomographyEstimator(BaseEstimator, RegressorMixin):
    """Wrapper for homography estimation for use with RANSAC."""
    def fit(self, X, y):
        # Compute the homography using the provided matched points
        self.homography = compute_homography(np.hstack((X, y)))
        return self

    def predict(self, X):
        # Convert Cartesian points to homogeneous coordinates
        X_h = np.hstack((X, np.ones((X.shape[0], 1))))
        # Transform points using the homography matrix
        y_h = (self.homography @ X_h.T).T
        
        # Debug: Print intermediate values (uncomment to inspect during execution)
        # print("Homography matrix:\n", self.homography)
        # print("X_h (homogeneous coordinates of X):\n", X_h)
        # print("y_h before normalization (projective coordinates):\n", y_h)
        
        # Handle potential division by zero during normalization
        epsilon = 1e-10  # A small value to replace zero to avoid division issues
        y_h[:, 2] = np.where(y_h[:, 2] == 0, epsilon, y_h[:, 2])
        
        # Normalize back to Cartesian coordinates
        y_h /= y_h[:, 2][:, np.newaxis]
        
        # Debug: Print normalized points
        # print("y_h after normalization (Cartesian coordinates):\n", y_h)
        
        return y_h[:, :2]  # Return only the Cartesian components (x', y')



def normalize_keypoints(points):
    """Normalize keypoints for numerical stability."""
    mean = np.mean(points, axis=0)
    std = np.std(points)
    scale = np.sqrt(2) / std

    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]
    ])

    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    normalized_points = (T @ points_h.T).T
    return normalized_points[:, :2], T

def compute_homography(matches):
    """
    Compute the homography matrix using the Direct Linear Transform (DLT) algorithm with normalization.
    matches: Nx4 array of matched points [x1, y1, x2, y2].
    """
    points1, T1 = normalize_keypoints(matches[:, :2])
    points2, T2 = normalize_keypoints(matches[:, 2:])

    A = []
    for (x1, y1), (x2, y2) in zip(points1, points2):
        A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
    A = np.array(A)

    _, _, V = np.linalg.svd(A)
    H_normalized = V[-1].reshape(3, 3)

    H = np.linalg.inv(T2) @ H_normalized @ T1
    return H / H[2, 2]

def compute_homography_with_ransac(matches, threshold=1):
    """
    Compute the homography matrix using RANSAC to reject outliers.
    matches: Nx4 array of matched points [x1, y1, x2, y2].
    threshold: RANSAC inlier threshold.
    """
    X = matches[:, :2]
    y = matches[:, 2:]

    ransac = RANSACRegressor(HomographyEstimator(), residual_threshold=threshold, max_trials=1000, 
                             min_samples=4)
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    H = ransac.estimator_.homography

    return H, inlier_mask

def match_descriptors(desc1, desc2):
    """
    Match descriptors between two sets of keypoints using nearest neighbor matching.
    desc1: MxD array of descriptors from the first image.
    desc2: NxD array of descriptors from the second image.
    Returns:
        matches: List of index pairs [(i, j)] where i is the index in desc1 and j is the index in desc2.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(desc2)
    distances, indices = nn.kneighbors(desc1)
    matches = []
    for i, (d1, d2) in enumerate(zip(distances, indices)):
        if d1[0] < 0.6 * d1[1]:
            matches.append((i, d2[0]))
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

def compute_homographies(reference_kp, reference_desc, input_dir):
    """
    Compute homographies from all frames to the reference image using consecutive frame homographies.
    reference_kp: Keypoints of the reference image.
    reference_desc: Descriptors of the reference image.
    input_dir: Directory containing keypoints and descriptors for all frames.
    Returns:
        homographies: List of 3x3 homography matrices for all frames.
        consecutive_homographies: List of 3x3 homography matrices between consecutive frames.
    """
    frames = sorted([f for f in os.listdir(input_dir) if f.startswith('kp_') and f.endswith('.mat')])
    homographies = [np.eye(3)] * len(frames)
    consecutive_homographies = [None] * (len(frames) - 1)

    # Find the best frame (kbest) by matching with the reference keypoints
    max_inliers = 0
    kbest = 0
    for i, frame in enumerate(frames):
        kp_data = loadmat(os.path.join(input_dir, frame))
        kp, desc = kp_data['kp'], kp_data['desc']

        matches = match_descriptors(reference_desc, desc)
        matched_kp_ref, matched_kp = get_matched_keypoints(reference_kp, kp, matches)

        match_array = np.hstack((matched_kp_ref, matched_kp))
        _, inlier_mask = compute_homography_with_ransac(match_array)

        num_inliers = inlier_mask.sum()
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            kbest = i
    print(f"Best frame: {kbest + 1}")

    # Compute consecutive homographies
    for i in range(1, len(frames)):
        kp_data = loadmat(os.path.join(input_dir, frames[i]))
        kp, desc = kp_data['kp'], kp_data['desc']

        prev_kp_data = loadmat(os.path.join(input_dir, frames[i - 1]))
        prev_kp, prev_desc = prev_kp_data['kp'], prev_kp_data['desc']

        matches = match_descriptors(prev_desc, desc)
        matched_kp_prev, matched_kp = get_matched_keypoints(prev_kp, kp, matches)

        match_array = np.hstack((matched_kp_prev, matched_kp))
        H, inlier_mask = compute_homography_with_ransac(match_array)
        #H = compute_homography(match_array)

        consecutive_homographies[i - 1] = H

    # Compute homographies to the reference frame
    Hbest_to_R = homographies[kbest]

    for k in range(len(frames)):
        if k < kbest:
            Hk_to_R = np.eye(3)
            for i in range(k, kbest):
                Hk_to_R = consecutive_homographies[i] @ Hk_to_R
            homographies[k] = Hk_to_R @ Hbest_to_R
        elif k > kbest:
            Hk_to_R = np.eye(3)
            for i in range(kbest, k):
                Hk_to_R = np.linalg.inv(consecutive_homographies[i]) @ Hk_to_R
            homographies[k] = Hk_to_R @ Hbest_to_R

    return homographies, consecutive_homographies

def transform_bounding_boxes(H, bounding_boxes):
    """
    Transform YOLO bounding boxes using the homography matrix and ensure they remain within the image bounds.
    H: 3x3 homography matrix.
    bounding_boxes: Mx4 array of bounding boxes (x_blc, y_blc, x_trc, y_trc).
    image_width: Width of the reference image.
    image_height: Height of the reference image.
    """
    transformed_boxes = []
    for box in bounding_boxes:
        x_blc, y_blc, x_trc, y_trc = box

        # Convert corners to homogeneous coordinates
        blc = np.array([x_blc, y_blc, 1]).T
        trc = np.array([x_trc, y_trc, 1]).T

        # Transform the corners using the homography
        blc_transformed = np.dot(H, blc)
        trc_transformed = np.dot(H, trc)

        # Normalize back to Cartesian coordinates
        if blc_transformed[2] != 0:  # Avoid divide-by-zero
            blc_transformed /= blc_transformed[2]
        if trc_transformed[2] != 0:  # Avoid divide-by-zero
            trc_transformed /= trc_transformed[2]

        # Extract transformed (x, y) coordinates
        x_blc_new, y_blc_new = blc_transformed[:2]
        x_trc_new, y_trc_new = trc_transformed[:2]

        # Ensure bounding boxes stay within the reference image bounds
        #x_blc_new = max(0, min(image_width, x_blc_new))
        #y_blc_new = max(0, min(image_height, y_blc_new))
        #x_trc_new = max(0, min(image_width, x_trc_new))
        #y_trc_new = max(0, min(image_height, y_trc_new))

        # Skip invalid bounding boxes (e.g., if they collapse to a single point)
        #if x_blc_new >= x_trc_new or y_blc_new >= y_trc_new:
        #    continue

        # Append the valid transformed bounding box
        transformed_boxes.append([x_blc_new, y_blc_new, x_trc_new, y_trc_new])

    return np.array(transformed_boxes)


def process_video(reference_kp, reference_desc, input_dir, output_dir):
    """
    Process a single video directory, computing homographies and transforming YOLO detections.
    reference_kp: Keypoints of the reference image.
    reference_desc: Descriptors of the reference image.
    input_dir: Directory containing video input data.
    output_dir: Directory to save the output.
    """
    os.makedirs(output_dir, exist_ok=True)

    homographies, consecutive_homographies = compute_homographies(reference_kp, reference_desc, input_dir)

    yolo_frames = sorted([f for f in os.listdir(input_dir) if f.startswith('yolo_') and f.endswith('.mat')])
    for i, frame in enumerate(yolo_frames):
        yolo_path = os.path.join(input_dir, frame)
        yolo_data = loadmat(yolo_path)
        if 'xyxy' in yolo_data and i < len(homographies) and homographies[i] is not None:
            transformed_boxes = transform_bounding_boxes(homographies[i], yolo_data['xyxy'])
            transformed_yolo = {
                'xyxy': transformed_boxes,
                'id': yolo_data['id'],
                'class': yolo_data['class']
            }
            savemat(os.path.join(output_dir, f"yolooutput_{i + 1:04d}.mat"), transformed_yolo)

    homographies_stack = np.stack([H if H is not None else np.eye(3) for H in homographies], axis=2)
    savemat(os.path.join(output_dir, 'homographies.mat'), {'H': homographies_stack})

    consecutive_homographies_stack = np.stack([H if H is not None else np.eye(3) for H in consecutive_homographies], axis=2)
    savemat(os.path.join(output_dir, 'consecutive_homographies.mat'), {'H': consecutive_homographies_stack})

def main():
    import sys
    args = sys.argv[1:]

    ref_dir = args[0]
    input_dir = args[1]
    output_dir = args[2]

    ref_kp_data = loadmat(os.path.join(ref_dir, 'kp_ref.mat'))
    reference_kp = ref_kp_data['kp']
    reference_desc = ref_kp_data['desc']

    process_video(reference_kp, reference_desc, input_dir, output_dir)

if __name__ == '__main__':
    main()
