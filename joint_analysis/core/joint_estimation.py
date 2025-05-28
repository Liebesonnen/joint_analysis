"""
Joint type estimation algorithms and parameter calculation with PyTorch acceleration.
"""
import math
import numpy as np
import torch
from scipy.signal import savgol_filter
from .scoring import (
    normalize_vector_torch, find_neighbors_batch, compute_basic_scores,
    compute_joint_probability_new, compute_motion_salience_batch, se3_log_map_batch,
    estimate_rotation_matrix_batch
)

# Global reference variables to maintain consistency during estimation
planar_normal_reference = None
planar_axis1_reference = None
planar_axis2_reference = None
plane_is_fixed = False
screw_axis_reference = None
prismatic_axis_reference = None
revolute_axis_reference = None


def calculate_velocity_and_angular_velocity_for_all_frames(
        all_points_history,
        dt=0.1,
        num_neighbors=10,
        use_savgol=True,
        savgol_window=5,
        savgol_poly=2,
        use_multi_frame=False,
        multi_frame_window_radius=2,
        v_max_threshold=10.0,
        w_max_threshold=30.0,
        use_percentile=False,
        outlier_percentile=95
):
    """
    Compute linear and angular velocity for (T,N,3) using improved filtering methods.
    Returns v_arr, w_arr => (T-1, N, 3).
    """
    from scipy.signal import savgol_filter

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not isinstance(all_points_history, torch.Tensor):
        all_points_history = torch.tensor(all_points_history, dtype=torch.float32, device=device)

    T, N, _ = all_points_history.shape
    if T < 2:
        return None, None

    # Step 1: Apply Savitzky-Golay filter to smooth point cloud data
    filtered_points = all_points_history.clone().cpu().numpy()

    if use_savgol and (T >= savgol_window):
        # Filter each point's trajectory individually
        for n in range(N):
            for dim in range(3):
                filtered_points[:, n, dim] = savgol_filter(
                    filtered_points[:, n, dim],
                    window_length=savgol_window,
                    polyorder=savgol_poly,
                    mode='mirror'
                )

    # Step 2: Calculate linear velocities using SG derivatives directly
    v_arr = np.zeros((T - 1, N, 3))

    if use_savgol and (T >= savgol_window):
        # Calculate derivatives directly using SG filter
        point_velocities = np.zeros((T, N, 3))
        for n in range(N):
            for dim in range(3):
                # Use deriv=1 to get first derivative
                point_velocities[:, n, dim] = savgol_filter(
                    all_points_history[:, n, dim].cpu().numpy(),
                    window_length=savgol_window,
                    polyorder=savgol_poly,
                    deriv=1,
                    delta=dt,
                    mode='mirror'
                )

        # Use the velocities for frames 0 to T-2
        v_arr = point_velocities[:-1]
    else:
        # Fall back to finite differences if not using SG
        for t in range(T - 1):
            v_arr[t] = (filtered_points[t + 1] - filtered_points[t]) / dt

    # Convert back to tensor for further processing
    filtered_points_tensor = torch.tensor(filtered_points, dtype=torch.float32, device=device)

    # Step 3: Calculate angular velocity using improved method
    w_arr = np.zeros((T - 1, N, 3))

    # Computing rotation matrices and extracting angular velocities
    for t in range(T - 1):
        # Current and next frame points
        current_points = filtered_points_tensor[t]
        next_points = filtered_points_tensor[t + 1]

        # Find neighbors for each point
        neighbor_idx = find_neighbors_batch(current_points.unsqueeze(0), num_neighbors)[0]

        # Compute angular velocity for each point
        for i in range(N):
            # Get neighborhoods
            src_neighborhood = current_points[neighbor_idx[i]]
            dst_neighborhood = next_points[neighbor_idx[i]]

            # Skip if not enough valid neighbors
            if src_neighborhood.shape[0] < 3 or dst_neighborhood.shape[0] < 3:
                continue

            # Center points
            src_center = torch.mean(src_neighborhood, dim=0)
            dst_center = torch.mean(dst_neighborhood, dim=0)
            src_centered = src_neighborhood - src_center
            dst_centered = dst_neighborhood - dst_center

            # Compute covariance matrix
            cov_matrix = torch.matmul(src_centered.transpose(0, 1), dst_centered)

            # SVD decomposition
            try:
                U, S, Vt = torch.linalg.svd(cov_matrix)

                # Construct rotation matrix
                R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

                # Handle reflection case
                if torch.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

                # Ensure R is a valid rotation matrix
                U, S, Vt = torch.linalg.svd(R)
                R = torch.matmul(U, Vt)

                # Compute rotation angle
                cos_theta = (torch.trace(R) - 1) / 2
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                theta = torch.acos(cos_theta)

                # Extract angular velocity vector
                if theta > 1e-6:
                    sin_theta = torch.sin(theta)
                    if abs(sin_theta) > 1e-6:
                        # Extract from skew-symmetric part
                        W = (R - R.transpose(0, 1)) / (2 * sin_theta)
                        omega = torch.zeros(3, device=device)
                        omega[0] = W[2, 1]
                        omega[1] = W[0, 2]
                        omega[2] = W[1, 0]

                        # Angular velocity = axis * angle / time
                        omega = omega * theta / dt
                        w_arr[t, i] = omega.cpu().numpy()
            except:
                # Handle numerical issues
                pass

    # Apply Savitzky-Golay filter to smooth angular velocity results
    if use_savgol and (T - 1 >= savgol_window):
        w_arr_filtered = np.zeros_like(w_arr)
        for i in range(N):
            for dim in range(3):
                w_arr_filtered[:, i, dim] = savgol_filter(
                    w_arr[:, i, dim],
                    window_length=savgol_window,
                    polyorder=savgol_poly,
                    mode='mirror'
                )
        w_arr = w_arr_filtered

    # Apply outlier handling
    v_magnitudes = np.linalg.norm(v_arr, axis=2)
    w_magnitudes = np.linalg.norm(w_arr, axis=2)
    B = T - 1

    # Apply thresholds
    if use_percentile and B * N > 10:
        v_threshold = np.percentile(v_magnitudes, outlier_percentile)
        w_threshold = np.percentile(w_magnitudes, outlier_percentile)
        v_threshold = min(v_threshold, v_max_threshold)
        w_threshold = min(w_threshold, w_max_threshold)
    else:
        v_threshold = v_max_threshold
        w_threshold = w_max_threshold

    # Handle outliers
    v_outlier_mask = v_magnitudes > v_threshold
    w_outlier_mask = w_magnitudes > w_threshold

    for b in range(B):
        for n in range(N):
            if v_outlier_mask[b, n]:
                v_arr[b, n, :] = 0.0
            if w_outlier_mask[b, n]:
                w_arr[b, n, :] = 0.0

    return v_arr, w_arr

# def calculate_velocity_and_angular_velocity_for_all_frames(
#         all_points_history,
#         dt=0.1,
#         num_neighbors=10,
#         use_savgol=True,
#         savgol_window=5,
#         savgol_poly=2,
#         use_multi_frame=False,
#         window_radius=2,
#         v_max_threshold=10.0,
#         w_max_threshold=30.0,
#         use_percentile=False,
#         outlier_percentile=95,
#         use_so3_filter=False  # New parameter to control SO(3) filtering
# ):
#     """
#     Compute linear and angular velocity for (T,N,3) using improved filtering methods.
#     Returns v_arr, w_arr => (T-1, N, 3).
#     """
#     from scipy.signal import savgol_filter
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     if not isinstance(all_points_history, torch.Tensor):
#         all_points_history = torch.tensor(all_points_history, dtype=torch.float32, device=device)
#
#     T, N, _ = all_points_history.shape
#     if T < 2:
#         return None, None
#
#     # Step 1: Apply SG filtering to position data first
#     filtered_points = all_points_history.clone().cpu().numpy()
#
#     if use_savgol and (T >= savgol_window):
#         # Filter each point's trajectory individually
#         for n in range(N):
#             for dim in range(3):
#                 filtered_points[:, n, dim] = savgol_filter(
#                     filtered_points[:, n, dim],
#                     window_length=savgol_window,
#                     polyorder=savgol_poly,
#                     mode='mirror'
#                 )
#
#     # Convert back to tensor for further processing
#     filtered_points_tensor = torch.tensor(filtered_points, dtype=torch.float32, device=device)
#
#     # Step 2: Calculate linear velocities by differentiating filtered positions
#     v_arr = np.zeros((T - 1, N, 3))
#     for t in range(T - 1):
#         v_arr[t] = (filtered_points[t + 1] - filtered_points[t]) / dt
#
#     # Step 3-6: Angular velocity calculation using SO(3) filtering
#     if use_so3_filter:
#         # Compute rotation matrices relative to first frame
#         rotation_matrices = np.zeros((3, 3, T))
#         rotation_matrices[:, :, 0] = np.eye(3)  # First frame is identity
#
#         # Compute rotation from first frame to each subsequent frame
#         for t in range(1, T):
#             # Get reference (first) frame and current frame points
#             ref_points = filtered_points_tensor[0]  # First frame as reference
#             cur_points = filtered_points_tensor[t]
#
#             # For each point, find its neighborhood in both frames
#             point_rotations = []
#             for i in range(N):
#                 # Find neighbors for this point
#                 neighbor_idx = find_neighbors_batch(ref_points[i:i + 1].unsqueeze(0),
#                                                     num_neighbors)[0][0]
#
#                 # Get same neighborhoods in both frames
#                 ref_neighborhood = ref_points[neighbor_idx]
#                 cur_neighborhood = cur_points[neighbor_idx]
#
#                 # Center the neighborhoods
#                 ref_center = torch.mean(ref_neighborhood, dim=0, keepdim=True)
#                 cur_center = torch.mean(cur_neighborhood, dim=0, keepdim=True)
#                 ref_centered = ref_neighborhood - ref_center
#                 cur_centered = cur_neighborhood - cur_center
#
#                 # Estimate rotation matrix using batch SVD
#                 R = estimate_rotation_matrix_batch(ref_centered.unsqueeze(0),
#                                                    cur_centered.unsqueeze(0))[0]
#                 point_rotations.append(R.cpu().numpy())
#
#             # Combine rotations from all points (with weighting based on motion salience)
#             avg_rotation = np.mean(np.array(point_rotations), axis=0)
#
#             # Ensure it's a valid rotation matrix
#             U, _, Vh = np.linalg.svd(avg_rotation)
#             rotation_matrices[:, :, t] = U @ Vh
#
#         # Apply SO(3) Savitzky-Golay filter
#         n = min(10, (T - 1) // 4)  # Half window size - adjust as needed
#         p = min(3, 2 * n - 1)  # Polynomial order - should be less than window size
#
#         # Apply SO(3) filter from the sg_filter package
#         R_filtered, w_filtered, _, tf = sgolayfiltSO3(
#             rotation_matrices,
#             p=p,
#             n=n,
#             freq=int(1 / dt)
#         )
#
#         # Create output angular velocity array
#         w_arr = np.zeros((T - 1, N, 3))
#
#         # The filtered data will be shorter - handle edge cases
#         filtered_length = w_filtered.shape[1]
#         start_idx = n
#         end_idx = start_idx + filtered_length
#
#         # Copy filtered angular velocities to all points
#         for t_idx in range(filtered_length):
#             t = t_idx + start_idx
#             if t < T - 1:
#                 w_arr[t] = np.tile(w_filtered[:, t_idx], (N, 1))
#
#         # Fill start and end regions that couldn't be filtered
#         for t in range(min(start_idx, T - 1)):
#             if start_idx < T - 1:
#                 w_arr[t] = w_arr[start_idx]
#
#         for t in range(min(end_idx, T - 1), T - 1):
#             if end_idx > 0 and end_idx - 1 < T - 1:
#                 w_arr[t] = w_arr[end_idx - 1]
#     else:
#         # Fallback to original neighborhood-based method for angular velocity
#         pts_prev = filtered_points_tensor[:-1]
#         pts_curr = filtered_points_tensor[1:]
#         B = T - 1
#         neighbor_idx_prev = find_neighbors_batch(pts_prev, num_neighbors)
#         neighbor_idx_curr = find_neighbors_batch(pts_curr, num_neighbors)
#         K = num_neighbors
#
#         src_batch = pts_prev[torch.arange(B, device=device).view(B, 1, 1), neighbor_idx_prev, :]
#         tar_batch = pts_curr[torch.arange(B, device=device).view(B, 1, 1), neighbor_idx_curr, :]
#         src_2d = src_batch.reshape(B * N, K, 3)
#         tar_2d = tar_batch.reshape(B * N, K, 3)
#
#         R_2d = estimate_rotation_matrix_batch(src_2d, tar_2d)
#         c1_2d = src_2d.mean(dim=1)
#         c2_2d = tar_2d.mean(dim=1)
#         delta_p_2d = c2_2d - c1_2d
#
#         eye_4 = torch.eye(4, device=device).unsqueeze(0).expand(B * N, -1, -1).clone()
#         eye_4[:, :3, :3] = R_2d
#         eye_4[:, :3, 3] = delta_p_2d
#         transform_matrices_2d = eye_4
#         se3_logs_2d = se3_log_map_batch(transform_matrices_2d)
#         w_2d = se3_logs_2d[:, 3:] / dt
#         w_arr = w_2d.reshape(B, N, 3).cpu().numpy()
#
#     # Apply outlier handling as in original code
#     v_magnitudes = np.linalg.norm(v_arr, axis=2)
#     w_magnitudes = np.linalg.norm(w_arr, axis=2)
#     B = T - 1
#
#     # Apply thresholds
#     if use_percentile and B * N > 10:
#         v_threshold = np.percentile(v_magnitudes, outlier_percentile)
#         w_threshold = np.percentile(w_magnitudes, outlier_percentile)
#         v_threshold = min(v_threshold, v_max_threshold)
#         w_threshold = min(w_threshold, w_max_threshold)
#     else:
#         v_threshold = v_max_threshold
#         w_threshold = w_max_threshold
#
#     # Handle outliers
#     v_outlier_mask = v_magnitudes > v_threshold
#     w_outlier_mask = w_magnitudes > w_threshold
#
#     for b in range(B):
#         for n in range(N):
#             if v_outlier_mask[b, n]:
#                 v_arr[b, n, :] = 0.0
#             if w_outlier_mask[b, n]:
#                 w_arr[b, n, :] = 0.0
#
#     return v_arr, w_arr

# def calculate_velocity_and_angular_velocity_for_all_frames(
#         all_points_history,
#         dt=0.1,
#         num_neighbors=400,
#         use_savgol=True,
#         savgol_window=5,
#         savgol_poly=2,
#         use_multi_frame=False,
#         window_radius=2,
#         # Added outlier handling parameters
#         v_max_threshold=10.0,  # Maximum allowed linear velocity (m/s)
#         w_max_threshold=30.0,  # Maximum allowed angular velocity (rad/s)
#         use_percentile=False,  # Use percentile-based outlier detection
#         outlier_percentile=95  # Percentile threshold for outliers
# ):
#     """
#     Compute linear and angular velocity for (T,N,3).
#     Returns v_arr, w_arr => (T-1, N, 3).
#
#     Args:
#         all_points_history: Point history of shape (T,N,3)
#         dt: Time step
#         num_neighbors: Number of neighbors for local estimation
#         use_savgol: Whether to apply Savitzky-Golay filtering
#         savgol_window: Window size for Savitzky-Golay filter
#         savgol_poly: Polynomial order for Savitzky-Golay filter
#         use_multi_frame: Whether to use multi-frame rigid fitting
#         window_radius: Radius for multi-frame fitting
#         v_max_threshold: Maximum allowed linear velocity (m/s)
#         w_max_threshold: Maximum allowed angular velocity (rad/s)
#         use_percentile: Use percentile-based outlier detection instead of fixed thresholds
#         outlier_percentile: Percentile threshold for outliers (only used if use_percentile is True)
#
#     Returns:
#         tuple: (v_arr, w_arr) linear and angular velocities of shape (T-1, N, 3)
#     """
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     if not isinstance(all_points_history, torch.Tensor):
#         all_points_history = torch.tensor(all_points_history, dtype=torch.float32, device=device)
#     T, N, _ = all_points_history.shape
#     if T < 2:
#         return None, None
#
#
#     # Neighbor-based estimation
#     pts_prev = all_points_history[:-1]
#     pts_curr = all_points_history[1:]
#     B = T - 1
#     neighbor_idx_prev = find_neighbors_batch(pts_prev, num_neighbors)
#     neighbor_idx_curr = find_neighbors_batch(pts_curr, num_neighbors)
#     K = num_neighbors
#
#     src_batch = pts_prev[
#                 torch.arange(B, device=device).view(B, 1, 1),
#                 neighbor_idx_prev,
#                 :
#                 ]
#     tar_batch = pts_curr[
#                 torch.arange(B, device=device).view(B, 1, 1),
#                 neighbor_idx_curr,
#                 :
#                 ]
#     src_2d = src_batch.reshape(B * N, K, 3)
#     tar_2d = tar_batch.reshape(B * N, K, 3)
#
#     R_2d = estimate_rotation_matrix_batch(src_2d, tar_2d)
#     c1_2d = src_2d.mean(dim=1)
#     c2_2d = tar_2d.mean(dim=1)
#     delta_p_2d = c2_2d - c1_2d
#
#     eye_4 = torch.eye(4, device=device).unsqueeze(0).expand(B * N, -1, -1).clone()
#     eye_4[:, :3, :3] = R_2d
#     eye_4[:, :3, 3] = delta_p_2d
#     transform_matrices_2d = eye_4
#     se3_logs_2d = se3_log_map_batch(transform_matrices_2d)
#     v_2d = se3_logs_2d[:, :3] / dt
#     w_2d = se3_logs_2d[:, 3:] / dt
#
#     v_arr = v_2d.reshape(B, N, 3).cpu().numpy()
#     w_arr = w_2d.reshape(B, N, 3).cpu().numpy()
#
#     # Convert to numpy arrays for further processing
#     v_arr = np.asarray(v_arr)
#     w_arr = np.asarray(w_arr)
#
#     # Outlier handling for both linear and angular velocities
#     # Compute velocity magnitudes
#     v_magnitudes = np.linalg.norm(v_arr, axis=2)  # (B, N)
#     w_magnitudes = np.linalg.norm(w_arr, axis=2)  # (B, N)
#
#     # Determine thresholds for outliers
#     if use_percentile and B * N > 10:  # Only use percentile if enough data points
#         v_threshold = np.percentile(v_magnitudes, outlier_percentile)
#         w_threshold = np.percentile(w_magnitudes, outlier_percentile)
#
#         # Cap the thresholds to reasonable values
#         v_threshold = min(v_threshold, v_max_threshold)
#         w_threshold = min(w_threshold, w_max_threshold)
#     else:
#         v_threshold = v_max_threshold
#         w_threshold = w_max_threshold
#
#     # Create masks for outliers
#     v_outlier_mask = v_magnitudes > v_threshold  # (B, N)
#     w_outlier_mask = w_magnitudes > w_threshold  # (B, N)
#
#     # Handle outliers in linear velocity
#     for b in range(B):
#         for n in range(N):
#             if v_outlier_mask[b, n]:
#                 # Set outlier velocities to zero or a reasonable value
#                 # Option 1: Set to zero
#                 v_arr[b, n, :] = 0.0
#
#                 # Option 2: Scale down to the threshold
#                 # scale_factor = v_threshold / v_magnitudes[b, n]
#                 # v_arr[b, n, :] *= scale_factor
#
#     # Handle outliers in angular velocity
#     for b in range(B):
#         for n in range(N):
#             if w_outlier_mask[b, n]:
#                 # Set outlier angular velocities to zero or a reasonable value
#                 # Option 1: Set to zero
#                 w_arr[b, n, :] = 0.0
#
#                 # Option 2: Scale down to the threshold
#                 # scale_factor = w_threshold / w_magnitudes[b, n]
#                 # w_arr[b, n, :] *= scale_factor
#
#     # Apply Savitzky-Golay filter if requested
#     if use_savgol and (T - 1) >= savgol_window:
#         v_arr = savgol_filter(
#             v_arr, window_length=savgol_window, polyorder=savgol_poly,
#             axis=0, mode='mirror'
#         )
#         w_arr = savgol_filter(
#             w_arr, window_length=savgol_window, polyorder=savgol_poly,
#             axis=0, mode='mirror'
#         )
#
#     return v_arr, w_arr

# def calculate_velocity_and_angular_velocity_for_all_frames(
#     all_points_history,
#     dt=0.1,
#     num_neighbors=400,
#     use_savgol=True,
#     savgol_window=5,
#     savgol_poly=2,
#     use_multi_frame=False,
#     window_radius=2
# ):
#     """
#     Compute linear and angular velocity for (T,N,3).
#     Returns v_arr, w_arr => (T-1, N, 3).
#     """
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     if not isinstance(all_points_history, torch.Tensor):
#         all_points_history = torch.tensor(all_points_history, dtype=torch.float32, device=device)
#     T, N, _ = all_points_history.shape
#     if T < 2:
#         return None, None
#
#     if use_multi_frame:
#         # Multi-frame rigid fit
#         v_list = []
#         w_list = []
#         for t in range(T - 1):
#             center_idx = t + 1
#             Tmat = multi_frame_rigid_fit(all_points_history, center_idx, window_radius)
#             Tmat_batch = Tmat.unsqueeze(0)
#             se3_logs = se3_log_map_batch(Tmat_batch)
#             se3_v = se3_logs[0, :3] / dt
#             se3_w = se3_logs[0, 3:] / dt
#             v_list.append(se3_v.unsqueeze(0).repeat(all_points_history.shape[1], 1))
#             w_list.append(se3_w.unsqueeze(0).repeat(all_points_history.shape[1], 1))
#         v_arr = torch.stack(v_list, dim=0).cpu().numpy()
#         w_arr = torch.stack(w_list, dim=0).cpu().numpy()
#     else:
#         # Neighbor-based estimation
#         pts_prev = all_points_history[:-1]
#         pts_curr = all_points_history[1:]
#         B = T - 1
#         neighbor_idx_prev = find_neighbors_batch(pts_prev, num_neighbors)
#         neighbor_idx_curr = find_neighbors_batch(pts_curr, num_neighbors)
#         K = num_neighbors
#
#         src_batch = pts_prev[
#             torch.arange(B, device=device).view(B, 1, 1),
#             neighbor_idx_prev,
#             :
#         ]
#         tar_batch = pts_curr[
#             torch.arange(B, device=device).view(B, 1, 1),
#             neighbor_idx_curr,
#             :
#         ]
#         src_2d = src_batch.reshape(B * N, K, 3)
#         tar_2d = tar_batch.reshape(B * N, K, 3)
#
#         R_2d = estimate_rotation_matrix_batch(src_2d, tar_2d)
#         c1_2d = src_2d.mean(dim=1)
#         c2_2d = tar_2d.mean(dim=1)
#         delta_p_2d = c2_2d - c1_2d
#
#         eye_4 = torch.eye(4, device=device).unsqueeze(0).expand(B * N, -1, -1).clone()
#         eye_4[:, :3, :3] = R_2d
#         eye_4[:, :3, 3] = delta_p_2d
#         transform_matrices_2d = eye_4
#         se3_logs_2d = se3_log_map_batch(transform_matrices_2d)
#         v_2d = se3_logs_2d[:, :3] / dt
#         w_2d = se3_logs_2d[:, 3:] / dt
#
#         v_arr = v_2d.reshape(B, N, 3).cpu().numpy()
#         w_arr = w_2d.reshape(B, N, 3).cpu().numpy()
#
#     if use_savgol and (T - 1) >= savgol_window:
#         v_arr = savgol_filter(
#             v_arr, window_length=savgol_window, polyorder=savgol_poly,
#             axis=0, mode='mirror'
#         )
#         w_arr = savgol_filter(
#             w_arr, window_length=savgol_window, polyorder=savgol_poly,
#             axis=0, mode='mirror'
#         )
#
#     return v_arr, w_arr
#

def point_to_line_distance_batch(points, line_origin, line_dir):
    """Calculate distances from points to a line (batch version)

    Args:
        points: Point coordinates (B, 3)
        line_origin: A point on the line (3,)
        line_dir: Line direction (unit vector) (3,)

    Returns:
        Distances (B,)
    """
    # Vector from origin to points
    vec = points - line_origin  # (B, 3)

    # Calculate cross product
    cross = torch.cross(vec, line_dir.expand_as(vec), dim=1)  # (B, 3)

    # Distance is the magnitude of the cross product
    dist = torch.norm(cross, dim=1)  # (B,)

    return dist


def compute_planar_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Estimate parameters for a planar joint.

    Args:
        all_points_history (ndarray): Point history of shape (T,N,3)
        v_history (ndarray): Linear velocity history of shape (T-1,N,3)
        omega_history (ndarray): Angular velocity history of shape (T-1,N,3)
        w_i (ndarray): Point weights of shape (N,)
        device (str): Device to use for computation

    Returns:
        dict: Dictionary containing planar joint parameters
    """
    global planar_normal_reference, plane_is_fixed

    all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    v_history = torch.as_tensor(v_history, dtype=torch.float32, device=device) if v_history is not None else None
    omega_history = torch.as_tensor(omega_history, dtype=torch.float32,
                                    device=device) if omega_history is not None else None
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)
    T, N = all_points_history.shape[0], all_points_history.shape[1]

    if T < 3:
        return {"normal": np.array([0., 0., 0.]), "motion_limit": (0., 0.)}

    if not plane_is_fixed:
        # Compute displacement vectors between consecutive frames
        displacements = all_points_history[1:] - all_points_history[:-1]  # (T-1, N, 3)

        # Use angular velocity vectors to estimate the plane normal (if available)
        if omega_history is not None and torch.sum(torch.abs(omega_history)) > 1e-6:
            # Use the mean direction of angular velocity as reference for normal estimation
            omega_mean = torch.mean(omega_history.reshape(-1, 3), dim=0)
            omega_norm = torch.norm(omega_mean)

            if omega_norm > 1e-6:
                # If there's significant rotational motion, use angular velocity as plane normal estimate
                planar_normal_reference = omega_mean / omega_norm
                plane_is_fixed = True
                return {
                    "normal": planar_normal_reference.cpu().numpy(),
                    "motion_limit": (1.0, 1.0)  # Use reasonable default values
                }

        # Only use displacements larger than a threshold
        disp_norms = torch.norm(displacements, dim=2)  # (T-1, N)
        disp_mean = torch.mean(disp_norms)
        mask = disp_norms > 0.1 * disp_mean  # Threshold set to 10% of mean displacement

        # If not enough valid displacements, use all of them
        if torch.sum(mask) < 10:
            mask = torch.ones_like(mask, dtype=torch.bool)

        # Reshape displacements using mask for filtering
        # Vectorized approach to filter valid displacements
        valid_mask_expanded = mask.unsqueeze(-1).expand(-1, -1, 3)
        valid_displacements = displacements[valid_mask_expanded].reshape(-1, 3)

        # Get corresponding weights
        valid_indices = torch.nonzero(mask, as_tuple=True)
        valid_weights = w_i[valid_indices[1]]  # Gather weights from point indices
        weight_sum = valid_weights.sum()

        if valid_displacements.shape[0] > 0:
            # Compute weighted covariance matrix using valid displacements
            weighted_disps = valid_displacements * valid_weights.unsqueeze(-1)
            cov_mat = torch.matmul(weighted_disps.T, valid_displacements) / weight_sum
        else:
            # Fallback to original PCA method
            pts_torch = all_points_history.reshape(-1, 3)
            weights_expanded = w_i.repeat(T)
            w_sum = weights_expanded.sum()

            # Compute weighted mean
            weighted_mean = torch.sum(weights_expanded.unsqueeze(-1) * pts_torch, dim=0) / w_sum

            # Center points
            centered = pts_torch - weighted_mean
            weighted_centered = centered * weights_expanded.unsqueeze(-1)

            # Compute weighted covariance matrix
            cov_mat = torch.matmul(weighted_centered.T, centered) / w_sum

        # Add regularization
        cov_mat += 1e-9 * torch.eye(3, device=device)

        # Eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(cov_mat)

        # Improvement: Choose the eigenvector corresponding to the smallest eigenvalue as normal
        normal_ = eigvecs[:, 0]  # Corresponding to smallest eigenvalue
        normal_ = normalize_vector_torch(normal_.unsqueeze(0))[0]

        # Get the two principal directions in the plane
        axis1_ = eigvecs[:, 2]  # Corresponding to largest eigenvalue
        axis2_ = eigvecs[:, 1]  # Corresponding to middle eigenvalue

        axis1_ = normalize_vector_torch(axis1_.unsqueeze(0))[0]
        axis2_ = normalize_vector_torch(axis2_.unsqueeze(0))[0]

        # Verify normal's correctness: should be orthogonal to both principal axes
        dot1 = torch.abs(torch.sum(normal_ * axis1_))
        dot2 = torch.abs(torch.sum(normal_ * axis2_))

        # If not orthogonal enough, recompute
        if dot1 > 0.1 or dot2 > 0.1:
            normal_ = torch.cross(axis1_, axis2_, dim=0)
            normal_ = normalize_vector_torch(normal_.unsqueeze(0))[0]

        # Save normal as reference
        planar_normal_reference = normal_.clone()
        plane_is_fixed = True
    else:
        normal_ = planar_normal_reference.clone()

    # Compute motion limits
    if T < 2:
        motion_limit = (0., 0.)
    else:
        pt0_0 = all_points_history[0, 0]
        pts = all_points_history[:, 0] - pt0_0

        # Calculate principal axes on the plane
        x_axis = torch.tensor([1., 0., 0.], device=device)
        y_axis = torch.tensor([0., 1., 0.], device=device)

        # Select a reference vector not parallel to the normal
        dot_x = torch.abs(torch.dot(normal_, x_axis))
        dot_y = torch.abs(torch.dot(normal_, y_axis))
        ref = y_axis if dot_x < dot_y else x_axis

        # Compute orthogonal vectors
        axis1_ = torch.linalg.cross(normal_, ref)
        # axis1_ = torch.cross(normal_, ref)
        axis1_ = normalize_vector_torch(axis1_.unsqueeze(0))[0]
        axis2_ = torch.linalg.cross(normal_, axis1_)
        # axis2_ = torch.cross(normal_, axis1_)
        axis2_ = normalize_vector_torch(axis2_.unsqueeze(0))[0]

        # Project onto principal axes
        proj1 = torch.matmul(pts, axis1_)
        proj2 = torch.matmul(pts, axis2_)

        # Get motion limits
        min_p1 = proj1.min().item()
        max_p1 = proj1.max().item()
        min_p2 = proj2.min().item()
        max_p2 = proj2.max().item()

        # Use maximum range as motion limit
        motion_limit = (max_p1 - min_p1, max_p2 - min_p2)

    return {
        "normal": normal_.cpu().numpy(),
        "motion_limit": motion_limit
    }


def compute_ball_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Estimate parameters for a ball joint.

    Args:
        all_points_history (ndarray): Point history of shape (T,N,3)
        v_history (ndarray): Linear velocity history of shape (T-1,N,3)
        omega_history (ndarray): Angular velocity history of shape (T-1,N,3)
        w_i (ndarray): Point weights of shape (N,)
        device (str): Device to use for computation

    Returns:
        dict: Dictionary containing ball joint parameters
    """
    all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    v_history = torch.as_tensor(v_history, dtype=torch.float32, device=device)
    omega_history = torch.as_tensor(omega_history, dtype=torch.float32, device=device)
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)

    T, N = all_points_history.shape[0], all_points_history.shape[1]
    if T < 2:
        return {"center": np.array([0., 0., 0.]), "motion_limit": (0., 0., 0.)}

    # Method 1: Estimate using angular velocity direction and velocity values
    # Compute radius = v / ω
    v_norm = torch.norm(v_history, dim=2)
    w_norm = torch.norm(omega_history, dim=2)
    EPS_W = 1e-3
    mask_w = (w_norm > EPS_W)
    r_mat = torch.zeros_like(v_norm)
    r_mat[mask_w] = v_norm[mask_w] / w_norm[mask_w]

    # Compute direction from cross product
    v_u = normalize_vector_torch(v_history)
    w_u = normalize_vector_torch(omega_history)
    dir_ = -torch.cross(v_u, w_u, dim=2)
    dir_ = normalize_vector_torch(dir_)

    # Estimate center position as point + r * direction
    r_3d = r_mat.unsqueeze(-1)
    c_pos = all_points_history[:-1] + dir_ * r_3d

    # Method 2: Direct geometric approach - find intersection of spheres
    # Vectorized approach for creating constraints
    # Each pair of points (p1, p2) must be equidistant from the center
    center_normals = []
    center_ds = []
    center_weights = []

    # Generate all frame pairs with the first frame
    for t in range(1, T):
        # Calculate differences between points at t=0 and t=current
        p1 = all_points_history[0].cpu()  # (N, 3)
        p2 = all_points_history[t].cpu()  # (N, 3)

        # Vectorized computation of plane equations: 2(p2-p1)·c = p2²-p1²
        eq_normals = 2 * (p2 - p1)  # (N, 3)
        eq_ds = torch.sum(p2 ** 2, dim=1) - torch.sum(p1 ** 2, dim=1)  # (N)

        # Calculate displacement norms
        disp_norms = torch.norm(eq_normals, dim=1)  # (N)

        # Filter significant displacements
        valid_mask = disp_norms > 1e-6
        if valid_mask.any():
            valid_normals = eq_normals[valid_mask]  # (K, 3)
            valid_ds = eq_ds[valid_mask]  # (K)
            valid_weights = w_i.cpu()[valid_mask]  # (K)

            # Normalize equations
            valid_normals = valid_normals / disp_norms[valid_mask].unsqueeze(-1)  # (K, 3)
            valid_ds = valid_ds / disp_norms[valid_mask]  # (K)

            center_normals.append(valid_normals)
            center_ds.append(valid_ds)
            center_weights.append(valid_weights)

    # Process collected constraints
    if center_normals and sum(len(n) for n in center_normals) > 10:
        # Concatenate all constraints
        A = torch.cat(center_normals, dim=0).numpy()  # (M, 3)
        b = torch.cat(center_ds, dim=0).numpy()  # (M)
        weights = torch.cat(center_weights, dim=0).numpy()  # (M)

        # Weighted least squares solution
        center_lstsq = np.linalg.lstsq(A * weights.reshape(-1, 1), b * weights, rcond=None)[0]
        center_direct = torch.tensor(center_lstsq, dtype=torch.float32, device=device)

        # Evaluate solution accuracy - calculate residuals
        residuals = np.abs(np.sum(A * center_lstsq, axis=1) - b)
        mean_residual = np.mean(residuals)

        # If direct method has small residuals, use it
        if mean_residual < 1e-2:
            center_sum = center_direct
        else:
            # Use Method 1 results with robust median filtering
            center_each = torch.mean(c_pos, dim=0)  # (N, 3)
            center_median = torch.median(center_each, dim=0).values  # (3)
            center_dists = torch.norm(center_each - center_median, dim=1)  # (N)
            median_dist = torch.median(center_dists)
            valid_centers = center_each[center_dists < 2.0 * median_dist]

            if valid_centers.shape[0] > 0:
                center_sum = torch.mean(valid_centers, dim=0)
            else:
                center_sum = center_median
    else:
        # If not enough candidates, use Method 1 results with robust median filtering
        center_each = torch.mean(c_pos, dim=0)
        center_median = torch.median(center_each, dim=0).values
        center_dists = torch.norm(center_each - center_median, dim=1)
        median_dist = torch.median(center_dists)
        valid_centers = center_each[center_dists < 2.0 * median_dist]

        if valid_centers.shape[0] > 0:
            center_sum = torch.mean(valid_centers, dim=0)
        else:
            center_sum = center_median

    # Calculate radius - use average distance to first frame points
    pts_0 = all_points_history[0]
    dists_0 = torch.norm(pts_0 - center_sum, dim=1)
    radius = torch.mean(dists_0).item()

    # Compute approximate motion limits (maximum angle)
    base_pt_0 = all_points_history[0, 0]
    base_vec = base_pt_0 - center_sum
    norm_b = torch.norm(base_vec) + 1e-6

    pts = all_points_history[:, 0, :]
    vecs = pts - center_sum

    # Calculate angles using dot product
    dotv = torch.sum(vecs * base_vec.unsqueeze(0), dim=1)
    norm_v = torch.norm(vecs, dim=1) + 1e-6

    cosval = torch.clamp(dotv / (norm_b * norm_v), -1., 1.)
    angles = torch.acos(cosval)
    max_angle = angles.max().item()

    return {
        "center": center_sum.cpu().numpy(),
        "radius": radius,
        "motion_limit": (max_angle, max_angle, max_angle)
    }


def compute_screw_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Estimate parameters for a screw joint.

    Args:
        all_points_history (ndarray): Point history of shape (T,N,3)
        v_history (ndarray): Linear velocity history of shape (T-1,N,3)
        omega_history (ndarray): Angular velocity history of shape (T-1,N,3)
        w_i (ndarray): Point weights of shape (N,)
        device (str): Device to use for computation

    Returns:
        dict: Dictionary containing screw joint parameters
    """
    global screw_axis_reference

    all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    v_history = torch.as_tensor(v_history, dtype=torch.float32, device=device)
    omega_history = torch.as_tensor(omega_history, dtype=torch.float32, device=device)
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)

    T, N = all_points_history.shape[0], all_points_history.shape[1]
    if T < 2:
        return {
            "axis": np.array([0., 0., 0.]),
            "origin": np.array([0., 0., 0.]),
            "pitch": 0.,
            "motion_limit": (0., 0.)
        }

    # Constants
    EPS_W = 1e-3

    # Step 1: Estimate rotation axis direction using angular velocity
    w_norm = torch.norm(omega_history, dim=2)  # (T-1, N)
    mask_w = (w_norm > EPS_W)  # (T-1, N)

    # Extract valid angular velocities and weights using the mask
    valid_mask = mask_w.reshape(-1)
    if valid_mask.any():
        # Reshape omega_history to (T-1*N, 3) for easier filtering
        reshaped_omega = omega_history.reshape(-1, 3)
        valid_omegas = reshaped_omega[valid_mask]  # (K, 3)

        # Get corresponding weights
        batch_indices, point_indices = torch.nonzero(mask_w, as_tuple=True)
        valid_weights = w_i[point_indices]  # (K)

        # Normalize all angular velocity vectors
        valid_omega_dirs = normalize_vector_torch(valid_omegas)  # (K, 3)

        # Weighted average to get axis direction
        axis_dir_raw = torch.sum(valid_omega_dirs * valid_weights.unsqueeze(1), dim=0)  # (3)
        axis_dir = normalize_vector_torch(axis_dir_raw.unsqueeze(0))[0]  # (3)
    else:
        # If no valid angular velocities, analyze point cloud trajectories using PCA
        # Stack all trajectories and apply PCA
        traj_tensor = all_points_history.permute(1, 0, 2)  # (N, T, 3)

        # Calculate trajectory directions using PCA
        traj_centered = traj_tensor - traj_tensor.mean(dim=1, keepdim=True)  # (N, T, 3)

        # Batch covariance calculation
        batch_cov = torch.matmul(traj_centered.transpose(1, 2), traj_centered)  # (N, 3, 3)

        # Add small regularization
        batch_cov += 1e-9 * torch.eye(3, device=device).unsqueeze(0)

        # Batch eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(batch_cov)  # (N, 3), (N, 3, 3)

        # Extract principal directions (largest eigenvalue)
        axis_candidates = eigvecs[:, :, 2]  # (N, 3)

        # Ensure consistent direction
        ref_dir = axis_candidates[0]  # (3)
        # Calculate dot products with reference direction
        dots = torch.sum(axis_candidates * ref_dir.unsqueeze(0), dim=1)  # (N)
        # Flip directions where dot product is negative
        flip_mask = dots < 0
        axis_candidates[flip_mask] = -axis_candidates[flip_mask]

        # Weighted average of all candidate directions
        axis_dir = torch.sum(axis_candidates * w_i.unsqueeze(1), dim=0)  # (3)
        axis_dir = normalize_vector_torch(axis_dir.unsqueeze(0))[0]  # (3)

    # Ensure direction consistency
    if screw_axis_reference is not None:
        if torch.dot(axis_dir, screw_axis_reference) < 0:
            axis_dir = -axis_dir
    screw_axis_reference = axis_dir.clone()

    # Step 2: Estimate screw axis position and pitch
    # Calculate velocity parallel and perpendicular components
    v_parallel = torch.sum(v_history * axis_dir, dim=2, keepdim=True) * axis_dir  # (T-1, N, 3)
    v_perp = v_history - v_parallel  # (T-1, N, 3)

    # For screw motion: v_perp = ω × r where r is perpendicular distance vector to axis
    # Extract points with valid angular velocities for axis position estimation
    valid_points_mask = mask_w.unsqueeze(-1).expand(-1, -1, 3)  # (T-1, N, 3)

    if valid_points_mask.any():
        # Reshape tensors for easier filtering
        pts_flat = all_points_history[:-1].reshape(-1, 3)  # ((T-1)*N, 3)
        v_perp_flat = v_perp.reshape(-1, 3)  # ((T-1)*N, 3)
        omega_flat = omega_history.reshape(-1, 3)  # ((T-1)*N, 3)
        w_norm_flat = w_norm.reshape(-1)  # ((T-1)*N)
        mask_flat = mask_w.reshape(-1)  # ((T-1)*N)

        # Extract valid points and values
        valid_pts = pts_flat[mask_flat]  # (K, 3)
        valid_v_perp = v_perp_flat[mask_flat]  # (K, 3)
        valid_omega = omega_flat[mask_flat]  # (K, 3)
        valid_w_norm = w_norm_flat[mask_flat]  # (K)

        # Calculate perpendicular distance vectors
        # r_perp = (w × v_perp) / w²
        w_cross_v = torch.cross(valid_omega, valid_v_perp, dim=1)  # (K, 3)
        r_perp = w_cross_v / (valid_w_norm.unsqueeze(-1) ** 2)  # (K, 3)

        # Calculate points on axis: p - r_perp
        origin_candidates = valid_pts - r_perp  # (K, 3)

        # Project all estimated axis points to a plane perpendicular to axis_dir
        # Select perpendicular basis vectors
        if abs(axis_dir[0]) < abs(axis_dir[1]) and abs(axis_dir[0]) < abs(axis_dir[2]):
            perp1 = torch.tensor([0., axis_dir[2], -axis_dir[1]], device=device)
        else:
            perp1 = torch.tensor([axis_dir[2], 0., -axis_dir[0]], device=device)

        perp1 = normalize_vector_torch(perp1.unsqueeze(0))[0]
        # perp2 = torch.cross(axis_dir, perp1)
        perp2 = torch.linalg.cross(axis_dir, perp1)

        # Project to plane (calculate 2D coordinates)
        proj1 = torch.matmul(origin_candidates, perp1)  # (K)
        proj2 = torch.matmul(origin_candidates, perp2)  # (K)
        proj_coords = torch.stack([proj1, proj2], dim=1)  # (K, 2)

        # Calculate center of projections (weighted)
        # Get point indices from mask for weights
        _, point_indices = torch.nonzero(mask_w, as_tuple=True)
        candidate_weights = w_i[point_indices]  # (K)

        # Weighted center
        weighted_coords = proj_coords * candidate_weights.unsqueeze(1)  # (K, 2)
        center_proj = torch.sum(weighted_coords, dim=0) / torch.sum(candidate_weights)  # (2)

        # Transform back to 3D
        origin_sum = center_proj[0] * perp1 + center_proj[1] * perp2  # (3)

        # Step 3: Calculate pitch (pitch = axial velocity / angular velocity)
        # Extract valid pitch values
        v_along_axis = torch.sum(v_history * axis_dir, dim=2)  # (T-1, N)

        # Filter valid values
        valid_v_axis = v_along_axis[mask_w]  # (K)
        valid_pitches = valid_v_axis / valid_w_norm  # (K)

        # Weighted average of pitch values
        pitch_sum = torch.sum(valid_pitches * candidate_weights) / torch.sum(candidate_weights)
    else:
        # If no valid data, use default values
        origin_sum = all_points_history[0, 0]

        # Project to axis
        proj = torch.dot(origin_sum, axis_dir) * axis_dir
        perp = origin_sum - proj
        origin_sum = origin_sum - perp  # Project point to axis

        pitch_sum = torch.tensor(0.0, device=device)

    # Calculate motion limits - angle range of a point around the axis
    i0 = 0
    base_pt = all_points_history[0, i0]  # (3)

    # Calculate perpendicular vector from base point to axis
    proj_base = torch.dot(base_pt - origin_sum, axis_dir) * axis_dir  # (3)
    perp_base = base_pt - origin_sum - proj_base  # (3)
    perp_base_norm = torch.norm(perp_base)  # scalar

    if perp_base_norm > 1e-6:
        perp_base_dir = perp_base / perp_base_norm  # (3)

        # Calculate angles for all frames
        angles = []

        # Extract all points for the reference point across frames
        pts = all_points_history[:, i0]  # (T, 3)

        # Calculate perpendicular components for all frames
        pts_to_origin = pts - origin_sum  # (T, 3)
        projs = torch.sum(pts_to_origin * axis_dir, dim=1, keepdim=True) * axis_dir  # (T, 3)
        perps = pts_to_origin - projs  # (T, 3)
        perp_norms = torch.norm(perps, dim=1)  # (T)

        # Filter valid perpendicular components
        valid_mask = perp_norms > 1e-6
        if valid_mask.any():
            valid_perps = perps[valid_mask]  # (M, 3)
            valid_perp_norms = perp_norms[valid_mask]  # (M)

            # Normalize perpendicular vectors
            valid_perp_dirs = valid_perps / valid_perp_norms.unsqueeze(1)  # (M, 3)

            # Calculate dot products with base direction
            cos_angles = torch.sum(valid_perp_dirs * perp_base_dir, dim=1)  # (M)
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            valid_angles = torch.acos(cos_angles)  # (M)

            # Determine angle signs
            cross_prods = torch.cross(perp_base_dir.unsqueeze(0).expand_as(valid_perp_dirs), valid_perp_dirs,
                                      dim=1)  # (M, 3)
            signs = torch.sign(torch.sum(cross_prods * axis_dir, dim=1))  # (M)
            valid_angles = valid_angles * signs  # (M)

            # Convert to list for min/max calculation
            angles = valid_angles.cpu().tolist()

        if angles:
            min_angle = min(angles)
            max_angle = max(angles)
            motion_limit = (min_angle, max_angle)
        else:
            motion_limit = (0.0, 0.0)
    else:
        motion_limit = (0.0, 0.0)

    return {
        "axis": axis_dir.cpu().numpy(),
        "origin": origin_sum.cpu().numpy(),
        "pitch": pitch_sum.cpu().numpy().item(),
        "motion_limit": motion_limit
    }


def compute_prismatic_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Estimate parameters for a prismatic joint.

    Args:
        all_points_history (ndarray): Point history of shape (T,N,3)
        v_history (ndarray): Linear velocity history of shape (T-1,N,3)
        omega_history (ndarray): Angular velocity history of shape (T-1,N,3)
        w_i (ndarray): Point weights of shape (N,)
        device (str): Device to use for computation

    Returns:
        dict: Dictionary containing prismatic joint parameters
    """
    global prismatic_axis_reference

    all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    # v_history = torch.as_tensor(v_history, dtype=torch.float32, device=device) if v_history is not None else None
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)

    T, N = all_points_history.shape[0], all_points_history.shape[1]
    if T < 2:
        return {
            "axis": np.array([0., 0., 0.]),
            "origin": np.array([0., 0., 0.]),
            "motion_limit": (0., 0.)
        }

    # Method 1: Use displacement vectors to directly estimate axis direction
    # Calculate displacement vectors between consecutive frames
    displacements = all_points_history[1:] - all_points_history[:-1]  # (T-1, N, 3)

    # Keep only significant displacement vectors
    disp_norms = torch.norm(displacements, dim=2)  # (T-1, N)
    mean_disp = torch.mean(disp_norms)
    mask = disp_norms > 0.1 * mean_disp  # (T-1, N)

    if mask.any():
        # Reshape for easier filtering
        disp_flat = displacements.reshape(-1, 3)  # ((T-1)*N, 3)
        mask_flat = mask.reshape(-1)  # ((T-1)*N)

        # Extract valid displacements
        valid_disps = disp_flat[mask_flat]  # (K, 3)

        # Get corresponding weights for each displacement
        batch_indices, point_indices = torch.nonzero(mask, as_tuple=True)
        valid_weights = w_i[point_indices]  # (K)

        # Normalize displacement vectors
        disp_dirs = normalize_vector_torch(valid_disps)  # (K, 3)

        # Weighted average
        axis_direct = torch.sum(disp_dirs * valid_weights.unsqueeze(1), dim=0)  # (3)
        axis_direct = normalize_vector_torch(axis_direct.unsqueeze(0))[0]  # (3)

        # Calculate axis alignment score
        alignment_scores = torch.abs(torch.sum(disp_dirs * axis_direct.unsqueeze(0), dim=1))  # (K)
        mean_alignment = torch.mean(alignment_scores).item()

        # If alignment is high, use direct method
        if mean_alignment > 0.9:
            axis_sum = axis_direct
        else:
            # Otherwise use PCA method (potentially more stable)
            # Reshape for point-wise processing
            pos_history_n_tc = all_points_history.permute(1, 0, 2).contiguous()  # (N, T, 3)

            # Center points
            mean_pos = pos_history_n_tc.mean(dim=1, keepdim=True)  # (N, 1, 3)
            centered = pos_history_n_tc - mean_pos  # (N, T, 3)

            # Compute batch covariance matrices
            covs = torch.bmm(centered.transpose(1, 2), centered)  # (N, 3, 3)

            # Add regularization
            eye_batch = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
            covs = covs + 1e-9 * eye_batch

            # Batch eigendecomposition
            eigvals, eigvecs = torch.linalg.eigh(covs)  # (N, 3), (N, 3, 3)

            # Extract principal directions
            max_vecs = eigvecs[:, :, 2]  # (N, 3)
            max_vecs = normalize_vector_torch(max_vecs)  # (N, 3)

            # Ensure direction consistency
            ref_vec = max_vecs[0]  # (3)
            # Calculate dot products with reference
            dots = torch.sum(max_vecs * ref_vec.unsqueeze(0), dim=1)  # (N)
            # Flip where needed
            flip_mask = dots < 0
            max_vecs[flip_mask] = -max_vecs[flip_mask]

            # Compute weighted direction with squared weights
            confidence_weight = torch.pow(w_i, 2.0)  # (N)
            weighted_dir = torch.sum(max_vecs * confidence_weight.unsqueeze(-1), dim=0)  # (3)
            axis_sum = normalize_vector_torch(weighted_dir.unsqueeze(0))[0]  # (3)

            # If PCA and direct methods agree, increase confidence
            if torch.dot(axis_sum, axis_direct) > 0.9:
                # Use original method result
                pass
            else:
                # If methods disagree, take the one with larger dot product
                if torch.dot(axis_direct, axis_direct) > torch.dot(axis_sum, axis_sum):
                    axis_sum = axis_direct
    else:
        # If no valid displacements, use Method 2 (PCA)
        # Reshape for point-wise processing
        pos_history_n_tc = all_points_history.permute(1, 0, 2).contiguous()  # (N, T, 3)

        # Center points
        mean_pos = pos_history_n_tc.mean(dim=1, keepdim=True)  # (N, 1, 3)
        centered = pos_history_n_tc - mean_pos  # (N, T, 3)

        # Compute batch covariance matrices
        covs = torch.bmm(centered.transpose(1, 2), centered)  # (N, 3, 3)

        # Add regularization
        eye_batch = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)
        covs = covs + 1e-9 * eye_batch

        # Batch eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(covs)  # (N, 3), (N, 3, 3)

        # Extract principal directions
        max_vecs = eigvecs[:, :, 2]  # (N, 3)
        max_vecs = normalize_vector_torch(max_vecs)  # (N, 3)

        # Ensure direction consistency
        ref_vec = max_vecs[0]  # (3)
        # Calculate dot products with reference
        dots = torch.sum(max_vecs * ref_vec.unsqueeze(0), dim=1)  # (N)
        # Flip where needed
        flip_mask = dots < 0
        max_vecs[flip_mask] = -max_vecs[flip_mask]

        # Compute weighted direction - use squared weights for emphasis
        confidence_weight = torch.pow(w_i, 2.0)  # (N)
        weighted_dir = torch.sum(max_vecs * confidence_weight.unsqueeze(-1), dim=0)  # (3)
        axis_sum = normalize_vector_torch(weighted_dir.unsqueeze(0))[0]  # (3)

    # Ensure direction consistency
    if prismatic_axis_reference is None:
        prismatic_axis_reference = axis_sum.clone()
    else:
        dot_val = torch.dot(axis_sum, prismatic_axis_reference)
        if dot_val < 0:
            axis_sum = -axis_sum

    # Use first frame first point as origin
    origin = all_points_history[0, 0]  # (3)

    # Calculate motion limits - displacement along axis direction
    pts = all_points_history[:, 0, :]  # (T, 3)

    # Calculate projections to axis
    origin_to_pts = pts - origin  # (T, 3)
    projections = torch.sum(origin_to_pts * axis_sum, dim=1)  # (T)

    min_proj = float(projections.min().item())
    max_proj = float(projections.max().item())

    return {
        "axis": axis_sum.cpu().numpy(),
        "origin": origin.cpu().numpy(),
        "motion_limit": (min_proj, max_proj)
    }


def normalize_vector_torch(v, eps=1e-6):
    """向量归一化（批处理版本）"""
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    mask = (norm_v > eps).squeeze(-1)
    out = torch.zeros_like(v)
    out[mask] = v[mask] / norm_v[mask]
    return out

# def compute_revolute_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
#     """
#     Estimate parameters for a revolute joint.
#
#     Args:
#         all_points_history (ndarray): Point history of shape (T,N,3)
#         v_history (ndarray): Linear velocity history of shape (T-1,N,3)
#         omega_history (ndarray): Angular velocity history of shape (T-1,N,3)
#         w_i (ndarray): Point weights of shape (N,)
#         device (str): Device to use for computation
#
#     Returns:
#         dict: Dictionary containing revolute joint parameters
#     """
#     global revolute_axis_reference
#
#     all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
#     v_history = torch.as_tensor(v_history, dtype=torch.float32, device=device)
#     omega_history = torch.as_tensor(omega_history, dtype=torch.float32, device=device)
#     w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)
#
#     T, N = all_points_history.shape[0], all_points_history.shape[1]
#     if T < 2:
#         return {
#             "axis": np.array([0., 0., 0.]),
#             "origin": np.array([0., 0., 0.]),
#             "motion_limit": (0., 0.)
#         }
#
#     B = T - 1
#     if B < 1:
#         return {
#             "axis": np.array([0., 0., 0.]),
#             "origin": np.array([0., 0., 0.]),
#             "motion_limit": (0., 0.)
#         }
#
#     # Reshape for point-wise processing
#     omega_nbc = omega_history.permute(1, 0, 2).contiguous()
#
#     # Compute covariance matrices of angular velocities
#     covs = torch.einsum('ibm,ibn->imn', omega_nbc, omega_nbc)
#     covs += 1e-9 * torch.eye(3, device=device)
#
#     # Eigen decomposition
#     eigvals, eigvecs = torch.linalg.eigh(covs)
#
#     # Extract principal directions
#     max_vecs = eigvecs[:, :, 2]
#     max_vecs = normalize_vector_torch(max_vecs)
#
#     # Compute weighted axis
#     revolve_axis = torch.sum(max_vecs * w_i.unsqueeze(-1), dim=0)
#     revolve_axis = normalize_vector_torch(revolve_axis.unsqueeze(0))[0]
#
#     # Ensure consistent direction with previous estimations
#     if revolute_axis_reference is None:
#         revolute_axis_reference = revolve_axis.clone()
#     else:
#         dot_val = torch.dot(revolve_axis, revolute_axis_reference)
#         if dot_val < 0:
#             revolve_axis = -revolve_axis
#
#     # Compute radius and center
#     eps_ = 1e-8
#     v_norm = torch.norm(v_history, dim=2)
#     w_norm = torch.norm(omega_history, dim=2)
#     mask_w = (w_norm > eps_)
#
#     r_mat = torch.zeros_like(v_norm)
#     r_mat[mask_w] = v_norm[mask_w] / w_norm[mask_w]
#
#     v_u = normalize_vector_torch(v_history)
#     w_u = normalize_vector_torch(omega_history)
#
#     dir_ = -torch.cross(v_u, w_u, dim=2)
#     dir_ = normalize_vector_torch(dir_)
#
#     r_3d = r_mat.unsqueeze(-1)
#     c_pos = all_points_history[:-1] + dir_ * r_3d
#
#     # Reshape for robustness
#     c_pos_nbc = c_pos.permute(1, 0, 2).contiguous()
#
#     # Use median for robust center estimation
#     c_pos_median = c_pos_nbc.median(dim=1).values
#     dev = torch.norm(c_pos_nbc - c_pos_median.unsqueeze(1), dim=2)
#     scale = dev.median(dim=1).values + 1e-6
#
#     # Compute weights based on deviation
#     ratio = dev / scale.unsqueeze(-1)
#     # w_r = 1.0 / (1.0 + ratio * ratio)
#     w_r = 1.0 / (1.0 + torch.pow(ratio, 3.0))  # 使用更高次幂增强离群点抑制
#     w_r_3d = w_r.unsqueeze(-1)
#
#     # Compute weighted center
#     c_pos_weighted = c_pos_nbc * w_r_3d
#     sum_pos = c_pos_weighted.sum(dim=1)
#     sum_w = w_r.sum(dim=1, keepdim=True) + 1e-6
#
#     origin_each = sum_pos / sum_w
#     origin_sum = torch.sum(origin_each * w_i.unsqueeze(-1), dim=0)
#
#     # Compute motion limits
#     i0 = 0
#     base_pt = all_points_history[0, i0]
#     base_vec = base_pt - origin_sum
#     nb = torch.norm(base_vec) + 1e-6
#
#     pts = all_points_history[:, i0, :]
#     vecs = pts - origin_sum
#
#     dotv = torch.sum(vecs * base_vec.unsqueeze(0), dim=1)
#     norm_v = torch.norm(vecs, dim=1) + 1e-6
#
#     cosval = torch.clamp(dotv / (nb * norm_v), -1., 1.)
#     angles = torch.acos(cosval)
#     min_a = float(angles.min().item())
#     max_a = float(angles.max().item())
#
#     return {
#         "axis": revolve_axis.cpu().numpy(),
#         "origin": origin_sum.cpu().numpy(),
#         "motion_limit": (min_a, max_a)
#     }


def compute_revolute_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Estimate parameters for a revolute joint:
    - Use all points' trajectories to estimate axis direction
    - Use the most significant point to estimate axis position

    Args:
        all_points_history (ndarray): Point history of shape (T,N,3)
        v_history (ndarray): Linear velocity history of shape (T-1,N,3)
        omega_history (ndarray): Angular velocity history of shape (T-1,N,3)
        w_i (ndarray): Point weights of shape (N,)
        device (str): Device to use for computation

    Returns:
        dict: Dictionary containing revolute joint parameters
    """
    global revolute_axis_reference

    # 转换为张量
    all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)

    # 获取形状
    T, N, _ = all_points_history.shape
    if T < 3:
        return {
            "axis": np.array([0., 0., 0.]),
            "origin": np.array([0., 0., 0.]),
            "motion_limit": (0., 0.)
        }

    # 1. 使用所有点估计轴方向
    point_trajectories = all_points_history.permute(1, 0, 2)  # (N, T, 3)
    trajectory_means = torch.mean(point_trajectories, dim=1, keepdim=True)  # (N, 1, 3)
    centered_trajectories = point_trajectories - trajectory_means  # (N, T, 3)

    # 批量计算协方差矩阵
    cov_matrices = torch.bmm(centered_trajectories.transpose(1, 2), centered_trajectories)  # (N, 3, 3)
    cov_matrices = cov_matrices + 1e-9 * torch.eye(3, device=device).unsqueeze(0)  # 正则化

    # 批量SVD分解
    U, S, V = torch.linalg.svd(cov_matrices)

    # 获取最小特征值对应的特征向量
    min_eigval_idx = torch.argmin(S, dim=1)  # (N,)
    axes = torch.stack([V[i, idx] for i, idx in enumerate(min_eigval_idx)], dim=0)  # (N, 3)

    # 归一化
    axes = axes / (torch.norm(axes, dim=1, keepdim=True) + 1e-6)

    # 确保轴方向一致性
    reference_axis = axes[0]
    dot_products = torch.sum(axes * reference_axis.unsqueeze(0), dim=1)
    axes = torch.where(dot_products.unsqueeze(1) < 0, -axes, axes)

    # 使用运动显著性加权平均
    revolve_axis = torch.sum(axes * w_i.unsqueeze(1), dim=0)  # (3)
    revolve_axis = revolve_axis / torch.norm(revolve_axis)  # 归一化

    # 确保与先前估计的轴方向一致
    if revolute_axis_reference is not None:
        if torch.dot(revolve_axis, revolute_axis_reference) < 0:
            revolve_axis = -revolve_axis
    else:
        revolute_axis_reference = revolve_axis.clone()

    # 2. 找到运动最大的点作为参考点
    displacements = torch.diff(point_trajectories, dim=1)  # (N, T-1, 3)
    total_movement = torch.sum(torch.norm(displacements, dim=2), dim=1)  # (N,)
    most_significant_idx = torch.argmax(total_movement).item()
    significant_points = point_trajectories[most_significant_idx]  # (T, 3)

    # 3. 构建垂直于轴的坐标系
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device)

    x_cross = torch.linalg.cross(revolve_axis, x_axis)
    y_cross = torch.linalg.cross(revolve_axis, y_axis)

    if torch.norm(x_cross) > torch.norm(y_cross):
        basis1 = normalize_vector_torch(x_cross.unsqueeze(0))[0]
    else:
        basis1 = normalize_vector_torch(y_cross.unsqueeze(0))[0]

    basis2 = torch.linalg.cross(revolve_axis, basis1)
    basis2 = normalize_vector_torch(basis2.unsqueeze(0))[0]

    # 4. 投影显著点到垂直平面
    trajectory_mean = torch.mean(significant_points, dim=0)
    centered_points = significant_points - trajectory_mean

    projected_points = torch.zeros((T, 2), device=device)
    for i in range(T):
        p = centered_points[i]
        # 减去轴向分量
        p_proj = p - torch.dot(p, revolve_axis) * revolve_axis
        # 在基向量上的投影作为坐标
        x_coord = torch.dot(p_proj, basis1)
        y_coord = torch.dot(p_proj, basis2)
        projected_points[i] = torch.tensor([x_coord, y_coord])

    # 5. 代数圆拟合
    def fit_circle_algebraic(points):
        """使用最小二乘法的代数圆拟合"""
        points = points.cpu().numpy()
        A = np.column_stack([
            points[:, 0] * 2,
            points[:, 1] * 2,
            np.ones(len(points))
        ])
        b = points[:, 0] ** 2 + points[:, 1] ** 2

        try:
            solution = np.linalg.lstsq(A, b, rcond=None)[0]
            center_x, center_y = solution[0], solution[1]
            c = solution[2]
            radius = np.sqrt(c + center_x ** 2 + center_y ** 2)
            return np.array([center_x, center_y]), radius
        except:
            return None, None

    def compute_circle_residuals(points, center, radius):
        """计算点到圆的残差"""
        points_np = points.cpu().numpy()
        dists = np.sqrt(np.sum((points_np - center.reshape(1, 2)) ** 2, axis=1))
        return np.abs(dists - radius)

    # 6. 使用RANSAC拟合圆
    def ransac_circle_fit(points, max_iterations=100, distance_threshold=0.05, min_inliers_ratio=0.5):
        points_np = points.cpu().numpy()
        best_inliers = 0
        best_center = None
        best_radius = None
        n_points = len(points_np)
        min_inliers = int(n_points * min_inliers_ratio)

        # 首先尝试全局拟合
        center, radius = fit_circle_algebraic(points)
        if center is not None:
            residuals = compute_circle_residuals(points, center, radius)
            inliers = np.sum(residuals < distance_threshold)
            if inliers >= min_inliers:
                best_center = center
                best_radius = radius
                best_inliers = inliers

        # 如果全局拟合失败，使用RANSAC
        if best_center is None:
            for _ in range(max_iterations):
                sample_indices = np.random.choice(n_points, min(n_points, 5), replace=False)
                sample_points = points[sample_indices]

                center, radius = fit_circle_algebraic(sample_points)
                if center is None:
                    continue

                residuals = compute_circle_residuals(points, center, radius)
                inliers = np.sum(residuals < distance_threshold)

                if inliers > best_inliers:
                    best_inliers = inliers
                    best_center = center
                    best_radius = radius

        return best_center, best_radius, best_inliers

    # 执行圆拟合
    best_center, best_radius, n_inliers = ransac_circle_fit(
        projected_points,
        max_iterations=100,
        distance_threshold=0.05,
        min_inliers_ratio=0.5
    )

    # 7. 转换回3D坐标
    if best_center is not None and n_inliers > max(3, T // 4):
        center_3d = (
                trajectory_mean +
                best_center[0] * basis1 +
                best_center[1] * basis2
        )
    else:
        # 回退方案
        center_3d = trajectory_mean
        axis_proj = torch.dot(center_3d, revolve_axis) * revolve_axis
        center_3d = axis_proj

    # 8. 计算运动限制
    base_pt = significant_points[0]
    base_vec = base_pt - center_3d
    base_vec = base_vec - torch.dot(base_vec, revolve_axis) * revolve_axis
    base_norm = torch.norm(base_vec) + 1e-6

    angles = []
    for i in range(T):
        pos = significant_points[i]
        vec = pos - center_3d
        vec = vec - torch.dot(vec, revolve_axis) * revolve_axis
        vec_norm = torch.norm(vec) + 1e-6

        cos_angle = torch.clamp(
            torch.dot(vec, base_vec) / (vec_norm * base_norm),
            -1., 1.
        )
        angle = torch.acos(cos_angle)

        cross_prod = torch.linalg.cross(base_vec, vec)
        sign = torch.sign(torch.dot(cross_prod, revolve_axis))
        signed_angle = angle * sign
        angles.append(signed_angle.item())

    min_angle = min(angles) if angles else 0.0
    max_angle = max(angles) if angles else 0.0

    return {
        "axis": revolve_axis.cpu().numpy(),
        "origin": center_3d.cpu().numpy(),
        "motion_limit": (min_angle, max_angle)
    }




# def compute_revolute_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
#     """
#     使用运动最显著的一个点计算旋转关节参数
#     """
#     import math
#     global revolute_axis_reference
#
#     # 转换为张量
#     all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
#     w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)
#
#     # 获取形状
#     T, N, _ = all_points_history.shape
#
#     if T < 3:
#         return {
#             "axis": np.array([0., 0., 0.]),
#             "origin": np.array([0., 0., 0.]),
#             "motion_limit": (0., 0.)
#         }
#
#     # 1. 找出运动最显著的点
#     # 计算每个点的总位移作为运动显著性指标
#     point_trajectories = all_points_history.permute(1, 0, 2)  # (N, T, 3)
#     displacements = torch.diff(point_trajectories, dim=1)  # (N, T-1, 3)
#     total_movement = torch.sum(torch.norm(displacements, dim=2), dim=1)  # (N,)
#
#     # 找出运动最显著的点索引
#     most_significant_idx = torch.argmax(total_movement).item()
#
#     # 获取最显著点的轨迹
#     significant_trajectory = point_trajectories[most_significant_idx]  # (T, 3)
#
#     # 2. 使用最显著点计算旋转轴
#     # 中心化轨迹
#     trajectory_mean = torch.mean(significant_trajectory, dim=0, keepdim=True)  # (1, 3)
#     centered_trajectory = significant_trajectory - trajectory_mean  # (T, 3)
#
#     # 计算协方差矩阵
#     cov_matrix = torch.matmul(centered_trajectory.transpose(0, 1), centered_trajectory)  # (3, 3)
#     cov_matrix = cov_matrix + 1e-9 * torch.eye(3, device=device)  # 添加正则化
#
#     # SVD分解
#     U, S, V = torch.linalg.svd(cov_matrix)
#
#     # 获取最小特征值对应的特征向量作为旋转轴方向
#     min_eigval_idx = torch.argmin(S)
#     axis = V[min_eigval_idx]
#
#     # 归一化轴方向
#     axis = axis / torch.norm(axis)
#
#     # 确保与先前估计的轴方向一致
#     if revolute_axis_reference is not None:
#         dot_val = torch.dot(axis, revolute_axis_reference)
#         if dot_val < 0:
#             axis = -axis
#     else:
#         revolute_axis_reference = axis.clone()
#
#     # 3. 使用最显著点的三个连续帧计算旋转中心
#     centers = []
#
#     # 专用Z轴旋转函数
#     def rotateZ(vec, rad):
#         return torch.tensor([
#             vec[0] * math.cos(rad) - vec[1] * math.sin(rad),
#             vec[0] * math.sin(rad) + vec[1] * math.cos(rad),
#             vec[2]
#         ], dtype=torch.float32, device=device)
#
#     # 对每三个连续帧计算旋转中心
#     for t in range(T - 2):
#         # 获取三个连续帧的位置
#         c1 = significant_trajectory[t]  # (3)
#         c2 = significant_trajectory[t + 1]  # (3)
#         c3 = significant_trajectory[t + 2]  # (3)
#
#         # 计算位移向量
#         v_12 = c2 - c1  # (3)
#         v_13 = c3 - c1  # (3)
#
#         # 计算法向量
#         n_ = torch.linalg.cross(v_12, v_13)  # (3)
#         nn = torch.norm(n_)
#
#         # 跳过数值不稳定的情况
#         if nn < 1e-6:
#             continue
#
#         n_ = n_ / nn
#
#         # 构建平面坐标系
#         plane_x = v_12 / (torch.norm(v_12) + 1e-6)
#         plane_y = v_13 - torch.dot(v_13, plane_x) * plane_x
#         ny = torch.norm(plane_y)
#
#         if ny < 1e-6:
#             continue
#
#         plane_y = plane_y / ny
#         plane_z = torch.linalg.cross(plane_x, plane_y)
#
#         # 构建坐标系变换矩阵
#         plane_rot = torch.eye(4, dtype=torch.float32, device=device)
#         plane_rot[:3, 0] = plane_x
#         plane_rot[:3, 1] = plane_y
#         plane_rot[:3, 2] = plane_z
#         plane_rot[:3, 3] = c1
#
#         # 坐标系逆变换
#         plane_inv = torch.eye(4, dtype=torch.float32, device=device)
#         plane_inv[:3, :3] = plane_rot[:3, :3].transpose(0, 1)
#         plane_inv[:3, 3] = -torch.matmul(plane_rot[:3, :3].transpose(0, 1), c1)
#
#         # 将点变换到局部坐标系
#         p1_local = torch.cat([c1, torch.tensor([1.0], device=device)])
#         p2_local = torch.cat([c2, torch.tensor([1.0], device=device)])
#         p3_local = torch.cat([c3, torch.tensor([1.0], device=device)])
#
#         p1_local = torch.matmul(plane_inv, p1_local)
#         p2_local = torch.matmul(plane_inv, p2_local)
#         p3_local = torch.matmul(plane_inv, p3_local)
#
#         # 计算中点
#         c1_ = 0.5 * (p1_local[:3] + p2_local[:3])
#         c2_ = 0.5 * (p1_local[:3] + p3_local[:3])
#
#         # 使用rotateZ计算垂直方向
#         p21 = rotateZ(p2_local[:3] - p1_local[:3], math.pi / 2.0)
#         p43 = rotateZ(p3_local[:3] - p1_local[:3], math.pi / 2.0)
#
#         # 设置方程组 A * x = b - 只使用XY坐标
#         A = torch.stack([p21[:2], -p43[:2]], dim=1)
#         b = (c2_[:2] - c1_[:2])
#
#         # 求解中垂线交点
#         try:
#             if torch.abs(torch.det(A)) < 1e-6:
#                 continue
#
#             lam = torch.linalg.solve(A, b)
#             onplane_center = c1_[:2] + lam[0] * p21[:2]
#
#             # 重建三维点 - 平面坐标系下的中心点
#             plane_center_4 = torch.tensor([
#                 onplane_center[0].item(),
#                 onplane_center[1].item(),
#                 c1_[2].item(),
#                 1.0
#             ], dtype=torch.float32, device=device)
#
#             # 转换回世界坐标系
#             center_world_4 = torch.matmul(plane_rot, plane_center_4)
#             center = center_world_4[:3]
#
#             centers.append(center)
#         except:
#             continue
#
#     # 如果有计算结果，取平均值作为最终旋转中心
#     if len(centers) > 0:
#         centers = torch.stack(centers, dim=0)  # (K, 3)
#         origin = torch.mean(centers, dim=0)  # (3)
#     else:
#         # 如果没有有效计算结果，使用最显著点第一帧位置作为近似中心
#         origin = significant_trajectory[0]
#
#     # 4. 使用最显著点计算运动限制
#     base_pt = significant_trajectory[0]
#     base_vec = base_pt - origin
#     nb = torch.norm(base_vec) + 1e-6
#
#     vecs = significant_trajectory - origin.unsqueeze(0)  # (T, 3)
#
#     dotv = torch.sum(vecs * base_vec.unsqueeze(0), dim=1)  # (T,)
#     norm_v = torch.norm(vecs, dim=1) + 1e-6  # (T,)
#
#     cosval = torch.clamp(dotv / (nb * norm_v), -1., 1.)  # (T,)
#     angles = torch.acos(cosval)  # (T,)
#     min_a = float(angles.min().item())
#     max_a = float(angles.max().item())
#
#     return {
#         "axis": axis.cpu().numpy(),
#         "origin": origin.cpu().numpy(),
#         "motion_limit": (min_a, max_a)
#     }

# def compute_revolute_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
#     """
#     使用PyTorch高效计算旋转关节参数
#     """
#     global revolute_axis_reference
#
#     # 转换为张量
#     all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
#     w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)
#
#     # 获取形状
#     T, N, _ = all_points_history.shape
#
#     # 1. 计算旋转轴方向 - 使用批处理SVD
#     point_trajectories = all_points_history.permute(1, 0, 2)  # (N, T, 3)
#     trajectory_means = torch.mean(point_trajectories, dim=1, keepdim=True)  # (N, 1, 3)
#     centered_trajectories = point_trajectories - trajectory_means  # (N, T, 3)
#
#     # 批量计算协方差矩阵
#     cov_matrices = torch.bmm(centered_trajectories.transpose(1, 2), centered_trajectories)  # (N, 3, 3)
#
#     # 添加正则化
#     cov_matrices = cov_matrices + 1e-9 * torch.eye(3, device=device).unsqueeze(0)  # (N, 3, 3)
#
#     # 批量SVD分解
#     U, S, V = torch.linalg.svd(cov_matrices)
#
#     # 获取最小特征值对应的特征向量
#     min_eigval_idx = torch.argmin(S, dim=1)  # (N,)
#
#     # 提取每个点的旋转轴方向
#     axes = torch.stack([V[i, idx] for i, idx in enumerate(min_eigval_idx)], dim=0)  # (N, 3)
#
#     # 归一化
#     axes = axes / (torch.norm(axes, dim=1, keepdim=True) + 1e-6)
#
#     # 确保轴方向一致性
#     reference_axis = axes[0]
#     dot_products = torch.sum(axes * reference_axis.unsqueeze(0), dim=1)
#     axes = torch.where(dot_products.unsqueeze(1) < 0, -axes, axes)
#
#     # 使用运动显著性对轴方向进行加权平均
#     weighted_axis = torch.sum(axes * w_i.unsqueeze(1), dim=0)  # (3)
#     weighted_axis = weighted_axis / torch.norm(weighted_axis)  # 归一化
#
#     # 确保与先前估计的轴方向一致
#     if revolute_axis_reference is not None:
#         dot_val = torch.dot(weighted_axis, revolute_axis_reference)
#         if dot_val < 0:
#             weighted_axis = -weighted_axis
#     else:
#         revolute_axis_reference = weighted_axis.clone()
#
#     # 2. 高效计算旋转中心 - 关键修复部分
#     centers = []
#
#     # 专用Z轴旋转函数，与RevoluteModel中的rotateZ功能相同
#     def rotateZ(vec, rad):
#         return torch.tensor([
#             vec[0] * math.cos(rad) - vec[1] * math.sin(rad),
#             vec[0] * math.sin(rad) + vec[1] * math.cos(rad),
#             vec[2]
#         ], dtype=torch.float32, device=device)
#
#     for i in range(N):
#         for t in range(T - 2):
#             # 获取三个连续帧的位置
#             c1 = all_points_history[t, i]  # (3)
#             c2 = all_points_history[t + 1, i]  # (3)
#             c3 = all_points_history[t + 2, i]  # (3)
#
#             # 计算位移向量
#             v_12 = c2 - c1  # (3)
#             v_13 = c3 - c1  # (3)
#
#             # 计算法向量
#             n_ = torch.linalg.cross(v_12, v_13)  # (3)
#             nn = torch.norm(n_)
#
#             # 跳过数值不稳定的情况
#             if nn < 1e-6:
#                 continue
#
#             n_ = n_ / nn
#
#             # 构建平面坐标系 - 这是关键修复部分
#             plane_x = v_12 / (torch.norm(v_12) + 1e-6)
#             plane_y = v_13 - torch.dot(v_13, plane_x) * plane_x
#             ny = torch.norm(plane_y)
#
#             if ny < 1e-6:
#                 continue
#
#             plane_y = plane_y / ny
#             plane_z = torch.linalg.cross(plane_x, plane_y)
#
#             # 构建坐标系变换矩阵
#             plane_rot = torch.eye(4, dtype=torch.float32, device=device)
#             plane_rot[:3, 0] = plane_x
#             plane_rot[:3, 1] = plane_y
#             plane_rot[:3, 2] = plane_z
#             plane_rot[:3, 3] = c1
#
#             # 坐标系逆变换
#             plane_inv = torch.eye(4, dtype=torch.float32, device=device)
#             plane_inv[:3, :3] = plane_rot[:3, :3].transpose(0, 1)
#             plane_inv[:3, 3] = -torch.matmul(plane_rot[:3, :3].transpose(0, 1), c1)
#
#             # 将点变换到局部坐标系
#             p1_local = torch.cat([c1, torch.tensor([1.0], device=device)])
#             p2_local = torch.cat([c2, torch.tensor([1.0], device=device)])
#             p3_local = torch.cat([c3, torch.tensor([1.0], device=device)])
#
#             p1_local = torch.matmul(plane_inv, p1_local)
#             p2_local = torch.matmul(plane_inv, p2_local)
#             p3_local = torch.matmul(plane_inv, p3_local)
#
#             # 计算中点
#             c1_ = 0.5 * (p1_local[:3] + p2_local[:3])
#             c2_ = 0.5 * (p1_local[:3] + p3_local[:3])
#
#             # 使用rotateZ而不是任意轴旋转
#             p21 = rotateZ(p2_local[:3] - p1_local[:3], math.pi / 2.0)
#             p43 = rotateZ(p3_local[:3] - p1_local[:3], math.pi / 2.0)
#
#             # 设置方程组 A * x = b - 只使用XY坐标
#             A = torch.stack([p21[:2], -p43[:2]], dim=1)
#             b = (c2_[:2] - c1_[:2])
#
#             # 求解中垂线交点
#             try:
#                 if torch.abs(torch.det(A)) < 1e-6:
#                     continue
#
#                 lam = torch.linalg.solve(A, b)
#                 onplane_center = c1_[:2] + lam[0] * p21[:2]
#
#                 # 重建三维点 - 平面坐标系下的中心点
#                 plane_center_4 = torch.tensor([
#                     onplane_center[0].item(),
#                     onplane_center[1].item(),
#                     c1_[2].item(),
#                     1.0
#                 ], dtype=torch.float32, device=device)
#
#                 # 转换回世界坐标系 - 这是修复的关键
#                 center_world_4 = torch.matmul(plane_rot, plane_center_4)
#                 center = center_world_4[:3]
#
#                 centers.append(center)
#             except:
#                 continue
#
#     # 如果有计算结果，直接平均
#     if len(centers) > 0:
#         centers = torch.stack(centers, dim=0)  # (K, 3)
#         origin_sum = torch.mean(centers, dim=0)  # 使用平均值
#     else:
#         # 如果没有有效计算结果使用第一帧平均位置
#         origin_sum = torch.mean(all_points_history[0], dim=0)
#
#     # 3. 计算运动限制
#     i0 = 0
#     base_pt = all_points_history[0, i0]
#     base_vec = base_pt - origin_sum
#     nb = torch.norm(base_vec) + 1e-6
#
#     pts = all_points_history[:, i0, :]
#     vecs = pts - origin_sum
#
#     dotv = torch.sum(vecs * base_vec.unsqueeze(0), dim=1)
#     norm_v = torch.norm(vecs, dim=1) + 1e-6
#
#     cosval = torch.clamp(dotv / (nb * norm_v), -1., 1.)
#     angles = torch.acos(cosval)
#     min_a = float(angles.min().item())
#     max_a = float(angles.max().item())
#
#     return {
#         "axis": weighted_axis.cpu().numpy(),
#         "origin": origin_sum.cpu().numpy(),
#         "motion_limit": (min_a, max_a)
#     }

def compute_joint_info_all_types(
        all_points_history,
        neighbor_k=400,
        col_sigma=0.2, col_order=4.0,
        cop_sigma=0.2, cop_order=4.0,
        rad_sigma=0.2, rad_order=4.0,
        zp_sigma=0.2, zp_order=4.0,
        prob_sigma=0.2, prob_order=4.0,
        prismatic_sigma=0.08, prismatic_order=5.0,
        planar_sigma=0.12, planar_order=4.0,
        revolute_sigma=0.08, revolute_order=5.0,
        screw_sigma=0.15, screw_order=4.0,
        ball_sigma=0.12, ball_order=4.0,
        use_savgol=True,
        savgol_window=11,
        savgol_poly=2,
        use_multi_frame=False,
        multi_frame_window_radius=10
):
    """
    Main entry for joint type estimation:
      1) Compute velocity & angular velocity
      2) Compute the four fundamental scores (col/cop/rad/zp)
      3) Compute joint type probability
      4) Estimate geometric parameters for each joint type

    Args:
        all_points_history (ndarray): Point history of shape (T,N,3)
        neighbor_k (int): Number of neighbors for local estimation
        col_sigma (float): Width parameter for collinearity score
        col_order (float): Order parameter for collinearity score
        cop_sigma (float): Width parameter for coplanarity score
        cop_order (float): Order parameter for coplanarity score
        rad_sigma (float): Width parameter for radius consistency score
        rad_order (float): Order parameter for radius consistency score
        zp_sigma (float): Width parameter for zero pitch score
        zp_order (float): Order parameter for zero pitch score
        prob_sigma (float): Default width parameter for probability function
        prob_order (float): Default order parameter for probability function
        prismatic_sigma (float): Width parameter for prismatic joint
        prismatic_order (float): Order parameter for prismatic joint
        planar_sigma (float): Width parameter for planar joint
        planar_order (float): Order parameter for planar joint
        revolute_sigma (float): Width parameter for revolute joint
        revolute_order (float): Order parameter for revolute joint
        screw_sigma (float): Width parameter for screw joint
        screw_order (float): Order parameter for screw joint
        ball_sigma (float): Width parameter for ball joint
        ball_order (float): Order parameter for ball joint
        use_savgol (bool): Whether to apply Savitzky-Golay filtering
        savgol_window (int): Window size for Savitzky-Golay filter
        savgol_poly (int): Polynomial order for Savitzky-Golay filter
        use_multi_frame (bool): Whether to use multi-frame rigid fitting
        multi_frame_window_radius (int): Radius for multi-frame fitting

    Returns:
        tuple: (joint_params_dict, best_joint, info_dict)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = all_points_history.shape[0]
    N = all_points_history.shape[1]

    # Handle insufficient frames case
    if T < 2:
        ret = {
            "planar": {"normal": np.array([0., 0., 0.]), "motion_limit": (0., 0.)},
            "ball": {"center": np.array([0., 0., 0.]), "motion_limit": (0., 0., 0.)},
            "screw": {"axis": np.array([0., 0., 0.]), "origin": np.array([0., 0., 0.]), "pitch": 0.,
                      "motion_limit": (0., 0.)},
            "prismatic": {"axis": np.array([0., 0., 0.]), "motion_limit": (0., 0.)},
            "revolute": {"axis": np.array([0., 0., 0.]), "origin": np.array([0., 0., 0.]), "motion_limit": (0., 0.)}
        }
        return ret, "Unknown", None

    # Compute velocities
    dt = 0.1
    v_arr, w_arr = calculate_velocity_and_angular_velocity_for_all_frames(
        all_points_history,
        dt=dt,
        num_neighbors=neighbor_k,
        use_savgol=use_savgol,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly
    )

    if v_arr is None or w_arr is None:
        ret = {
            "planar": {"normal": np.array([0., 0., 0.]), "motion_limit": (0., 0.)},
            "ball": {"center": np.array([0., 0., 0.]), "motion_limit": (0., 0., 0.)},
            "screw": {"axis": np.array([0., 0., 0.]), "origin": np.array([0., 0., 0.]), "pitch": 0.,
                      "motion_limit": (0., 0.)},
            "prismatic": {"axis": np.array([0., 0., 0.]), "motion_limit": (0., 0.)},
            "revolute": {"axis": np.array([0., 0., 0.]), "origin": np.array([0., 0., 0.]), "motion_limit": (0., 0.)}
        }
        return ret, "Unknown", None

    # Compute motion salience
    ms_torch = compute_motion_salience_batch(all_points_history, neighbor_k=neighbor_k, device=device)
    ms = ms_torch.cpu().numpy()

    # Normalize motion salience to get weights
    sum_ms = ms.sum()
    if sum_ms < 1e-6:
        w_i = np.ones_like(ms) / ms.shape[0]
    else:
        w_i = ms / sum_ms

    # Compute joint parameters for each type
    p_info = compute_planar_info(all_points_history, v_arr, w_arr, w_i, device=device)
    b_info = compute_ball_info(all_points_history, v_arr, w_arr, w_i, device=device)
    s_info = compute_screw_info(all_points_history, v_arr, w_arr, w_i, device=device)
    pm_info = compute_prismatic_info(all_points_history, v_arr, w_arr, w_i, device=device)
    r_info = compute_revolute_info(all_points_history, v_arr, w_arr, w_i, device=device)

    ret = {
        "planar": p_info,
        "ball": b_info,
        "screw": s_info,
        "prismatic": pm_info,
        "revolute": r_info
    }

    # Compute basic scores
    col, cop, rad, zp = compute_basic_scores(
        v_arr, w_arr, device=device,
        col_sigma=col_sigma, col_order=col_order,
        cop_sigma=cop_sigma, cop_order=cop_order,
        rad_sigma=rad_sigma, rad_order=rad_order,
        zp_sigma=zp_sigma, zp_order=zp_order
    )

    # Compute weighted averages
    w_t = torch.tensor(w_i, dtype=torch.float32, device=device)
    col_mean = float(torch.sum(w_t * col).item())
    cop_mean = float(torch.sum(w_t * cop).item())
    rad_mean = float(torch.sum(w_t * rad).item())
    zp_mean = float(torch.sum(w_t * zp).item())

    basic_score_avg = {
        "col_mean": col_mean,
        "cop_mean": cop_mean,
        "rad_mean": rad_mean,
        "zp_mean": zp_mean
    }

    # Compute joint type probabilities with joint-specific parameters
    prismatic_pt = compute_joint_probability_new(
        col, cop, rad, zp, "prismatic",
        prob_sigma=prob_sigma, prob_order=prob_order,
        prismatic_sigma=prismatic_sigma, prismatic_order=prismatic_order,
        planar_sigma=planar_sigma, planar_order=planar_order,
        revolute_sigma=revolute_sigma, revolute_order=revolute_order,
        screw_sigma=screw_sigma, screw_order=screw_order,
        ball_sigma=ball_sigma, ball_order=ball_order
    )

    planar_pt = compute_joint_probability_new(
        col, cop, rad, zp, "planar",
        prob_sigma=prob_sigma, prob_order=prob_order,
        prismatic_sigma=prismatic_sigma, prismatic_order=prismatic_order,
        planar_sigma=planar_sigma, planar_order=planar_order,
        revolute_sigma=revolute_sigma, revolute_order=revolute_order,
        screw_sigma=screw_sigma, screw_order=screw_order,
        ball_sigma=ball_sigma, ball_order=ball_order
    )

    revolute_pt = compute_joint_probability_new(
        col, cop, rad, zp, "revolute",
        prob_sigma=prob_sigma, prob_order=prob_order,
        prismatic_sigma=prismatic_sigma, prismatic_order=prismatic_order,
        planar_sigma=planar_sigma, planar_order=planar_order,
        revolute_sigma=revolute_sigma, revolute_order=revolute_order,
        screw_sigma=screw_sigma, screw_order=screw_order,
        ball_sigma=ball_sigma, ball_order=ball_order
    )

    screw_pt = compute_joint_probability_new(
        col, cop, rad, zp, "screw",
        prob_sigma=prob_sigma, prob_order=prob_order,
        prismatic_sigma=prismatic_sigma, prismatic_order=prismatic_order,
        planar_sigma=planar_sigma, planar_order=planar_order,
        revolute_sigma=revolute_sigma, revolute_order=revolute_order,
        screw_sigma=screw_sigma, screw_order=screw_order,
        ball_sigma=ball_sigma, ball_order=ball_order
    )

    ball_pt = compute_joint_probability_new(
        col, cop, rad, zp, "ball",
        prob_sigma=prob_sigma, prob_order=prob_order,
        prismatic_sigma=prismatic_sigma, prismatic_order=prismatic_order,
        planar_sigma=planar_sigma, planar_order=planar_order,
        revolute_sigma=revolute_sigma, revolute_order=revolute_order,
        screw_sigma=screw_sigma, screw_order=screw_order,
        ball_sigma=ball_sigma, ball_order=ball_order
    )

    # Compute weighted probabilities
    prismatic_prob = float(torch.sum(w_t * prismatic_pt).item())
    planar_prob = float(torch.sum(w_t * planar_pt).item())
    revolute_prob = float(torch.sum(w_t * revolute_pt).item())
    screw_prob = float(torch.sum(w_t * screw_pt).item())
    ball_prob = float(torch.sum(w_t * ball_pt).item())

    # Find best joint type
    joint_probs = [
        ("prismatic", prismatic_prob),
        ("planar", planar_prob),
        ("revolute", revolute_prob),
        ("screw", screw_prob),
        ("ball", ball_prob),
    ]
    joint_probs.sort(key=lambda x: x[1], reverse=True)
    best_joint, best_pval = joint_probs[0]

    # Threshold for confidence
    if best_pval < 0.3:
        best_joint = "Unknown"

    ret_probs = {
        "prismatic": prismatic_prob,
        "planar": planar_prob,
        "revolute": revolute_prob,
        "screw": screw_prob,
        "ball": ball_prob
    }

    # Collect additional information
    info_dict = {
        "basic_score_avg": basic_score_avg,
        "joint_probs": ret_probs,
        "v_arr": v_arr,
        "w_arr": w_arr,
        "w_i": w_i
    }

    return ret, best_joint, info_dict
