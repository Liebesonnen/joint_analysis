"""
Joint type estimation algorithms and parameter calculation with PyTorch acceleration.
"""

import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import random
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


def multi_frame_rigid_fit(all_points_history: torch.Tensor, center_idx: int, window_radius: int):
    """
    Multi-frame rigid fit around [center_idx-window_radius, center_idx+window_radius].
    Returns a 4x4 transform from the center frame to best fit.

    Args:
        all_points_history (Tensor): Point history of shape (T,N,3)
        center_idx (int): Index of the center frame
        window_radius (int): Radius of the window around center frame

    Returns:
        Tensor: Transformation matrix of shape (4,4)
    """
    T, N, _ = all_points_history.shape
    device = all_points_history.device

    # Define window boundaries
    i_min = max(0, center_idx - window_radius)
    i_max = min(T - 1, center_idx + window_radius)

    # Reference points from center frame
    ref_pts = all_points_history[center_idx]

    # Create frame indices for window (excluding center_idx)
    frame_indices = torch.cat([
        torch.arange(i_min, center_idx, device=device),
        torch.arange(center_idx + 1, i_max + 1, device=device)
    ])

    # Handle empty window case
    if len(frame_indices) == 0:
        return torch.eye(4, device=device)

    # Collect source and target points using indexing
    # Expand ref_pts to match frame_indices size [num_frames, N, 3]
    src_big = ref_pts.unsqueeze(0).expand(len(frame_indices), -1, -1).reshape(-1, 3)

    # Gather target points using frame_indices [num_frames, N, 3]
    tgt_big = all_points_history[frame_indices].reshape(-1, 3)

    # Center the point clouds
    src_mean = src_big.mean(dim=0)
    tgt_mean = tgt_big.mean(dim=0)
    src_centered = src_big - src_mean
    tgt_centered = tgt_big - tgt_mean

    # Compute cross-covariance matrix
    H = torch.einsum('ni,nj->ij', src_centered, tgt_centered)

    # SVD decomposition
    U, S, Vt = torch.linalg.svd(H)

    # Compute rotation matrix
    R_ = Vt.T @ U.T

    # Handle reflection case
    if torch.det(R_) < 0:
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T

    # Compute translation
    t_ = tgt_mean - R_ @ src_mean

    # Create full transformation matrix
    Tmat = torch.eye(4, device=device)
    Tmat[:3, :3] = R_
    Tmat[:3, 3] = t_

    return Tmat


def calculate_velocity_and_angular_velocity_for_all_frames(
        all_points_history,
        dt=0.1,
        num_neighbors=400,
        use_savgol=True,
        savgol_window=7,
        savgol_poly=3,
        use_multi_frame=True,
        window_radius=5
):
    """
    Compute linear and angular velocity for all frames.

    Args:
        all_points_history (ndarray): Point history of shape (T,N,3)
        dt (float): Time step
        num_neighbors (int): Number of neighbors for local estimation
        use_savgol (bool): Whether to apply Savitzky-Golay filtering
        savgol_window (int): Window size for Savitzky-Golay filter
        savgol_poly (int): Polynomial order for Savitzky-Golay filter
        use_multi_frame (bool): Whether to use multi-frame rigid fitting
        window_radius (int): Radius for multi-frame fitting

    Returns:
        tuple: (v_arr, w_arr) linear and angular velocities of shape (T-1,N,3)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert input to tensor if needed
    if not isinstance(all_points_history, torch.Tensor):
        all_points_history = torch.tensor(all_points_history, dtype=torch.float32, device=device)

    T, N, _ = all_points_history.shape
    if T < 2:
        return None, None

    if use_multi_frame:
        # Multi-frame rigid fit approach - compute for all frames in parallel
        center_indices = torch.arange(1, T, device=device)
        B = len(center_indices)

        v_batch = torch.zeros((B, N, 3), device=device)
        w_batch = torch.zeros((B, N, 3), device=device)

        # This part still needs a loop due to multi_frame_rigid_fit function
        for i, center_idx in enumerate(center_indices):
            Tmat = multi_frame_rigid_fit(all_points_history, center_idx, window_radius)
            Tmat_batch = Tmat.unsqueeze(0)

            # Convert to twist coordinates
            se3_logs = se3_log_map_batch(Tmat_batch)
            se3_v = se3_logs[0, :3] / dt
            se3_w = se3_logs[0, 3:] / dt

            # Replicate for all points
            v_batch[i] = se3_v.unsqueeze(0).expand(N, -1)
            w_batch[i] = se3_w.unsqueeze(0).expand(N, -1)

        v_arr = v_batch.cpu().numpy()
        w_arr = w_batch.cpu().numpy()
    else:
        # Neighbor-based estimation approach (already batch-oriented)
        pts_prev = all_points_history[:-1]  # (T-1, N, 3)
        pts_curr = all_points_history[1:]  # (T-1, N, 3)
        B = T - 1

        # Find neighbors
        neighbor_idx_prev = find_neighbors_batch(pts_prev, num_neighbors)  # (B, N, K)
        neighbor_idx_curr = find_neighbors_batch(pts_curr, num_neighbors)  # (B, N, K)
        K = num_neighbors

        # Create batch indices for gathering
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, K)

        # Get neighbor points using advanced indexing
        src_batch = pts_prev[batch_indices, neighbor_idx_prev]  # (B, N, K, 3)
        tar_batch = pts_curr[batch_indices, neighbor_idx_curr]  # (B, N, K, 3)

        # Reshape for batch processing
        src_2d = src_batch.reshape(B * N, K, 3)  # (B*N, K, 3)
        tar_2d = tar_batch.reshape(B * N, K, 3)  # (B*N, K, 3)

        # Estimate rotation matrices
        R_2d = estimate_rotation_matrix_batch(src_2d, tar_2d)  # (B*N, 3, 3)

        # Compute centroids
        c1_2d = src_2d.mean(dim=1)  # (B*N, 3)
        c2_2d = tar_2d.mean(dim=1)  # (B*N, 3)
        delta_p_2d = c2_2d - c1_2d  # (B*N, 3)

        # Create SE(3) transformation matrices
        eye_4 = torch.eye(4, device=device).unsqueeze(0).expand(B * N, -1, -1).clone()
        eye_4[:, :3, :3] = R_2d
        eye_4[:, :3, 3] = delta_p_2d
        transform_matrices_2d = eye_4  # (B*N, 4, 4)

        # Convert to twist coordinates
        se3_logs_2d = se3_log_map_batch(transform_matrices_2d)  # (B*N, 6)

        # Extract linear and angular velocities
        v_2d = se3_logs_2d[:, :3] / dt  # (B*N, 3)
        w_2d = se3_logs_2d[:, 3:] / dt  # (B*N, 3)

        # Reshape back to (B, N, 3)
        v_arr = v_2d.reshape(B, N, 3).cpu().numpy()
        w_arr = w_2d.reshape(B, N, 3).cpu().numpy()

    # Apply Savitzky-Golay filter if requested
    if use_savgol and (T - 1) >= savgol_window:
        v_arr = savgol_filter(
            v_arr, window_length=savgol_window, polyorder=savgol_poly,
            axis=0, mode='mirror'
        )
        w_arr = savgol_filter(
            w_arr, window_length=savgol_window, polyorder=savgol_poly,
            axis=0, mode='mirror'
        )

    return v_arr, w_arr


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
    v_history = torch.as_tensor(v_history, dtype=torch.float32, device=device) if v_history is not None else None
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

def compute_revolute_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Estimate parameters for a revolute joint.

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

    all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    v_history = torch.as_tensor(v_history, dtype=torch.float32, device=device)
    omega_history = torch.as_tensor(omega_history, dtype=torch.float32, device=device)
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)

    T, N = all_points_history.shape[0], all_points_history.shape[1]
    if T < 2:
        return {
            "axis": np.array([0., 0., 0.]),
            "origin": np.array([0., 0., 0.]),
            "motion_limit": (0., 0.)
        }

    B = T - 1
    if B < 1:
        return {
            "axis": np.array([0., 0., 0.]),
            "origin": np.array([0., 0., 0.]),
            "motion_limit": (0., 0.)
        }

    # Reshape for point-wise processing
    omega_nbc = omega_history.permute(1, 0, 2).contiguous()

    # Compute covariance matrices of angular velocities
    covs = torch.einsum('ibm,ibn->imn', omega_nbc, omega_nbc)
    covs += 1e-9 * torch.eye(3, device=device)

    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(covs)

    # Extract principal directions
    max_vecs = eigvecs[:, :, 2]
    max_vecs = normalize_vector_torch(max_vecs)

    # Compute weighted axis
    revolve_axis = torch.sum(max_vecs * w_i.unsqueeze(-1), dim=0)
    revolve_axis = normalize_vector_torch(revolve_axis.unsqueeze(0))[0]

    # Ensure consistent direction with previous estimations
    if revolute_axis_reference is None:
        revolute_axis_reference = revolve_axis.clone()
    else:
        dot_val = torch.dot(revolve_axis, revolute_axis_reference)
        if dot_val < 0:
            revolve_axis = -revolve_axis

    # Compute radius and center
    eps_ = 1e-8
    v_norm = torch.norm(v_history, dim=2)
    w_norm = torch.norm(omega_history, dim=2)
    mask_w = (w_norm > eps_)

    r_mat = torch.zeros_like(v_norm)
    r_mat[mask_w] = v_norm[mask_w] / w_norm[mask_w]

    v_u = normalize_vector_torch(v_history)
    w_u = normalize_vector_torch(omega_history)

    dir_ = -torch.cross(v_u, w_u, dim=2)
    dir_ = normalize_vector_torch(dir_)

    r_3d = r_mat.unsqueeze(-1)
    c_pos = all_points_history[:-1] + dir_ * r_3d

    # Reshape for robustness
    c_pos_nbc = c_pos.permute(1, 0, 2).contiguous()

    # Use median for robust center estimation
    c_pos_median = c_pos_nbc.median(dim=1).values
    dev = torch.norm(c_pos_nbc - c_pos_median.unsqueeze(1), dim=2)
    scale = dev.median(dim=1).values + 1e-6

    # Compute weights based on deviation
    ratio = dev / scale.unsqueeze(-1)
    # w_r = 1.0 / (1.0 + ratio * ratio)
    w_r = 1.0 / (1.0 + torch.pow(ratio, 3.0))  # 使用更高次幂增强离群点抑制
    w_r_3d = w_r.unsqueeze(-1)

    # Compute weighted center
    c_pos_weighted = c_pos_nbc * w_r_3d
    sum_pos = c_pos_weighted.sum(dim=1)
    sum_w = w_r.sum(dim=1, keepdim=True) + 1e-6

    origin_each = sum_pos / sum_w
    origin_sum = torch.sum(origin_each * w_i.unsqueeze(-1), dim=0)

    # Compute motion limits
    i0 = 0
    base_pt = all_points_history[0, i0]
    base_vec = base_pt - origin_sum
    nb = torch.norm(base_vec) + 1e-6

    pts = all_points_history[:, i0, :]
    vecs = pts - origin_sum

    dotv = torch.sum(vecs * base_vec.unsqueeze(0), dim=1)
    norm_v = torch.norm(vecs, dim=1) + 1e-6

    cosval = torch.clamp(dotv / (nb * norm_v), -1., 1.)
    angles = torch.acos(cosval)
    min_a = float(angles.min().item())
    max_a = float(angles.max().item())

    return {
        "axis": revolve_axis.cpu().numpy(),
        "origin": origin_sum.cpu().numpy(),
        "motion_limit": (min_a, max_a)
    }


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
        savgol_window=3,
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
        savgol_poly=savgol_poly,
        use_multi_frame=use_multi_frame,
        window_radius=multi_frame_window_radius
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

# def compute_joint_info_all_types(
#         all_points_history,
#         neighbor_k=400,
#         col_sigma=0.3, col_order=3.0,
#         cop_sigma=0.3, cop_order=3.0,
#         rad_sigma=0.3, rad_order=3.0,
#         zp_sigma=0.3, zp_order=3.0,
#         prob_sigma=0.3, prob_order=3.0,
#         use_savgol=True,
#         savgol_window=7,
#         savgol_poly=3,
#         use_multi_frame=True,
#         multi_frame_window_radius=5,
#         confidence_threshold=0.1
# ):
#     """
#     Main entry for joint type estimation:
#       1) Compute velocity & angular velocity
#       2) Compute the four fundamental scores (col/cop/rad/zp)
#       3) Compute joint type probability
#       4) Estimate geometric parameters for each joint type
#
#     Args:
#         all_points_history (ndarray): Point history of shape (T,N,3)
#         neighbor_k (int): Number of neighbors for local estimation
#         col_sigma (float): Width parameter for collinearity score
#         col_order (float): Order parameter for collinearity score
#         cop_sigma (float): Width parameter for coplanarity score
#         cop_order (float): Order parameter for coplanarity score
#         rad_sigma (float): Width parameter for radius consistency score
#         rad_order (float): Order parameter for radius consistency score
#         zp_sigma (float): Width parameter for zero pitch score
#         zp_order (float): Order parameter for zero pitch score
#         prob_sigma (float): Width parameter for probability function
#         prob_order (float): Order parameter for probability function
#         use_savgol (bool): Whether to apply Savitzky-Golay filtering
#         savgol_window (int): Window size for Savitzky-Golay filter
#         savgol_poly (int): Polynomial order for Savitzky-Golay filter
#         use_multi_frame (bool): Whether to use multi-frame rigid fitting
#         multi_frame_window_radius (int): Radius for multi-frame fitting
#         confidence_threshold (float): Threshold for joint type confidence
#
#     Returns:
#         tuple: (joint_params_dict, best_joint, info_dict)
#     """
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     T = all_points_history.shape[0]
#     N = all_points_history.shape[1]
#
#     # Handle insufficient frames case
#     if T < 2:
#         ret = {
#             "planar": {"normal": np.array([0., 0., 0.]), "motion_limit": (0., 0.)},
#             "ball": {"center": np.array([0., 0., 0.]), "motion_limit": (0., 0., 0.)},
#             "screw": {"axis": np.array([0., 0., 0.]), "origin": np.array([0., 0., 0.]), "pitch": 0.,
#                       "motion_limit": (0., 0.)},
#             "prismatic": {"axis": np.array([0., 0., 0.]), "motion_limit": (0., 0.)},
#             "revolute": {"axis": np.array([0., 0., 0.]), "origin": np.array([0., 0., 0.]), "motion_limit": (0., 0.)}
#         }
#         return ret, "Unknown", None
#
#     # Ensure contiguous memory layout for point cloud data
#     all_points_history_contiguous = np.ascontiguousarray(all_points_history)
#
#     # Compute velocities
#     dt = 0.1
#     v_arr, w_arr = calculate_velocity_and_angular_velocity_for_all_frames(
#         all_points_history_contiguous,
#         dt=dt,
#         num_neighbors=neighbor_k,
#         use_savgol=use_savgol,
#         savgol_window=savgol_window,
#         savgol_poly=savgol_poly,
#         use_multi_frame=use_multi_frame,
#         window_radius=multi_frame_window_radius
#     )
#
#     if v_arr is None or w_arr is None:
#         ret = {
#             "planar": {"normal": np.array([0., 0., 0.]), "motion_limit": (0., 0.)},
#             "ball": {"center": np.array([0., 0., 0.]), "motion_limit": (0., 0., 0.)},
#             "screw": {"axis": np.array([0., 0., 0.]), "origin": np.array([0., 0., 0.]), "pitch": 0.,
#                       "motion_limit": (0., 0.)},
#             "prismatic": {"axis": np.array([0., 0., 0.]), "motion_limit": (0., 0.)},
#             "revolute": {"axis": np.array([0., 0., 0.]), "origin": np.array([0., 0., 0.]), "motion_limit": (0., 0.)}
#         }
#         return ret, "Unknown", None
#
#     # Vectorized calculation of motion significance for weights
#     v_tensor = torch.tensor(v_arr, device=device)
#     w_tensor = torch.tensor(w_arr, device=device)
#
#     # Calculate absolute values
#     v_abs = torch.abs(v_tensor)
#     w_abs = torch.abs(w_tensor)
#
#     # Calculate total motion magnitude
#     motion_magnitude = torch.mean(v_abs, dim=0) + torch.mean(w_abs, dim=0)  # (N, 3)
#     motion_magnitude = torch.mean(motion_magnitude, dim=1)  # (N)
#
#     # Normalize to weights
#     sum_motion = torch.sum(motion_magnitude)
#     if sum_motion < 1e-6:
#         w_i = torch.ones_like(motion_magnitude) / motion_magnitude.shape[0]
#     else:
#         w_i = motion_magnitude / sum_motion
#
#     # Convert to numpy for compatibility with other functions
#     w_i_np = w_i.cpu().numpy()
#
#     # Compute basic scores - using improved score calculation
#     col, cop, rad, zp = compute_basic_scores(
#         v_arr, w_arr, device=device,
#         col_sigma=col_sigma, col_order=col_order,
#         cop_sigma=cop_sigma, cop_order=cop_order,
#         rad_sigma=rad_sigma, rad_order=rad_order,
#         zp_sigma=zp_sigma, zp_order=zp_order
#     )
#
#     # Calculate weighted averages
#     w_t = w_i
#     col_mean = float(torch.sum(w_t * col).item())
#     cop_mean = float(torch.sum(w_t * cop).item())
#     rad_mean = float(torch.sum(w_t * rad).item())
#     zp_mean = float(torch.sum(w_t * zp).item())
#
#     basic_score_avg = {
#         "col_mean": col_mean,
#         "cop_mean": cop_mean,
#         "rad_mean": rad_mean,
#         "zp_mean": zp_mean
#     }
#
#     # Compute joint type probabilities
#     # Vectorized computation of all joint probabilities
#     joint_types = ["prismatic", "planar", "revolute", "screw", "ball"]
#     joint_probs = {}
#
#     for joint_type in joint_types:
#         prob = compute_joint_probability_new(col, cop, rad, zp, joint_type,
#                                              prob_sigma=prob_sigma, prob_order=prob_order)
#         joint_probs[joint_type] = float(torch.sum(w_t * prob).item())
#
#     # Find best joint type
#     best_joint = max(joint_probs, key=joint_probs.get)
#     best_pval = joint_probs[best_joint]
#
#     # Apply confidence threshold
#     if best_pval < confidence_threshold:
#         best_joint = "Unknown"
#
#     # Compute parameters for all joint types
#     p_info = compute_planar_info(all_points_history, v_arr, w_arr, w_i_np, device=device)
#     b_info = compute_ball_info(all_points_history, v_arr, w_arr, w_i_np, device=device)
#     s_info = compute_screw_info(all_points_history, v_arr, w_arr, w_i_np, device=device)
#     pm_info = compute_prismatic_info(all_points_history, v_arr, w_arr, w_i_np, device=device)
#     r_info = compute_revolute_info(all_points_history, v_arr, w_arr, w_i_np, device=device)
#
#     # Collect all parameters in return dictionary
#     ret = {
#         "planar": p_info,
#         "ball": b_info,
#         "screw": s_info,
#         "prismatic": pm_info,
#         "revolute": r_info
#     }
#
#     # Collect additional information
#     info_dict = {
#         "basic_score_avg": basic_score_avg,
#         "joint_probs": joint_probs,
#         "v_arr": v_arr,
#         "w_arr": w_arr,
#         "w_i": w_i_np
#     }
#
#     return ret, best_joint, info_dict