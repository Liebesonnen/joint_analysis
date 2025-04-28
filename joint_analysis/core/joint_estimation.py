"""
Joint type estimation algorithms and parameter calculation.
"""

import numpy as np
import torch
from scipy.signal import savgol_filter

from .scoring import (
    normalize_vector_torch, find_neighbors_batch, compute_basic_scores,
    compute_joint_probability_new, compute_motion_salience_batch, se3_log_map_batch
)
from .scoring import estimate_rotation_matrix_batch
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

    # Collect source and target point sets from window
    src_list = []
    tgt_list = []
    for idx in range(i_min, i_max + 1):
        if idx == center_idx:
            continue
        cur_pts = all_points_history[idx]
        src_list.append(ref_pts)
        tgt_list.append(cur_pts)

    # Handle empty window case
    if not src_list:
        return torch.eye(4, device=device)

    # Concatenate all points
    src_big = torch.cat(src_list, dim=0)
    tgt_big = torch.cat(tgt_list, dim=0)

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
        savgol_window=5,
        savgol_poly=2,
        use_multi_frame=False,
        window_radius=2
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
        # Multi-frame rigid fit approach
        v_list = []
        w_list = []

        for t in range(T - 1):
            center_idx = t + 1
            Tmat = multi_frame_rigid_fit(all_points_history, center_idx, window_radius)
            Tmat_batch = Tmat.unsqueeze(0)

            # Convert to twist coordinates
            se3_logs = se3_log_map_batch(Tmat_batch)
            se3_v = se3_logs[0, :3] / dt
            se3_w = se3_logs[0, 3:] / dt

            # Replicate for all points
            v_list.append(se3_v.unsqueeze(0).repeat(all_points_history.shape[1], 1))
            w_list.append(se3_w.unsqueeze(0).repeat(all_points_history.shape[1], 1))

        v_arr = torch.stack(v_list, dim=0).cpu().numpy()
        w_arr = torch.stack(w_list, dim=0).cpu().numpy()

    else:
        # Neighbor-based estimation approach
        pts_prev = all_points_history[:-1]  # (T-1, N, 3)
        pts_curr = all_points_history[1:]  # (T-1, N, 3)
        B = T - 1

        # Find neighbors
        neighbor_idx_prev = find_neighbors_batch(pts_prev, num_neighbors)  # (B, N, K)
        neighbor_idx_curr = find_neighbors_batch(pts_curr, num_neighbors)  # (B, N, K)
        K = num_neighbors

        # Get neighbor points
        src_batch = pts_prev[
                    torch.arange(B, device=device).view(B, 1, 1),
                    neighbor_idx_prev,
                    :
                    ]  # (B, N, K, 3)

        tar_batch = pts_curr[
                    torch.arange(B, device=device).view(B, 1, 1),
                    neighbor_idx_curr,
                    :
                    ]  # (B, N, K, 3)

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
    global planar_normal_reference, planar_axis1_reference, planar_axis2_reference
    global plane_is_fixed

    all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)
    T, N = all_points_history.shape[0], all_points_history.shape[1]

    if T < 3:
        return {"normal": np.array([0., 0., 0.]), "motion_limit": (0., 0.)}

    if not plane_is_fixed:
        # First estimation: weighted PCA
        pts_torch = all_points_history.reshape(-1, 3)
        weights_expanded = w_i.repeat(T)
        w_sum = weights_expanded.sum()

        # Compute weighted mean
        sum_wx = torch.einsum('b,bj->j', weights_expanded, pts_torch)
        weighted_mean = sum_wx / (w_sum + 1e-9)

        # Center points
        centered = pts_torch - weighted_mean
        wc_ = centered * weights_expanded.unsqueeze(-1)

        # Compute weighted covariance matrix
        cov_mat = torch.einsum('bi,bj->ij', wc_, centered) / (w_sum + 1e-9)
        cov_mat += 1e-9 * torch.eye(3, device=device)

        # Eigen decomposition
        eigvals, eigvecs = torch.linalg.eigh(cov_mat)

        # Extract principal axes
        axis1_ = eigvecs[:, 2]
        axis2_ = eigvecs[:, 1]
        axis1_ = normalize_vector_torch(axis1_.unsqueeze(0))[0]
        axis2_ = normalize_vector_torch(axis2_.unsqueeze(0))[0]

        # Compute normal from principal axes
        normal_ = torch.cross(axis1_, axis2_, dim=0)
        normal_ = normalize_vector_torch(normal_.unsqueeze(0))[0]

        # Save reference directions
        planar_axis1_reference = axis1_.clone()
        planar_axis2_reference = axis2_.clone()
        planar_normal_reference = normal_.clone()
        plane_is_fixed = True
    else:
        axis1_ = planar_axis1_reference.clone()
        axis2_ = planar_axis2_reference.clone()
        normal_ = planar_normal_reference.clone()

    # Compute motion limits
    if T < 2:
        motion_limit = (0., 0.)
    else:
        pt0_0 = all_points_history[0, 0]
        pts = all_points_history[:, 0] - pt0_0

        # Project onto principal axes
        proj1 = torch.matmul(pts, axis1_)
        proj2 = torch.matmul(pts, axis2_)

        # Get motion limits
        max_p1 = proj1.max().item()
        max_p2 = proj2.max().item()
        motion_limit = (float(max_p1), float(max_p2))

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
    dir_ = - torch.cross(v_u, w_u, dim=2)
    dir_ = normalize_vector_torch(dir_)

    # Estimate center position as point + r * direction
    r_3d = r_mat.unsqueeze(-1)
    c_pos = all_points_history[:-1] + dir_ * r_3d

    # Average centers
    center_each = torch.mean(c_pos, dim=0)
    center_sum = torch.sum(w_i[:, None] * center_each, dim=0)

    # Compute approximate motion limits (maximum angle)
    base_pt_0 = all_points_history[0, 0]
    base_vec = base_pt_0 - center_sum
    norm_b = torch.norm(base_vec) + 1e-6

    pts = all_points_history[:, 0, :]
    vecs = pts - center_sum

    dotv = torch.sum(vecs * base_vec.unsqueeze(0), dim=1)
    norm_v = torch.norm(vecs, dim=1) + 1e-6

    cosval = torch.clamp(dotv / (norm_b * norm_v), -1., 1.)
    angles = torch.acos(cosval)
    max_angle = angles.max().item()

    return {
        "center": center_sum.cpu().numpy(),
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

    # Normalize velocity vectors
    v_u = normalize_vector_torch(v_history)
    w_u = normalize_vector_torch(omega_history)

    # Compute pitch
    v_norm = torch.norm(v_history, dim=2)
    w_norm = torch.norm(omega_history, dim=2)

    # s_mat = dot product of v and unit w = |v|*cos(angle)
    s_mat = torch.sum(v_history * w_u, dim=2)

    # Initialize ratio array
    ratio = torch.zeros_like(s_mat)
    mask_w_ = (w_norm > EPS_W)

    # Compute ratio of s_mat to v_norm where w is significant
    ratio[mask_w_] = s_mat[mask_w_] / (v_norm[mask_w_] + 1e-9)

    # Compute pitch angle and sin(pitch)
    pitch_all = torch.acos(torch.clamp(ratio, -1., 1.))
    pitch_sin = torch.sin(pitch_all)

    # Compute radius
    r_ = torch.zeros_like(v_norm)
    r_[mask_w_] = (v_norm[mask_w_] * pitch_sin[mask_w_]) / (w_norm[mask_w_] + 1e-9)

    # Compute direction
    dir_ = -torch.cross(v_u, w_u, dim=-1)
    dir_ = normalize_vector_torch(dir_)

    # Estimate position
    c_pos = all_points_history[:-1] + dir_ * r_.unsqueeze(-1)

    # Compute average pitch
    pitch_each = torch.mean(pitch_all, dim=0)
    pitch_sum = float(torch.sum(w_i * pitch_each).item())

    # Estimate axis direction from angular velocities
    w_u_flat = w_u.reshape(-1, 3)
    T_actual = w_u.shape[0]
    w_i_flat = w_i.unsqueeze(0).expand(T_actual, N).reshape(-1)
    W = torch.sum(w_i_flat) + 1e-9

    # Compute weighted mean of angular velocities
    weighted_mean = (w_u_flat * w_i_flat.unsqueeze(-1)).sum(dim=0) / W

    # Center angular velocities
    w_u_centered = w_u_flat - weighted_mean
    w_u_centered_weighted = w_u_centered * torch.sqrt(w_i_flat.unsqueeze(-1))

    # Compute covariance matrix
    cov_mat = (w_u_centered_weighted.transpose(0, 1) @ w_u_centered_weighted) / W
    cov_mat += 1e-9 * torch.eye(3, device=device)

    # Get principal direction via eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov_mat)
    idx_max = torch.argmax(eigvals)
    axis_raw = eigvecs[:, idx_max]
    axis_sum = normalize_vector_torch(axis_raw.unsqueeze(0))[0]

    # Ensure consistent direction with previous estimations
    if screw_axis_reference is not None:
        dotval = torch.dot(axis_sum, screw_axis_reference)
        if dotval < 0:
            axis_sum = -axis_sum
    screw_axis_reference = axis_sum.clone()

    # Handle origin estimation
    c_pos_flat = c_pos.reshape(-1, 3)
    if c_pos_flat.shape[0] == 0:
        return {
            "axis": axis_sum.cpu().numpy(),
            "origin": np.array([0., 0., 0.]),
            "pitch": pitch_sum,
            "motion_limit": (0., 0.)
        }

    # Find a robust origin using median filtering
    median_cp = c_pos_flat.median(dim=0).values
    dev = torch.norm(c_pos_flat - median_cp, dim=1)
    med_dev = dev.median()

    if med_dev < 1e-9:
        origin_pts = c_pos_flat
    else:
        # Filter outliers
        mask_in = (dev < 3.0 * med_dev)
        origin_pts = c_pos_flat[mask_in]

    if origin_pts.shape[0] == 0:
        origin_sum = median_cp
    else:
        origin_sum = origin_pts.mean(dim=0)

    # Compute motion limits
    i0 = 0
    base_pt = all_points_history[0, i0]
    base_vec = base_pt - origin_sum
    nb = torch.norm(base_vec) + 1e-9

    pts_0 = all_points_history[:, i0, :]
    vecs = pts_0 - origin_sum

    dotv = torch.sum(vecs * base_vec.unsqueeze(0), dim=1)
    norm_v = torch.norm(vecs, dim=1) + 1e-9

    cosval = torch.clamp(dotv / (nb * norm_v), -1., 1.)
    angles = torch.acos(cosval)
    motion_limit = (float(angles.min().item()), float(angles.max().item()))

    return {
        "axis": axis_sum.cpu().numpy(),
        "origin": origin_sum.cpu().numpy(),
        "pitch": pitch_sum,
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
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)

    T, N = all_points_history.shape[0], all_points_history.shape[1]
    if T < 2:
        return {
            "axis": np.array([0., 0., 0.]),
            "origin": np.array([0., 0., 0.]),
            "motion_limit": (0., 0.)
        }

    # Reshape for point-wise processing
    pos_history_n_tc = all_points_history.permute(1, 0, 2).contiguous()

    # Center points
    mean_pos = pos_history_n_tc.mean(dim=1, keepdim=True)
    centered = pos_history_n_tc - mean_pos

    # Compute covariance matrices for each point
    covs = torch.einsum('ntm,ntk->nmk', centered, centered)
    B_n = covs.shape[0]

    # Add small regularization
    epsilon_eye = 1e-9 * torch.eye(3, device=device)
    for i in range(B_n):
        covs[i] += epsilon_eye

    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(covs)

    # Extract principal directions
    max_vecs = eigvecs[:, :, 2]
    max_vecs = normalize_vector_torch(max_vecs)

    # Compute weighted direction
    weighted_dir = torch.sum(max_vecs * w_i.unsqueeze(-1), dim=0)
    axis_sum = normalize_vector_torch(weighted_dir.unsqueeze(0))[0]

    # Ensure consistent direction with previous estimations
    if prismatic_axis_reference is None:
        prismatic_axis_reference = axis_sum.clone()
    else:
        dot_val = torch.dot(axis_sum, prismatic_axis_reference)
        if dot_val < 0:
            axis_sum = -axis_sum

    # Compute motion limits
    base_pt = all_points_history[0, 0]
    pts = all_points_history[:, 0, :]
    vecs = pts - base_pt.unsqueeze(0)

    # Project vectors onto axis
    val_ = torch.einsum('tj,j->t', vecs, axis_sum)
    min_proj = float(val_.min().item())
    max_proj = float(val_.max().item())

    return {
        "axis": axis_sum.cpu().numpy(),
        "origin": base_pt.cpu().numpy(),
        "motion_limit": (min_proj, max_proj)
    }


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
    w_r = 1.0 / (1.0 + ratio * ratio)
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
        prob_sigma (float): Width parameter for probability function
        prob_order (float): Order parameter for probability function
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

    # Compute joint type probabilities
    prismatic_pt = compute_joint_probability_new(col, cop, rad, zp, "prismatic", prob_sigma=prob_sigma,
                                                 prob_order=prob_order)
    planar_pt = compute_joint_probability_new(col, cop, rad, zp, "planar", prob_sigma=prob_sigma,
                                              prob_order=prob_order)
    revolute_pt = compute_joint_probability_new(col, cop, rad, zp, "revolute", prob_sigma=prob_sigma,
                                                prob_order=prob_order)
    screw_pt = compute_joint_probability_new(col, cop, rad, zp, "screw", prob_sigma=prob_sigma,
                                             prob_order=prob_order)
    ball_pt = compute_joint_probability_new(col, cop, rad, zp, "ball", prob_sigma=prob_sigma,
                                            prob_order=prob_order)

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