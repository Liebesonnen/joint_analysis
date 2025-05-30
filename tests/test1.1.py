import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import os
import math
from functools import partial
import torch
from scipy.signal import savgol_filter


####################################################################################
# Press R to move the planar object randomly; press B to move the ball joint randomly.
####################################################################################


################################################################################
#                             Utility Functions
################################################################################

def super_gaussian(x, sigma, order):
    """
    Super-Gaussian function: f(x) = exp(- (|x| / sigma)^order ).
    """
    return torch.exp(- (torch.abs(x) / sigma)**order)


def normalize_vector_torch(v, eps=1e-3):
    """
    Normalize vectors of shape (..., 3). 
    If the norm is smaller than eps, returns zero vector.
    """
    norm_v = torch.norm(v, dim=-1)
    mask = (norm_v > eps)
    out = torch.zeros_like(v)
    out[mask] = v[mask] / norm_v[mask].unsqueeze(-1)
    return out


def estimate_rotation_matrix_batch(pcd_src: torch.Tensor, pcd_tar: torch.Tensor):
    """
    Estimate rotation matrices in batch using SVD. 
    Input shapes: (B,N,3). Output shape: (B,3,3).
    """
    assert pcd_src.shape == pcd_tar.shape
    pcd_src_centered = pcd_src - pcd_src.mean(dim=1, keepdim=True)
    pcd_tar_centered = pcd_tar - pcd_tar.mean(dim=1, keepdim=True)

    cov_matrix = torch.einsum('bni,bnj->bij', pcd_src_centered, pcd_tar_centered)
    U, S, Vt = torch.linalg.svd(cov_matrix, full_matrices=False)
    R = torch.einsum('bij,bjk->bik', Vt.transpose(-1, -2), U.transpose(-1, -2))

    det_r = torch.det(R)
    flip_mask = det_r < 0
    if flip_mask.any():
        Vt[flip_mask, :, -1] *= -1
        R = torch.einsum('bij,bjk->bik', Vt.transpose(-1, -2), U.transpose(-1, -2))

    return R


def se3_log_map_batch(transform_matrices: torch.Tensor):
    """
    SE(3) log map in batch. Input: (B,4,4). Output: (B,6) => [translation, rotation].
    """
    B = transform_matrices.shape[0]
    R = transform_matrices[:, :3, :3]
    t = transform_matrices[:, :3, 3]

    trace = torch.einsum('bii->b', R)
    tmp = (trace - 1.0) / 2.0
    tmp = torch.clamp(tmp, min=-1.0, max=1.0)
    theta = torch.acos(tmp).unsqueeze(-1)

    omega = torch.zeros_like(R)
    log_R = torch.zeros_like(R)
    mask = theta.squeeze(-1) > 1e-3
    if mask.any():
        theta_masked = theta[mask].squeeze(-1)
        skew_symmetric = (R[mask] - R[mask].transpose(-1, -2)) / (
            2 * torch.sin(theta_masked).view(-1, 1, 1)
        )
        omega[mask] = theta_masked.view(-1, 1, 1) * skew_symmetric
        log_R[mask] = skew_symmetric * theta_masked.view(-1, 1, 1)

    A_inv = (
        torch.eye(3, device=transform_matrices.device)
        .repeat(B, 1, 1)
        - 0.5 * log_R
    )
    if mask.any():
        theta_sq = (theta[mask] ** 2).squeeze(-1)
        A_inv[mask] += (
            (1 - theta[mask].squeeze(-1) / (2 * torch.tan(theta[mask].squeeze(-1) / 2))) / theta_sq
        ).view(-1, 1, 1) * (log_R[mask] @ log_R[mask])

    v = torch.einsum('bij,bj->bi', A_inv, t)

    omega_vector = torch.stack(
        [-omega[:, 1, 2], omega[:, 0, 2], -omega[:, 0, 1]], dim=-1
    )
    se3_log = torch.cat([v, omega_vector], dim=-1)
    return se3_log


def find_neighbors_batch(pcd_batch: torch.Tensor, num_neighbor_pts: int) -> torch.Tensor:
    """
    Find nearest neighbors for each frame in a batch. 
    pcd_batch: (B,P,3).
    Returns an index tensor of shape (B,P,num_neighbor_pts).
    """
    dist = torch.cdist(pcd_batch, pcd_batch, p=2.0)
    neighbor_indices = torch.topk(dist, k=num_neighbor_pts, dim=-1, largest=False).indices
    return neighbor_indices


def calculate_velocity_and_angular_velocity_for_all_frames(
    all_points_history,
    dt=1.0,
    num_neighbors=50,
    use_savgol=True,
    savgol_window=5,
    savgol_poly=2
):
    """
    Compute linear and angular velocity for a sequence of point clouds (T,N,3).
    Returns velocity and angular velocity arrays: (T-1,N,3).
    An optional Savitzky-Golay filter is applied in time if (T-1) >= savgol_window.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not isinstance(all_points_history, torch.Tensor):
        all_points_history = torch.tensor(all_points_history, dtype=torch.float32, device=device)

    T, N, _ = all_points_history.shape
    if T < 2:
        return None, None

    pts_prev = all_points_history[:-1]  # (T-1,N,3)
    pts_curr = all_points_history[1:]   # (T-1,N,3)

    B = T - 1

    neighbor_idx_prev = find_neighbors_batch(pts_prev, num_neighbors)  # (T-1,N,K)
    neighbor_idx_curr = find_neighbors_batch(pts_curr, num_neighbors)  # (T-1,N,K)

    K = num_neighbors

    src_batch = pts_prev[
        torch.arange(B, device=device).view(B, 1, 1),
        neighbor_idx_prev,
        :
    ]
    tar_batch = pts_curr[
        torch.arange(B, device=device).view(B, 1, 1),
        neighbor_idx_curr,
        :
    ]

    src_2d = src_batch.reshape(B*N, K, 3)
    tar_2d = tar_batch.reshape(B*N, K, 3)

    R_2d = estimate_rotation_matrix_batch(src_2d, tar_2d)
    c1_2d = src_2d.mean(dim=1)
    c2_2d = tar_2d.mean(dim=1)
    delta_p_2d = c2_2d - c1_2d

    eye_4 = torch.eye(4, device=device).unsqueeze(0).expand(B*N, -1, -1).clone()
    eye_4[:, :3, :3] = R_2d
    eye_4[:, :3, 3] = delta_p_2d
    transform_matrices_2d = eye_4

    se3_logs_2d = se3_log_map_batch(transform_matrices_2d)
    v_2d = se3_logs_2d[:, :3] / dt
    w_2d = se3_logs_2d[:, 3:] / dt

    v_arr = v_2d.reshape(B, N, 3).cpu().numpy()
    w_arr = w_2d.reshape(B, N, 3).cpu().numpy()

    if use_savgol and B >= savgol_window:
        v_arr = savgol_filter(
            v_arr, window_length=savgol_window, polyorder=savgol_poly,
            axis=0, mode='mirror'
        )
        w_arr = savgol_filter(
            w_arr, window_length=savgol_window, polyorder=savgol_poly,
            axis=0, mode='mirror'
        )

    return v_arr, w_arr


def compute_motion_salience_batch(all_points_history, device='cuda'):
    """
    Compute motion salience for each point, defined as the cumulative norm of displacements.
    Input: (T,N,3). Output: (N,).
    """
    pts = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    T, N, _ = pts.shape
    if T < 2:
        return torch.zeros(N, device=device)
    diffs = pts[1:] - pts[:-1]
    dist_per_frame = torch.norm(diffs, dim=-1)
    ms = dist_per_frame.sum(dim=0)
    return ms


def compute_basic_scores(
    v_history, w_history, device='cuda',
    col_sigma=0.2, col_order=4.0,
    cop_sigma=0.2, cop_order=4.0,
    rad_sigma=0.2, rad_order=4.0,
    zp_sigma=0.2,  zp_order=4.0
):
    """
    Compute four basic scores: Collinearity, Coplanarity, Radius Consistency, and Zero Pitch.
    Each output is typically in [0,1].
    """
    eps_vec = 1e-3
    v_t = torch.as_tensor(v_history, dtype=torch.float32, device=device)
    w_t = torch.as_tensor(w_history, dtype=torch.float32, device=device)
    Tm1, N = v_t.shape[0], v_t.shape[1]

    v_norm = torch.norm(v_t, dim=2)
    w_norm = torch.norm(w_t, dim=2)
    mask_v = (v_norm > eps_vec)
    mask_w = (w_norm > eps_vec)

    v_clean = torch.zeros_like(v_t)
    w_clean = torch.zeros_like(w_t)
    v_clean[mask_v] = v_t[mask_v]
    w_clean[mask_w] = w_t[mask_w]

    # Collinearity and Coplanarity
    col_score = torch.ones(N, device=device)
    cop_score = torch.ones(N, device=device)
    if Tm1 >= 2:
        v_unit = normalize_vector_torch(v_clean)
        v_unit_all = v_unit.permute(1, 0, 2).contiguous()
        if Tm1 >= 3:
            U, S, V = torch.linalg.svd(v_unit_all, full_matrices=False)
            s1 = S[:, 0]
            s2 = S[:, 1]
            s3 = S[:, 2]
            eps_svd = 1e-6
            mask_svd = (s1 > eps_svd)
            ratio_col = torch.zeros_like(s1)
            ratio_cop = torch.zeros_like(s1)
            ratio_col[mask_svd] = s2[mask_svd] / s1[mask_svd]
            ratio_cop[mask_svd] = s3[mask_svd] / s1[mask_svd]
            col_score = super_gaussian(ratio_col, col_sigma, col_order)
            cop_score = super_gaussian(ratio_cop, cop_sigma, cop_order)
        else:
            U, S_, V = torch.linalg.svd(v_unit_all, full_matrices=False)
            s1 = S_[:, 0]
            s2 = S_[:, 1] if S_.size(1) > 1 else torch.zeros_like(s1)
            ratio_col = torch.zeros_like(s1)
            mask_svd = (s1 > 1e-6)
            ratio_col[mask_svd] = s2[mask_svd] / s1[mask_svd]
            col_score = super_gaussian(ratio_col, col_sigma, col_order)

    # Radius Consistency
    rad_score = torch.zeros(N, device=device)
    if Tm1 > 1:
        EPS_W = 1e-3
        v_mag = torch.norm(v_clean, dim=2)
        w_mag = torch.norm(w_clean, dim=2)
        r_mat = torch.zeros_like(v_mag)
        mask_w_nonzero = (w_mag > EPS_W)
        r_mat[mask_w_nonzero] = v_mag[mask_w_nonzero] / w_mag[mask_w_nonzero]
        r_i = r_mat[:-1]
        r_ip1 = r_mat[1:]
        diff_val = torch.zeros_like(r_i)
        valid_mask = (r_i > 1e-6) & (r_ip1 > 0)
        diff_val[valid_mask] = torch.abs(r_ip1[valid_mask] - r_i[valid_mask]) / (r_i[valid_mask] + 1e-6)
        rad_mat = super_gaussian(diff_val, rad_sigma, rad_order)
        mask_w_zero = (w_mag < EPS_W)
        mask_w_i = mask_w_zero[:-1]
        mask_w_ip1 = mask_w_zero[1:]
        rad_mat[mask_w_i] = 0
        rad_mat[mask_w_ip1] = 0
        rad_score = rad_mat.mean(dim=0)

    # Zero Pitch
    zp_score = torch.ones(N, device=device)
    if Tm1 > 0:
        v_u = normalize_vector_torch(v_clean)
        w_u = normalize_vector_torch(w_clean)
        dot_val = torch.sum(v_u * w_u, dim=2).abs()
        mean_dot = torch.mean(dot_val, dim=0)
        zp_score = super_gaussian(mean_dot, zp_sigma, zp_order)

    return col_score, cop_score, rad_score, zp_score


def compute_joint_probability_new(
    col, cop, rad, zp,
    joint_type="prismatic",
    prob_sigma=0.1,
    prob_order=4.0
):
    """
    Compute a joint probability metric for a given joint type.
    This is a simplified scoring function based on col, cop, rad, and zp.
    """
    if joint_type == "prismatic":
        e = ((col - 1)**2 + (cop - 1)**2 + (rad - 0)**2 + (zp - 1)**2) / 4
        return super_gaussian(e, prob_sigma, prob_order)
    elif joint_type == "planar":
        e = (col**2 + (cop - 1)**2 + rad**2 + (zp - 1)**2) / 4
        return super_gaussian(e, prob_sigma, prob_order)
    elif joint_type == "revolute":
        e = (col**2 + (cop - 1)**2 + (rad - 1)**2 + (zp - 1)**2) / 4
        return super_gaussian(e, prob_sigma, prob_order)
    elif joint_type == "screw":
        e = (col**2 + cop**2 + (rad - 1)**2 + zp**2) / 4
        return super_gaussian(e, prob_sigma, prob_order)
    elif joint_type == "ball":
        e = (col**2 + cop**2 + (rad - 1)**2 + (zp - 1)**2) / 4
        return super_gaussian(e, prob_sigma, prob_order)
    return torch.zeros_like(col)


################################################################################
#                  Joint Geometry Computation Functions
################################################################################

def compute_planar_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Computes an approximate plane normal and a rough 2D bounding box in the plane.
    """
    all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    v_history = torch.as_tensor(v_history, dtype=torch.float32, device=device)
    omega_history = torch.as_tensor(omega_history, dtype=torch.float32, device=device)
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)

    T, N = all_points_history.shape[0], all_points_history.shape[1]
    if T < 3:
        return {"normal": np.array([0., 0., 0.]), "motion_limit": (0., 0.)}

    v_ref_unit = normalize_vector_torch(v_history[0])
    v_unit_hist = normalize_vector_torch(v_history)
    T2 = T - 2
    cross_sum = torch.zeros((N,3), device=device)
    cross_count = torch.zeros(N, device=device)

    if T2 >= 1:
        v_ref_expand = v_ref_unit.unsqueeze(0).expand(T2, N, 3)
        v_unit_sub = v_unit_hist[1:T-1]
        cross_mat = torch.cross(v_ref_expand, v_unit_sub, dim=2)
        cross_norm = torch.norm(cross_mat, dim=2)
        cross_valid = (cross_norm > 1e-6)

        cross_unit = torch.zeros_like(cross_mat)
        cross_unit[cross_valid] = cross_mat[cross_valid] / cross_norm[cross_valid].unsqueeze(-1)
        cross_count = cross_valid.sum(dim=0)
        cross_sum = cross_unit.sum(dim=0)

    normal_ = torch.zeros((N,3), device=device)
    mask_nonzero = (cross_count > 0)
    normal_[mask_nonzero] = cross_sum[mask_nonzero] / cross_count[mask_nonzero].unsqueeze(-1)
    normal_ = normalize_vector_torch(normal_)

    normal_sum = torch.sum(w_i[:,None] * normal_, dim=0)
    normal_sum = normalize_vector_torch(normal_sum.unsqueeze(0))[0]

    if T < 2:
        motion_limit = (0., 0.)
    else:
        pt0_0 = all_points_history[0, 0]
        v1_0 = v_ref_unit[0]
        if torch.norm(v1_0) < 1e-6:
            v1_0 = torch.tensor([1., 0., 0.], device=device)
        axis1 = normalize_vector_torch(v1_0.unsqueeze(0))[0]
        axis2 = torch.cross(normal_sum, axis1)
        axis2 = normalize_vector_torch(axis2.unsqueeze(0))[0]

        pts = all_points_history[:, 0] - pt0_0
        proj1 = torch.matmul(pts, axis1)
        proj2 = torch.matmul(pts, axis2)
        motion_limit = (float(proj1.max().item()), float(proj2.max().item()))

    return {
        "normal": normal_sum.cpu().numpy(),
        "motion_limit": motion_limit
    }


def compute_ball_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Computes an approximate center for a ball joint and an angle range (motion_limit).
    """
    all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    v_history = torch.as_tensor(v_history, dtype=torch.float32, device=device)
    omega_history = torch.as_tensor(omega_history, dtype=torch.float32, device=device)
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)

    T, N = all_points_history.shape[0], all_points_history.shape[1]
    if T < 2:
        return {"center": np.array([0., 0., 0.]), "motion_limit": (0., 0., 0.)}

    v_norm = torch.norm(v_history, dim=2)
    w_norm = torch.norm(omega_history, dim=2)
    EPS_W = 1e-3
    mask_w = (w_norm > EPS_W)
    r_mat = torch.zeros_like(v_norm)
    r_mat[mask_w] = v_norm[mask_w] / w_norm[mask_w]

    v_u = normalize_vector_torch(v_history)
    w_u = normalize_vector_torch(omega_history)
    dir_ = - torch.cross(v_u, w_u, dim=2)
    dir_ = normalize_vector_torch(dir_)
    r_3d = r_mat.unsqueeze(-1)
    c_pos = all_points_history[:-1] + dir_ * r_3d
    center_each = torch.mean(c_pos, dim=0)

    center_sum = torch.sum(w_i[:, None] * center_each, dim=0)

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
    motion_limit = (max_angle, max_angle, max_angle)

    return {
        "center": center_sum.cpu().numpy(),
        "motion_limit": motion_limit
    }


def compute_screw_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Computes axis, origin, pitch, and a rough motion limit for a screw joint.
    """
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

    EPS_W = 1e-3
    v_u = normalize_vector_torch(v_history)
    w_u = normalize_vector_torch(omega_history)
    v_norm = torch.norm(v_history, dim=2)
    w_norm = torch.norm(omega_history, dim=2)

    s_mat = torch.sum(v_history * w_u, dim=2)
    ratio = torch.zeros_like(s_mat)
    mask_w = (w_norm > EPS_W)
    ratio[mask_w] = s_mat[mask_w] / (v_norm[mask_w] + 1e-6)
    pitch_all = torch.acos(torch.clamp(ratio, -1., 1.))

    pitch_sin = torch.sin(pitch_all)
    r_ = torch.zeros_like(v_norm)
    mask_ws = (w_norm > EPS_W)
    r_[mask_ws] = (v_norm[mask_ws] * pitch_sin[mask_ws]) / (w_norm[mask_ws] + 1e-6)
    dir_ = - torch.cross(v_u, w_u, dim=2)
    dir_ = normalize_vector_torch(dir_)
    r_3d = r_.unsqueeze(-1)
    c_pos = all_points_history[:-1] + dir_ * r_3d
    origin_each = torch.mean(c_pos, dim=0)

    axis_each = torch.mean(w_u, dim=0)
    axis_each = normalize_vector_torch(axis_each)

    axis_sum = torch.sum(w_i[:, None] * axis_each, dim=0)
    axis_sum = normalize_vector_torch(axis_sum.unsqueeze(0))[0]
    pitch_each = torch.mean(pitch_all, dim=0)
    pitch_sum = torch.sum(w_i * pitch_each)
    origin_sum = torch.sum(w_i[:, None] * origin_each, dim=0)

    i0 = 0
    base_pt = all_points_history[0, i0]
    base_vec = base_pt - origin_sum
    nb = torch.norm(base_vec) + 1e-6
    pts = all_points_history[:, i0, :]
    vecs = pts - origin_sum
    dotv = torch.sum(vecs * base_vec.unsqueeze(0), dim=1)
    n_v = torch.norm(vecs, dim=1) + 1e-6
    cosval = torch.clamp(dotv / (nb * n_v), -1., 1.)
    angles = torch.acos(cosval)
    motion_limit = (float(angles.min().item()), float(angles.max().item()))

    return {
        "axis": axis_sum.cpu().numpy(),
        "origin": origin_sum.cpu().numpy(),
        "pitch": float(pitch_sum.item()),
        "motion_limit": motion_limit
    }


def compute_prismatic_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Computes the axis for prismatic motion and a bounding range along that axis.
    """
    all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    v_history = torch.as_tensor(v_history, dtype=torch.float32, device=device)
    omega_history = torch.as_tensor(omega_history, dtype=torch.float32, device=device)
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)

    T, N = all_points_history.shape[0], all_points_history.shape[1]
    if T < 2:
        return {
            "axis": np.array([0., 0., 0.]),
            "motion_limit": (0., 0.)
        }

    v_u = normalize_vector_torch(v_history)
    v_mean = torch.mean(v_u, dim=0)
    v_mean = normalize_vector_torch(v_mean)
    dir_sum = torch.sum(w_i[:, None] * v_mean, dim=0)
    dir_sum = normalize_vector_torch(dir_sum.unsqueeze(0))[0]

    base_pt = all_points_history[0, 0]
    pts = all_points_history[:, 0, :]
    vecs = pts - base_pt.unsqueeze(0)
    val_ = torch.matmul(vecs, dir_sum)
    motion_limit = (float(val_.min().item()), float(val_.max().item()))

    return {
        "axis": dir_sum.cpu().numpy(),
        "motion_limit": motion_limit
    }


def compute_revolute_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Computes the axis, origin, and angular bounds for a revolute joint.
    """
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

    v_norm = torch.norm(v_history, dim=2)
    w_norm = torch.norm(omega_history, dim=2)
    mask_w = (w_norm > 1e-6)
    r_mat = torch.zeros_like(v_norm)
    r_mat[mask_w] = v_norm[mask_w] / w_norm[mask_w]

    v_u = normalize_vector_torch(v_history)
    w_u = normalize_vector_torch(omega_history)
    dir_ = - torch.cross(v_u, w_u, dim=2)
    dir_ = normalize_vector_torch(dir_)
    r_3d = r_mat.unsqueeze(-1)
    c_pos = all_points_history[:-1] + dir_ * r_3d
    origin_each = torch.mean(c_pos, dim=0)

    axis_ave = torch.mean(w_u, dim=0)
    axis_ave = normalize_vector_torch(axis_ave)
    axis_sum = torch.sum(w_i[:, None] * axis_ave, dim=0)
    axis_sum = normalize_vector_torch(axis_sum.unsqueeze(0))[0]
    origin_sum = torch.sum(w_i[:, None] * origin_each, dim=0)

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
    motion_limit = (float(angles.min().item()), float(angles.max().item()))

    return {
        "axis": axis_sum.cpu().numpy(),
        "origin": origin_sum.cpu().numpy(),
        "motion_limit": motion_limit
    }


################################################################################
#      Compute geometry + basic scores + joint probabilities + best joint
################################################################################

def compute_joint_info_all_types(
    all_points_history,
    col_sigma=0.2, col_order=4.0,
    cop_sigma=0.2, cop_order=4.0,
    rad_sigma=0.2, rad_order=4.0,
    zp_sigma=0.2,  zp_order=4.0,
    prob_sigma=0.2, prob_order=4.0,
    use_savgol=True,
    savgol_window=5,
    savgol_poly=2
):
    """
    1) Compute velocity and angular velocity.
    2) Compute four basic scores (col, cop, rad, zp).
    3) Compute probabilities for five joint types.
    4) Pick the best joint type (if best prob < 0.3 => "Unknown").
    5) Compute geometry for each joint type.
    Returns (dict_of_geometry, best_joint_name, info_dict).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = all_points_history.shape[0]
    N = all_points_history.shape[1]
    if T < 2:
        ret = {
            "planar": {"normal": np.array([0.,0.,0.]), "motion_limit": (0.,0.)},
            "ball":   {"center": np.array([0.,0.,0.]), "motion_limit": (0.,0.,0.)},
            "screw":  {"axis":   np.array([0.,0.,0.]), "origin": np.array([0.,0.,0.]), "pitch":0., "motion_limit":(0.,0.)},
            "prismatic":{"axis": np.array([0.,0.,0.]), "motion_limit":(0.,0.)},
            "revolute":{"axis": np.array([0.,0.,0.]), "origin": np.array([0.,0.,0.]), "motion_limit":(0.,0.)}
        }
        return ret, "Unknown", None

    dt = 1.0
    v_arr, w_arr = calculate_velocity_and_angular_velocity_for_all_frames(
        all_points_history,
        dt=dt,
        num_neighbors=50,
        use_savgol=use_savgol,
        savgol_window=savgol_window,
        savgol_poly=savgol_poly
    )
    if v_arr is None or w_arr is None:
        ret = {
            "planar": {"normal": np.array([0.,0.,0.]), "motion_limit": (0.,0.)},
            "ball":   {"center": np.array([0.,0.,0.]), "motion_limit": (0.,0.,0.)},
            "screw":  {"axis":   np.array([0.,0.,0.]), "origin": np.array([0.,0.,0.]), "pitch":0., "motion_limit":(0.,0.)},
            "prismatic":{"axis": np.array([0.,0.,0.]), "motion_limit":(0.,0.)},
            "revolute":{"axis": np.array([0.,0.,0.]), "origin": np.array([0.,0.,0.]), "motion_limit":(0.,0.)}
        }
        return ret, "Unknown", None

    ms_torch = compute_motion_salience_batch(all_points_history, device=device)
    ms = ms_torch.cpu().numpy()
    sum_ms = ms.sum()
    if sum_ms < 1e-6:
        w_i = np.ones(N) / N
    else:
        w_i = ms / sum_ms

    p_info = compute_planar_info(all_points_history, v_arr, w_arr, w_i, device=device)
    b_info = compute_ball_info(all_points_history, v_arr, w_arr, w_i, device=device)
    s_info = compute_screw_info(all_points_history, v_arr, w_arr, w_i, device=device)
    pm_info= compute_prismatic_info(all_points_history, v_arr, w_arr, w_i, device=device)
    r_info = compute_revolute_info(all_points_history, v_arr, w_arr, w_i, device=device)

    ret = {
        "planar": p_info,
        "ball":   b_info,
        "screw":  s_info,
        "prismatic": pm_info,
        "revolute": r_info
    }

    col, cop, rad, zp = compute_basic_scores(
        v_arr, w_arr, device=device,
        col_sigma=col_sigma, col_order=col_order,
        cop_sigma=cop_sigma, cop_order=cop_order,
        rad_sigma=rad_sigma, rad_order=rad_order,
        zp_sigma=zp_sigma,   zp_order=zp_order
    )
    w_t = torch.tensor(w_i, dtype=torch.float32, device=device)
    col_mean = float(torch.sum(w_t * col).item())
    cop_mean = float(torch.sum(w_t * cop).item())
    rad_mean = float(torch.sum(w_t * rad).item())
    zp_mean  = float(torch.sum(w_t * zp).item())

    basic_score_avg = {
        "col_mean": col_mean,
        "cop_mean": cop_mean,
        "rad_mean": rad_mean,
        "zp_mean":  zp_mean
    }

    prismatic_pt = compute_joint_probability_new(col, cop, rad, zp, "prismatic", prob_sigma=prob_sigma, prob_order=prob_order)
    planar_pt    = compute_joint_probability_new(col, cop, rad, zp, "planar",    prob_sigma=prob_sigma, prob_order=prob_order)
    revolute_pt  = compute_joint_probability_new(col, cop, rad, zp, "revolute",  prob_sigma=prob_sigma, prob_order=prob_order)
    screw_pt     = compute_joint_probability_new(col, cop, rad, zp, "screw",     prob_sigma=prob_sigma, prob_order=prob_order)
    ball_pt      = compute_joint_probability_new(col, cop, rad, zp, "ball",      prob_sigma=prob_sigma, prob_order=prob_order)

    prismatic_prob = float(torch.sum(w_t * prismatic_pt).item())
    planar_prob    = float(torch.sum(w_t * planar_pt).item())
    revolute_prob  = float(torch.sum(w_t * revolute_pt).item())
    screw_prob     = float(torch.sum(w_t * screw_pt).item())
    ball_prob      = float(torch.sum(w_t * ball_pt).item())

    joint_probs = [
        ("prismatic", prismatic_prob),
        ("planar", planar_prob),
        ("revolute", revolute_prob),
        ("screw", screw_prob),
        ("ball", ball_prob),
    ]
    joint_probs.sort(key=lambda x: x[1], reverse=True)
    best_joint, best_pval = joint_probs[0]
    if best_pval < 0.3:
        best_joint = "Unknown"

    ret_probs = {
        "prismatic": prismatic_prob,
        "planar":    planar_prob,
        "revolute":  revolute_prob,
        "screw":     screw_prob,
        "ball":      ball_prob
    }

    info_dict = {
        "basic_score_avg": basic_score_avg,
        "joint_probs": ret_probs
    }

    return ret, best_joint, info_dict


################################################################################
#                               Polyscope Logic
################################################################################

point_cloud_history = {}
file_counter = {}

def store_point_cloud(points, joint_type):
    key = joint_type.replace(" ", "_")
    if key not in point_cloud_history:
        point_cloud_history[key] = []
    point_cloud_history[key].append(points.copy())


def save_all_to_npy(joint_type):
    output_dir = "exported_pointclouds"
    os.makedirs(output_dir, exist_ok=True)
    joint_dir = os.path.join(output_dir, joint_type.replace(" ", "_"))
    os.makedirs(joint_dir, exist_ok=True)

    key = joint_type.replace(" ", "_")
    if key not in point_cloud_history or len(point_cloud_history[key]) == 0:
        print("No data to save for", joint_type)
        return

    global file_counter
    if joint_type not in file_counter:
        file_counter[joint_type] = 0
    else:
        file_counter[joint_type] += 1

    all_points = np.stack(point_cloud_history[key], axis=0)
    filename = f"{joint_type.replace(' ', '_')}_{file_counter[joint_type]}.npy"
    filepath = os.path.join(joint_dir, filename)
    np.save(filepath, all_points)
    print("Saved data to", filepath)


tracked_indices = {}
tracked_points_history = {}
neighbors_history = {}
velocity_history = {}
angular_velocity_history = {}
velocity_history_per_point = {}
angular_velocity_history_per_point = {}
point_weights = {}

modes = ["Prismatic Door", "Revolute Door", "Drawer", "Planar Mouse", "Ball Joint", "Screw Joint"]
for m in modes:
    tracked_indices[m] = None
    tracked_points_history[m] = {"p1": [], "p2": [], "p3": []}
    neighbors_history[m] = {"p1": [], "p2": [], "p3": []}
    velocity_history[m] = []
    angular_velocity_history[m] = []
    velocity_history_per_point[m] = []
    angular_velocity_history_per_point[m] = []
    point_weights[m] = None


def clear_data_for_mode(mode):
    key = mode.replace(" ", "_")
    if key in point_cloud_history:
        point_cloud_history[key] = []

    tracked_indices[mode] = None
    tracked_points_history[mode] = {"p1": [], "p2": [], "p3": []}
    neighbors_history[mode] = {"p1": [], "p2": [], "p3": []}
    velocity_history[mode] = []
    angular_velocity_history[mode] = []
    velocity_history_per_point[mode] = []
    angular_velocity_history_per_point[mode] = []
    point_weights[mode] = None


def remove_joint_visual():
    for name in [
        "Planar Normal", "Ball Center", "Screw Axis", "Screw Axis Pitch",
        "Prismatic Axis", "Revolute Axis", "Revolute Origin", "Planar Axes"
    ]:
        if ps.has_curve_network(name):
            ps.remove_curve_network(name)
    if ps.has_point_cloud("BallCenterPC"):
        ps.remove_point_cloud("BallCenterPC")


def show_joint_visual(joint_type, joint_params):
    remove_joint_visual()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps = 1e-6

    def torch_normalize(vec_t):
        norm_ = torch.norm(vec_t)
        if norm_ < eps:
            return vec_t
        return vec_t / norm_

    if joint_type == "planar":
        n_np = joint_params.get("normal", np.array([0., 0., 1.]))
        seg_nodes = np.array([[0, 0, 0], n_np])
        seg_edges = np.array([[0, 1]])
        planarnet = ps.register_curve_network("Planar Normal", seg_nodes, seg_edges)
        planarnet.set_color((1.0, 0.0, 0.0))
        planarnet.set_radius(0.02)

        n_t = torch.tensor(n_np, device=device)
        y_t = torch.tensor([0., 1., 0.], device=device)
        cross_1 = torch.cross(n_t, y_t)
        cross_1 = torch_normalize(cross_1)
        if torch.norm(cross_1) < eps:
            cross_1 = torch.tensor([1., 0., 0.], device=device)
        cross_2 = torch.cross(n_t, cross_1)
        cross_2 = torch_normalize(cross_2.unsqueeze(0))[0]

        seg_nodes2 = np.array([
            [0, 0, 0], cross_1.cpu().numpy(),
            [0, 0, 0], cross_2.cpu().numpy()
        ], dtype=np.float32)
        seg_edges2 = np.array([[0, 1], [2, 3]])
        planarex = ps.register_curve_network("Planar Axes", seg_nodes2, seg_edges2)
        planarex.set_color((0., 1., 0.))
        planarex.set_radius(0.02)

    elif joint_type == "ball":
        center_np = joint_params.get("center", np.array([0., 0., 0.]))
        c_pc = ps.register_point_cloud("BallCenterPC", center_np.reshape(1, 3))
        c_pc.set_radius(0.05)
        c_pc.set_enabled(True)

        x_ = np.array([1., 0., 0.])
        y_ = np.array([0., 1., 0.])
        z_ = np.array([0., 0., 1.])
        seg_nodes = np.array([
            center_np, center_np + x_,
            center_np, center_np + y_,
            center_np, center_np + z_
        ])
        seg_edges = np.array([[0, 1], [2, 3], [4, 5]])
        axisviz = ps.register_curve_network("Ball Center", seg_nodes, seg_edges)
        axisviz.set_radius(0.02)
        axisviz.set_color((1., 0., 1.))

    elif joint_type == "screw":
        axis_np = joint_params.get("axis", np.array([0., 1., 0.]))
        origin_np = joint_params.get("origin", np.array([0., 0., 0.]))
        pitch_ = joint_params.get("pitch", 0.0)

        seg_nodes = np.array([
            origin_np - axis_np * 0.5,
            origin_np + axis_np * 0.5
        ])
        seg_edges = np.array([[0, 1]])
        scv = ps.register_curve_network("Screw Axis", seg_nodes, seg_edges)
        scv.set_radius(0.02)
        scv.set_color((0., 0., 1.0))

        pitch_arrow_start = origin_np + axis_np * 0.6
        pitch_arrow_end = pitch_arrow_start + 0.2 * pitch_ * np.array([1, 0, 0])
        seg_nodes2 = np.array([pitch_arrow_start, pitch_arrow_end])
        seg_edges2 = np.array([[0, 1]])
        pitch_net = ps.register_curve_network("Screw Axis Pitch", seg_nodes2, seg_edges2)
        pitch_net.set_color((1., 0., 0.))
        pitch_net.set_radius(0.02)

    elif joint_type == "prismatic":
        axis_np = joint_params.get("axis", np.array([1., 0., 0.]))
        seg_nodes = np.array([[0., 0., 0.], axis_np])
        seg_edges = np.array([[0, 1]])
        pcv = ps.register_curve_network("Prismatic Axis", seg_nodes, seg_edges)
        pcv.set_radius(0.02)
        pcv.set_color((0., 1., 1.))

    elif joint_type == "revolute":
        axis_np = joint_params.get("axis", np.array([0., 1., 0.]))
        origin_np = joint_params.get("origin", np.array([0., 0., 0.]))
        seg_nodes = np.array([
            origin_np - axis_np * 0.5,
            origin_np + axis_np * 0.5
        ])
        seg_edges = np.array([[0, 1]])
        rvnet = ps.register_curve_network("Revolute Axis", seg_nodes, seg_edges)
        rvnet.set_radius(0.02)
        rvnet.set_color((1., 1., 0.))

        seg_nodes2 = np.array([origin_np, origin_np])
        seg_edges2 = np.array([[0, 1]])
        origin_net = ps.register_curve_network("Revolute Origin", seg_nodes2, seg_edges2)
        origin_net.set_radius(0.03)
        origin_net.set_color((1., 0., 0.))


def highlight_max_points(ps_cloud, current_points, prev_points, mode):
    if prev_points is None or current_points.shape != prev_points.shape:
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    curr_t = torch.tensor(current_points, dtype=torch.float32, device=device)
    prev_t = torch.tensor(prev_points,  dtype=torch.float32, device=device)
    dist_t = torch.norm(curr_t - prev_t, dim=1)
    max_d_t = torch.max(dist_t)
    P = current_points.shape[0]

    if max_d_t < 1e-3:
        colors_t = torch.full((P, 3), 0.7, device=device, dtype=torch.float32)
    else:
        ratio_t = dist_t / max_d_t
        r_ = ratio_t
        g_ = 0.5 * (1.0 - ratio_t)
        b_ = 1.0 - ratio_t
        colors_t = torch.stack([r_, g_, b_], dim=-1)
    colors_np = colors_t.cpu().numpy()
    ps_cloud.add_color_quantity("Deviation Highlight", colors_np, enabled=True)


ps.init()
ps.set_ground_plane_mode("none")

output_dir = "exported_pointclouds"
os.makedirs(output_dir, exist_ok=True)

current_mode = "Prismatic Door"
previous_mode = None

dt = 0.1
prev_points_for_deviation = None

current_joint_info_dict = {}
current_best_joint = "Unknown"
current_scores_info = None

noise_sigma = 0.0
col_sigma   = 0.2
col_order   = 4.0
cop_sigma   = 0.2
cop_order   = 4.0
rad_sigma   = 0.2
rad_order   = 4.0
zp_sigma    = 0.2
zp_order    = 4.0
prob_sigma  = 0.2
prob_order  = 4.0

savgol_window_length = 5
savgol_polyorder     = 2

def compute_point_weights_if_needed(mode):
    key = mode.replace(" ", "_")
    if point_weights[mode] is None:
        if key in point_cloud_history and len(point_cloud_history[key]) >= 2:
            pts0 = point_cloud_history[key][0]
            pts1 = point_cloud_history[key][1]
            disp = np.linalg.norm(pts1 - pts0, axis=1)
            sum_disp = np.sum(disp)
            if sum_disp < 1e-6:
                w = np.ones_like(disp) / disp.shape[0]
            else:
                w = disp / sum_disp
            point_weights[mode] = w


################################################################################
#                  Random geometry for demonstration
################################################################################

door_width, door_height, door_thickness = 2.0, 3.0, 0.2
num_points = 500

original_prismatic_door_points = np.random.rand(num_points, 3)
original_prismatic_door_points[:, 0] = original_prismatic_door_points[:, 0] * door_width - 0.5 * door_width
original_prismatic_door_points[:, 1] = original_prismatic_door_points[:, 1] * door_height
original_prismatic_door_points[:, 2] = original_prismatic_door_points[:, 2] * door_thickness - 0.5 * door_thickness
prismatic_door_points = original_prismatic_door_points.copy()
door_position = 0.0
door_max_displacement = 5.0
door_axis = np.array([1.0, 0.0, 0.0])
ps_prismatic_door = ps.register_point_cloud("Prismatic Door", prismatic_door_points)

original_revolute_door_points = prismatic_door_points.copy()
revolute_door_points = original_revolute_door_points.copy()
door_angle = 0.0
door_hinge_position = np.array([1.0, 0.0, 0.0])
door_hinge_axis = np.array([0.0, 1.0, 0.0])
ps_revolute_door = ps.register_point_cloud("Revolute Door", revolute_door_points, enabled=False)

drawer_width, drawer_height, drawer_depth = 1.5, 1.0, 2.0
original_drawer_points = np.random.rand(num_points, 3)
original_drawer_points[:, 0] = original_drawer_points[:, 0] * drawer_width - 0.5 * drawer_width
original_drawer_points[:, 1] = original_drawer_points[:, 1] * drawer_height
original_drawer_points[:, 2] = original_drawer_points[:, 2] * drawer_depth
drawer_points = original_drawer_points.copy()
drawer_position = 0.0
drawer_max_displacement = 3.0
drawer_axis = np.array([0.0, 0.0, 1.0])
ps_drawer = ps.register_point_cloud("Drawer", drawer_points, enabled=False)

mouse_length, mouse_width, mouse_height = 1.0, 0.6, 0.3
original_mouse_points = np.zeros((num_points, 3))
original_mouse_points[:, 0] = np.random.rand(num_points) * mouse_length - 0.5 * mouse_length
original_mouse_points[:, 2] = np.random.rand(num_points) * mouse_width - 0.5 * mouse_width
original_mouse_points[:, 1] = np.random.rand(num_points) * mouse_height
mouse_points = original_mouse_points.copy()
translation_x = 0.0
translation_z = 0.0
rotation_y_planar = 0.0
ps_mouse = ps.register_point_cloud("Planar Mouse", mouse_points, enabled=False)

def generate_sphere(center, radius, num_points):
    phi = np.random.rand(num_points) * 2 * np.pi
    costheta = 2 * np.random.rand(num_points) - 1
    theta = np.arccos(costheta)
    r = radius * (np.random.rand(num_points) ** (1 / 3))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return center + np.vstack([x, y, z]).T

def generate_cylinder(radius, height, num_points=500):
    zs = np.random.rand(num_points) * height - height / 2
    phi = np.random.rand(num_points) * 2 * np.pi
    rs = radius * np.sqrt(np.random.rand(num_points))
    xs = rs * np.cos(phi)
    ys = zs
    zs = rs * np.sin(phi)
    return np.vstack([xs, ys, zs]).T

def generate_ball_joint_points(center, sphere_radius, rod_length, rod_radius,
                               num_points_sphere=250, num_points_rod=250):
    sphere_pts = generate_sphere(center, sphere_radius, num_points_sphere)
    rod_pts = generate_cylinder(rod_radius, rod_length, num_points_rod)
    rod_pts[:, 1] += center[1]
    return np.concatenate([sphere_pts, rod_pts], axis=0)

joint_center = np.array([0.0, 0.0, 0.0])
sphere_radius = 0.3
rod_length = sphere_radius * 10.0
rod_radius = 0.05
num_points_sphere = 250
num_points_rod = 250
original_joint_points = generate_ball_joint_points(
    joint_center,
    sphere_radius,
    rod_length,
    rod_radius,
    num_points_sphere,
    num_points_rod
)
joint_points = original_joint_points.copy()
rotation_angle_x = 0.0
rotation_angle_y = 0.0
rotation_angle_z = 0.0
ps_joint = ps.register_point_cloud("Ball Joint", joint_points, enabled=False)

def generate_hollow_cylinder(radius, height, thickness,
                             num_points=500, cap_position="top", cap_points_ratio=0.2):
    num_wall_points = int(num_points * (1 - cap_points_ratio))
    num_cap_points = num_points - num_wall_points
    rr_wall = radius - np.random.rand(num_wall_points) * thickness
    theta_wall = np.random.rand(num_wall_points) * 2 * np.pi
    z_wall = np.random.rand(num_wall_points) * height
    x_wall = rr_wall * np.cos(theta_wall)
    zs = rr_wall * np.sin(theta_wall)
    y_wall = z_wall
    z_wall = zs

    rr_cap = np.random.rand(num_cap_points) * radius
    theta_cap = np.random.rand(num_cap_points) * 2 * np.pi
    z_cap = np.full_like(rr_cap, height if cap_position == "top" else 0.0)
    x_cap = rr_cap * np.cos(theta_cap)
    y_cap = z_cap
    z_cap = rr_cap * np.sin(theta_cap)

    x = np.hstack([x_wall, x_cap])
    y = np.hstack([y_wall, y_cap])
    z = np.hstack([z_wall, z_cap])
    return np.vstack([x, y, z]).T

screw_axis = np.array([0.0, 1.0, 0.0])
screw_pitch = 0.5
origin = np.array([0.0, 0.0, 0.0])
screw_angle = 0.0
cap_outer_radius = 0.4
cap_height = 0.2
cap_thickness = 0.05
num_points_cap = 500
original_cap_points = generate_hollow_cylinder(cap_outer_radius,
                                               cap_height,
                                               cap_thickness,
                                               num_points_cap)
cap_points = original_cap_points.copy()
ps_cap = ps.register_point_cloud("Bottle Cap", cap_points, enabled=False)


def translate_points(points, displacement, axis):
    return points + displacement * axis

def rotate_points(points, angle, axis, origin):
    points = points - origin
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    R = np.array([
        [t * axis[0]*axis[0] + c,         t*axis[0]*axis[1] - s*axis[2], t*axis[0]*axis[2] + s*axis[1]],
        [t * axis[0]*axis[1] + s*axis[2], t*axis[1]*axis[1] + c,         t*axis[1]*axis[2] - s*axis[0]],
        [t * axis[0]*axis[2] - s*axis[1], t*axis[1]*axis[2] + s*axis[0], t*axis[2]*axis[2] + c]
    ])
    rotated_points = points @ R.T
    rotated_points += origin
    return rotated_points

def apply_planar_motion(points, dx, dz):
    return points + np.array([dx, 0., dz])

def rotate_points_y(points, angle, center):
    points = points - center
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([
        [ c, 0.,  s],
        [0., 1., 0.],
        [-s, 0.,  c]
    ])
    rotated_points = points @ R.T
    rotated_points += center
    return rotated_points

def rotate_points_xyz(points, angle_x, angle_y, angle_z, center):
    points = points - center
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x),  np.cos(angle_x)]
    ])
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z),  np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    rotated = points @ Rx.T @ Ry.T @ Rz.T
    rotated += center
    return rotated

def apply_screw_motion(points, angle, axis, origin, pitch):
    points = points - origin
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    R = np.array([
        [t*axis[0]*axis[0] + c,         t*axis[0]*axis[1] - s*axis[2], t*axis[0]*axis[2] + s*axis[1]],
        [t*axis[0]*axis[1] + s*axis[2], t*axis[1]*axis[1] + c,         t*axis[1]*axis[2] - s*axis[0]],
        [t*axis[0]*axis[2] - s*axis[1], t*axis[1]*axis[2] + s*axis[0], t*axis[2]*axis[2] + c]
    ])
    rotated_points = points @ R.T
    translation = (angle / (2*np.pi)) * pitch * axis
    transformed_points = rotated_points + translation
    transformed_points += origin
    return transformed_points

def restore_original_shape(mode):
    """
    Restore objects to their initial transforms for each mode.
    """
    global prismatic_door_points, revolute_door_points, drawer_points
    global mouse_points, joint_points, cap_points
    global rotation_angle_x, rotation_angle_y, rotation_angle_z
    global translation_x, translation_z, rotation_y_planar
    global screw_angle, door_position, drawer_position
    global door_angle

    ps_prismatic_door.set_enabled(mode == "Prismatic Door")
    ps_revolute_door.set_enabled(mode == "Revolute Door")
    ps_drawer.set_enabled(mode == "Drawer")
    ps_mouse.set_enabled(mode == "Planar Mouse")
    ps_joint.set_enabled(mode == "Ball Joint")
    ps_cap.set_enabled(mode == "Screw Joint")

    if mode == "Prismatic Door":
        door_position = 0.0
        prismatic_door_points = original_prismatic_door_points.copy()
        prismatic_door_points = translate_points(prismatic_door_points, door_position, door_axis)
        ps_prismatic_door.update_point_positions(prismatic_door_points)

    elif mode == "Revolute Door":
        door_angle = 0.0
        revolute_door_points = original_revolute_door_points.copy()
        revolute_door_points = rotate_points(revolute_door_points, door_angle, door_hinge_axis, door_hinge_position)
        ps_revolute_door.update_point_positions(revolute_door_points)

    elif mode == "Drawer":
        drawer_position = 0.0
        drawer_points = original_drawer_points.copy()
        drawer_points = translate_points(drawer_points, drawer_position, drawer_axis)
        ps_drawer.update_point_positions(drawer_points)

    elif mode == "Planar Mouse":
        translation_x = 0.0
        translation_z = 0.0
        rotation_y_planar = 0.0
        mouse_points = original_mouse_points.copy()
        ps_mouse.update_point_positions(mouse_points)

    elif mode == "Ball Joint":
        rotation_angle_x = rotation_angle_y = rotation_angle_z = 0.0
        joint_points = original_joint_points.copy()
        ps_joint.update_point_positions(joint_points)

    elif mode == "Screw Joint":
        screw_angle = 0.0
        cap_points = original_cap_points.copy()
        ps_cap.update_point_positions(cap_points)

    store_point_cloud_for_mode(mode)


def store_point_cloud_for_mode(mode):
    if mode == "Prismatic Door":
        store_point_cloud(prismatic_door_points, mode)
    elif mode == "Revolute Door":
        store_point_cloud(revolute_door_points, mode)
    elif mode == "Drawer":
        store_point_cloud(drawer_points, mode)
    elif mode == "Planar Mouse":
        store_point_cloud(mouse_points, mode)
    elif mode == "Ball Joint":
        store_point_cloud(joint_points, mode)
    elif mode == "Screw Joint":
        store_point_cloud(cap_points, mode)


device = "cuda" if torch.cuda.is_available() else "cpu"

planar_dofs = torch.zeros(3, dtype=torch.float32, device=device)
planar_vel  = torch.zeros(3, dtype=torch.float32, device=device)

ball_dofs   = torch.zeros(3, dtype=torch.float32, device=device)
ball_vel    = torch.zeros(3, dtype=torch.float32, device=device)

def callback():
    global current_mode, previous_mode
    global door_position, door_angle, drawer_position
    global translation_x, translation_z, rotation_y_planar
    global rotation_angle_x, rotation_angle_y, rotation_angle_z
    global screw_angle
    global prev_points_for_deviation
    global prismatic_door_points, revolute_door_points, drawer_points
    global mouse_points, joint_points, cap_points
    global current_joint_info_dict, current_best_joint
    global current_scores_info
    global device

    global noise_sigma
    global col_sigma, col_order
    global cop_sigma, cop_order
    global rad_sigma, rad_order
    global zp_sigma,  zp_order
    global prob_sigma, prob_order

    global savgol_window_length, savgol_polyorder

    global planar_dofs, planar_vel
    global ball_dofs,   ball_vel

    # Check for mode change
    if previous_mode is not None and previous_mode != current_mode:
        clear_data_for_mode(previous_mode)
        remove_joint_visual()
        prev_points_for_deviation = None
        restore_original_shape(current_mode)
    previous_mode = current_mode

    # Mode selection combo
    changed = psim.BeginCombo("Object Mode", current_mode)
    if changed:
        for mode in modes:
            _, selected = psim.Selectable(mode, current_mode == mode)
            if selected and mode != current_mode:
                current_mode = mode
        psim.EndCombo()

    psim.Separator()

    # Noise & super-Gaussian + SG filter
    if psim.TreeNodeEx("Noise & SuperGaussian + SG Filter", flags=psim.ImGuiTreeNodeFlags_DefaultOpen):
        psim.Columns(2, "mycolumns", False)
        psim.SetColumnWidth(0, 230)

        changed_noise, new_noise_sigma = psim.InputFloat("Noise Sigma", noise_sigma, 0.001)
        if changed_noise:
            noise_sigma = max(0.0, new_noise_sigma)

        changed_col_sigma, new_cs = psim.InputFloat("col_sigma", col_sigma, 0.001)
        if changed_col_sigma:
            col_sigma = max(1e-6, new_cs)
        changed_col_order, new_co = psim.InputFloat("col_order", col_order, 0.1)
        if changed_col_order:
            col_order = max(0.1, new_co)

        changed_cop_sigma, new_cops = psim.InputFloat("cop_sigma", cop_sigma, 0.001)
        if changed_cop_sigma:
            cop_sigma = max(1e-6, new_cops)
        changed_cop_order, new_copo = psim.InputFloat("cop_order", cop_order, 0.1)
        if changed_cop_order:
            cop_order = max(0.1, new_copo)

        changed_rad_sigma, new_rs = psim.InputFloat("rad_sigma", rad_sigma, 0.001)
        if changed_rad_sigma:
            rad_sigma = max(1e-6, new_rs)
        changed_rad_order, new_ro = psim.InputFloat("rad_order", rad_order, 0.1)
        if changed_rad_order:
            rad_order = max(0.1, new_ro)

        changed_zp_sigma, new_zs = psim.InputFloat("zp_sigma", zp_sigma, 0.001)
        if changed_zp_sigma:
            zp_sigma = max(1e-6, new_zs)
        changed_zp_order, new_zo = psim.InputFloat("zp_order", zp_order, 0.1)
        if changed_zp_order:
            zp_order = max(0.1, new_zo)

        changed_prob_sigma, new_ps = psim.InputFloat("prob_sigma", prob_sigma, 0.001)
        if changed_prob_sigma:
            prob_sigma = max(1e-6, new_ps)
        changed_prob_order, new_po = psim.InputFloat("prob_order", prob_order, 0.1)
        if changed_prob_order:
            prob_order = max(0.1, new_po)

        changed_sg_win, new_sg_win = psim.InputInt("SG Window", savgol_window_length, 1)
        if changed_sg_win:
            savgol_window_length = max(3, new_sg_win)
        changed_sg_poly, new_sg_poly = psim.InputInt("SG PolyOrder", savgol_polyorder, 1)
        if changed_sg_poly:
            savgol_polyorder = max(1, new_sg_poly)

        psim.NextColumn()
        psim.TextWrapped(
            
            "1) Noise Sigma: Add Gaussian noise to the point cloud.\n"  
            "2) col/cop/rad/zp/prob_*: Super-Gaussian parameters.\n"  
            "3) SG Window & PolyOrder: Savitzky-Golay filter parameters.\n"             
            "4) Press R to move the planar object randomly; press B to move the ball joint randomly. \n"
            "5) Note: Window size must be smaller than the frame count.\n"
        )
        psim.Columns(1)
        psim.TreePop()

    psim.Separator()

    # Controls for each mode
    if current_mode == "Prismatic Door":
        psim.TextUnformatted(f"Door Position: {door_position:.2f}")
        changed, new_position = psim.SliderFloat("Door Position", door_position, 0.0, door_max_displacement)
        if changed:
            door_position = new_position
            prismatic_door_points = original_prismatic_door_points.copy()
            prismatic_door_points = translate_points(prismatic_door_points, door_position, door_axis)
            if noise_sigma > 1e-6:
                prismatic_door_points += np.random.normal(0, noise_sigma, prismatic_door_points.shape)
            ps_prismatic_door.update_point_positions(prismatic_door_points)
            store_point_cloud(prismatic_door_points, current_mode)
            highlight_max_points(ps_prismatic_door, prismatic_door_points, prev_points_for_deviation, current_mode)
            compute_point_weights_if_needed(current_mode)
            prev_points_for_deviation = prismatic_door_points.copy()

    elif current_mode == "Revolute Door":
        psim.TextUnformatted(f"Door Angle (deg): {np.degrees(door_angle):.2f}")
        changed, new_angle = psim.SliderAngle("Door Angle", door_angle, v_degrees_min=-90, v_degrees_max=90)
        if changed:
            door_angle = new_angle
            revolute_door_points = original_revolute_door_points.copy()
            revolute_door_points = rotate_points(revolute_door_points, door_angle, door_hinge_axis, door_hinge_position)
            if noise_sigma > 1e-6:
                revolute_door_points += np.random.normal(0, noise_sigma, revolute_door_points.shape)
            ps_revolute_door.update_point_positions(revolute_door_points)
            store_point_cloud(revolute_door_points, current_mode)
            highlight_max_points(ps_revolute_door, revolute_door_points, prev_points_for_deviation, current_mode)
            compute_point_weights_if_needed(current_mode)
            prev_points_for_deviation = revolute_door_points.copy()

    elif current_mode == "Drawer":
        psim.TextUnformatted(f"Drawer Position: {drawer_position:.2f}")
        changed, new_position = psim.SliderFloat("Drawer Pos", drawer_position, 0.0, drawer_max_displacement)
        if changed:
            drawer_position = new_position
            drawer_points = original_drawer_points.copy()
            drawer_points = translate_points(drawer_points, drawer_position, drawer_axis)
            if noise_sigma > 1e-6:
                drawer_points += np.random.normal(0, noise_sigma, drawer_points.shape)
            ps_drawer.update_point_positions(drawer_points)
            store_point_cloud(drawer_points, current_mode)
            highlight_max_points(ps_drawer, drawer_points, prev_points_for_deviation, current_mode)
            compute_point_weights_if_needed(current_mode)
            prev_points_for_deviation = drawer_points.copy()

    elif current_mode == "Planar Mouse":
        changed_x, translation_x_new = psim.SliderFloat("Translation X", translation_x, -2.0, 2.0)
        changed_z, translation_z_new = psim.SliderFloat("Translation Z", translation_z, -2.0, 2.0)
        changed_r, rotation_y_new = psim.SliderFloat("Rotation Y", rotation_y_planar, -np.pi, np.pi)
        if changed_x or changed_z or changed_r:
            translation_x = translation_x_new
            translation_z = translation_z_new
            rotation_y_planar = rotation_y_new

            mouse_points = original_mouse_points.copy()
            mouse_points = apply_planar_motion(mouse_points, translation_x, translation_z)
            mouse_points = rotate_points_y(mouse_points, rotation_y_planar, [0.0, 0.0, 0.0])
            if noise_sigma > 1e-6:
                mouse_points += np.random.normal(0, noise_sigma, mouse_points.shape)
            ps_mouse.update_point_positions(mouse_points)
            store_point_cloud(mouse_points, current_mode)
            highlight_max_points(ps_mouse, mouse_points, prev_points_for_deviation, current_mode)
            compute_point_weights_if_needed(current_mode)
            prev_points_for_deviation = mouse_points.copy()

    elif current_mode == "Ball Joint":
        changed_x, rotation_angle_x_new = psim.SliderFloat("Rotation X", rotation_angle_x, -np.pi, np.pi)
        changed_y, rotation_angle_y_new = psim.SliderFloat("Rotation Y", rotation_angle_y, -np.pi, np.pi)
        changed_z, rotation_angle_z_new = psim.SliderFloat("Rotation Z", rotation_angle_z, -np.pi, np.pi)
        if changed_x or changed_y or changed_z:
            rotation_angle_x = rotation_angle_x_new
            rotation_angle_y = rotation_angle_y_new
            rotation_angle_z = rotation_angle_z_new

            joint_points = original_joint_points.copy()
            joint_points = rotate_points_xyz(joint_points, rotation_angle_x, rotation_angle_y, rotation_angle_z, joint_center)
            if noise_sigma > 1e-6:
                joint_points += np.random.normal(0, noise_sigma, joint_points.shape)
            ps_joint.update_point_positions(joint_points)
            store_point_cloud(joint_points, current_mode)
            highlight_max_points(ps_joint, joint_points, prev_points_for_deviation, current_mode)
            compute_point_weights_if_needed(current_mode)
            prev_points_for_deviation = joint_points.copy()

    elif current_mode == "Screw Joint":
        changed, screw_angle_new = psim.SliderFloat("Screw Angle", screw_angle, -2*np.pi, 2*np.pi)
        if changed:
            screw_angle = screw_angle_new
            cap_points = original_cap_points.copy()
            cap_points = apply_screw_motion(cap_points, screw_angle, screw_axis, origin, screw_pitch)
            if noise_sigma > 1e-6:
                cap_points += np.random.normal(0, noise_sigma, cap_points.shape)
            ps_cap.update_point_positions(cap_points)
            store_point_cloud(cap_points, current_mode)
            highlight_max_points(ps_cap, cap_points, prev_points_for_deviation, current_mode)
            compute_point_weights_if_needed(current_mode)
            prev_points_for_deviation = cap_points.copy()

    # Key-based random motions for planar (R) and ball (B)
    key_r = psim.GetKeyIndex(psim.ImGuiKey_R)
    key_b = psim.GetKeyIndex(psim.ImGuiKey_B)
    acc_scale_r = 0.7
    acc_scale_b = 0.7
    damp_factor  = 0.4

    if psim.IsKeyDown(key_r) and (current_mode == "Planar Mouse"):
        acc_r = (torch.rand(3, device=device) - 0.5) * 2.0
        acc_r *= acc_scale_r
        planar_vel += acc_r
        planar_vel *= damp_factor
        planar_dofs += planar_vel
        translation_x = planar_dofs[0].item()
        translation_z = planar_dofs[1].item()
        rotation_y_planar = planar_dofs[2].item()

        mouse_points = original_mouse_points.copy()
        mouse_points = apply_planar_motion(mouse_points, translation_x, translation_z)
        mouse_points = rotate_points_y(mouse_points, rotation_y_planar, [0.,0.,0.])
        if noise_sigma > 1e-6:
            mouse_points += np.random.normal(0, noise_sigma, mouse_points.shape)
        ps_mouse.update_point_positions(mouse_points)
        store_point_cloud(mouse_points, current_mode)
        highlight_max_points(ps_mouse, mouse_points, prev_points_for_deviation, current_mode)
        prev_points_for_deviation = mouse_points.copy()

    if psim.IsKeyDown(key_b) and (current_mode == "Ball Joint"):
        acc_b = (torch.rand(3, device=device) - 0.5) * 2.0
        acc_b *= acc_scale_b
        ball_vel += acc_b
        ball_vel *= damp_factor
        ball_dofs += ball_vel
        rotation_angle_x = ball_dofs[0].item()
        rotation_angle_y = ball_dofs[1].item()
        rotation_angle_z = ball_dofs[2].item()

        joint_points = original_joint_points.copy()
        joint_points = rotate_points_xyz(joint_points, rotation_angle_x, rotation_angle_y, rotation_angle_z, joint_center)
        if noise_sigma > 1e-6:
            joint_points += np.random.normal(0, noise_sigma, joint_points.shape)
        ps_joint.update_point_positions(joint_points)
        store_point_cloud(joint_points, current_mode)
        highlight_max_points(ps_joint, joint_points, prev_points_for_deviation, current_mode)
        prev_points_for_deviation = joint_points.copy()

    # Classification
    key = current_mode.replace(" ", "_")
    all_frames = point_cloud_history.get(key, None)
    if all_frames is not None and len(all_frames) >= 2:
        all_points_history = np.stack(all_frames, axis=0)

        param_dict, best_type, scores_info = compute_joint_info_all_types(
            all_points_history,
            col_sigma=col_sigma, col_order=col_order,
            cop_sigma=cop_sigma, cop_order=cop_order,
            rad_sigma=rad_sigma, rad_order=rad_order,
            zp_sigma=zp_sigma,   zp_order=zp_order,
            prob_sigma=prob_sigma, prob_order=prob_order,
            use_savgol=True,
            savgol_window=savgol_window_length,
            savgol_poly=savgol_polyorder
        )
        global current_joint_info_dict, current_best_joint, current_scores_info
        current_joint_info_dict = param_dict
        current_best_joint = best_type
        current_scores_info = scores_info

        if scores_info is not None:
            col_m = scores_info["basic_score_avg"]["col_mean"]
            cop_m = scores_info["basic_score_avg"]["cop_mean"]
            rad_m = scores_info["basic_score_avg"]["rad_mean"]
            zp_m  = scores_info["basic_score_avg"]["zp_mean"]
            psim.TextUnformatted("=== Basic Scores ===")
            psim.TextUnformatted(f"  Collinearity = {col_m:.3f}")
            psim.TextUnformatted(f"  Coplanar     = {cop_m:.3f}")
            psim.TextUnformatted(f"  RadiusConsis = {rad_m:.3f}")
            psim.TextUnformatted(f"  ZeroPitch    = {zp_m:.3f}")

            joint_probs = scores_info["joint_probs"]
            psim.TextUnformatted("=== Joint Probability ===")
            psim.TextUnformatted(f"Prismatic = {joint_probs['prismatic']:.4f}")
            psim.TextUnformatted(f"Planar    = {joint_probs['planar']:.4f}")
            psim.TextUnformatted(f"Revolute  = {joint_probs['revolute']:.4f}")
            psim.TextUnformatted(f"Screw     = {joint_probs['screw']:.4f}")
            psim.TextUnformatted(f"Ball      = {joint_probs['ball']:.4f}")

            psim.TextUnformatted(f"Best Joint Type: {best_type}")

        if best_type in param_dict:
            jinfo = param_dict[best_type]
            if best_type == "planar":
                n_ = jinfo["normal"]
                lim = jinfo["motion_limit"]
                psim.TextUnformatted(f"  Normal=({n_[0]:.2f}, {n_[1]:.2f}, {n_[2]:.2f})")
                psim.TextUnformatted(f"  MotionLimit=({lim[0]:.2f}, {lim[1]:.2f})")
            elif best_type == "ball":
                c_ = jinfo["center"]
                lim = jinfo["motion_limit"]
                psim.TextUnformatted(f"  Center=({c_[0]:.2f}, {c_[1]:.2f}, {c_[2]:.2f})")
                psim.TextUnformatted(f"  MotionLimit=Rx:{lim[0]:.2f}, Ry:{lim[1]:.2f}, Rz:{lim[2]:.2f}")
            elif best_type == "screw":
                a_ = jinfo["axis"]
                o_ = jinfo["origin"]
                p_ = jinfo["pitch"]
                lim = jinfo["motion_limit"]
                psim.TextUnformatted(f"  Axis=({a_[0]:.2f}, {a_[1]:.2f}, {a_[2]:.2f}), pitch={p_:.3f}")
                psim.TextUnformatted(f"  Origin=({o_[0]:.2f}, {o_[1]:.2f}, {o_[2]:.2f})")
                psim.TextUnformatted(f"  MotionLimit=({lim[0]:.2f} rad, {lim[1]:.2f} rad)")
            elif best_type == "prismatic":
                a_ = jinfo["axis"]
                lim = jinfo["motion_limit"]
                psim.TextUnformatted(f"  Axis=({a_[0]:.2f}, {a_[1]:.2f}, {a_[2]:.2f})")
                psim.TextUnformatted(f"  MotionLimit=({lim[0]:.2f}, {lim[1]:.2f})")
            elif best_type == "revolute":
                a_ = jinfo["axis"]
                o_ = jinfo["origin"]
                lim = jinfo["motion_limit"]
                psim.TextUnformatted(f"  Axis=({a_[0]:.2f}, {a_[1]:.2f}, {a_[2]:.2f})")
                psim.TextUnformatted(f"  Origin=({o_[0]:.2f}, {o_[1]:.2f}, {o_[2]:.2f})")
                psim.TextUnformatted(f"  MotionLimit=({lim[0]:.2f} rad, {lim[1]:.2f} rad)")
            show_joint_visual(best_type, jinfo)
        else:
            psim.TextUnformatted(f"Best Type: {best_type} (no param)")
    else:
        psim.TextUnformatted("Not enough frames to do joint classification.")

    if psim.Button("Save .npy for current joint type"):
        save_all_to_npy(current_mode)


ps.set_user_callback(callback)
restore_original_shape(current_mode)
ps.show()
