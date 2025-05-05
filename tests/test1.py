import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import os
import math
import torch
from scipy.signal import savgol_filter

# DearPyGUI
import dearpygui.dearpygui as dpg
import threading

# =============== Load Real Drawer Data (120 frames) ===============
file_path = "/home/rui/projects/kitchen_drawer/exp2_pro_kvil/pro_kvil/data/demo1/obj/xyz_filtered.pt"
real_data_dict = torch.load(file_path, weights_only=False)
drawer_points_tensor = real_data_dict.data["drawer"]
real_drawer_data_np = np.array(drawer_points_tensor)  # shape (120, N, 3), 120 frames
# ===============================================================

ps.init()

# =====================================================================
# motion_sim.py (core geometry and motion functions)
# =====================================================================

def point_line_distance(p, line_origin, line_dir):
    """Compute the distance from a 3D point p to a line defined by (line_origin, line_dir).
       line_dir should be a unit vector."""
    v = p - line_origin
    cross_ = np.cross(v, line_dir)
    return np.linalg.norm(cross_)

def translate_points(points, displacement, axis):
    """Translate points by a given displacement along a specified axis."""
    return points + displacement * axis

def rotate_points(points, angle, axis, origin):
    """Rotate points around a given origin by an angle around a specified axis (which need not be unit)."""
    axis = axis / np.linalg.norm(axis)
    points = points - origin
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    R = np.array([
        [t * axis[0] * axis[0] + c,         t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
        [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c,         t * axis[1] * axis[2] - s * axis[0]],
        [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c       ]
    ])
    rotated_points = points @ R.T
    rotated_points += origin
    return rotated_points

def rotate_points_y(points, angle, center):
    """Rotate points around the y-axis."""
    points = points - center
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([
        [c, 0., s],
        [0., 1., 0.],
        [-s, 0., c]
    ])
    rotated_points = points @ R.T
    rotated_points += center
    return rotated_points

def rotate_points_xyz(points, angle_x, angle_y, angle_z, center):
    """Rotate points around X, then Y, then Z axes in sequence."""
    points = points - center
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z),  np.cos(angle_z), 0],
        [0,                0,               1]
    ])
    rotated = points @ Rx.T
    rotated = rotated @ Ry.T
    rotated = rotated @ Rz.T
    rotated += center
    return rotated

def apply_screw_motion(points, angle, axis, origin, pitch):
    """Apply a screw motion: rotate around an axis by angle, then translate along the axis proportionally."""
    axis = axis / np.linalg.norm(axis)
    points = points - origin
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    R = np.array([
        [t * axis[0] * axis[0] + c,         t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
        [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c,         t * axis[1] * axis[2] - s * axis[0]],
        [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c       ]
    ])
    rotated_points = points @ R.T
    translation = (angle / (2 * np.pi)) * pitch * axis
    transformed_points = rotated_points + translation
    transformed_points += origin
    return transformed_points

def generate_sphere(center, radius, num_points):
    """Randomly sample points inside a sphere."""
    phi = np.random.rand(num_points) * 2 * np.pi
    costheta = 2 * np.random.rand(num_points) - 1
    theta = np.arccos(costheta)
    r = radius * (np.random.rand(num_points) ** (1 / 3))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return center + np.vstack([x, y, z]).T

def generate_cylinder(radius, height, num_points=500):
    """Randomly sample points in a cylinder."""
    zs = np.random.rand(num_points) * height - height / 2
    phi = np.random.rand(num_points) * 2 * np.pi
    rs = radius * np.sqrt(np.random.rand(num_points))
    xs = rs * np.cos(phi)
    ys = zs
    zs = rs * np.sin(phi)
    return np.vstack([xs, ys, zs]).T

def generate_ball_joint_points(center, sphere_radius, rod_length, rod_radius,
                               num_points_sphere=250, num_points_rod=250):
    """Generate a ball joint: a sphere + a cylinder-like rod."""
    sphere_pts = generate_sphere(center, sphere_radius, num_points_sphere)
    rod_pts = generate_cylinder(rod_radius, rod_length, num_points_rod)
    rod_pts[:, 1] += center[1]
    return np.concatenate([sphere_pts, rod_pts], axis=0)

def generate_hollow_cylinder(radius, height, thickness,
                             num_points=500, cap_position="top", cap_points_ratio=0.2):
    """Generate a hollow cylinder with one end capped."""
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

# =====================================================================
# joint_estimation.py (joint estimation methods)
# =====================================================================

def super_gaussian(x, sigma, order):
    """Super-Gaussian function: exp(-(|x|/sigma)^order)."""
    return torch.exp(- (torch.abs(x) / sigma) ** order)

def normalize_vector_torch(v, eps=1e-3):
    """Normalize 3D vectors in a PyTorch tensor; if norm < eps, return zero vector."""
    norm_v = torch.norm(v, dim=-1)
    mask = (norm_v > eps)
    out = torch.zeros_like(v)
    out[mask] = v[mask] / norm_v[mask].unsqueeze(-1)
    return out

def estimate_rotation_matrix_batch(pcd_src: torch.Tensor, pcd_tar: torch.Tensor):
    """Estimate rotation matrices via SVD in batch. Input shape (B,N,3), output (B,3,3)."""
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
    SE(3) log map in batch. Input: (B,4,4), Output: (B,6) => [translation, rotation].
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
    """Find nearest neighbors for each frame in a batch: (B,P,3) -> (B,P,k)."""
    dist = torch.cdist(pcd_batch, pcd_batch, p=2.0)
    neighbor_indices = torch.topk(dist, k=num_neighbor_pts, dim=-1, largest=False).indices
    return neighbor_indices

def multi_frame_rigid_fit(all_points_history: torch.Tensor, center_idx: int, window_radius: int):
    """
    Multi-frame rigid fit around [center_idx-window_radius, center_idx+window_radius].
    Returns a 4x4 transform from the center frame to best fit.
    """
    T, N, _ = all_points_history.shape
    device = all_points_history.device
    i_min = max(0, center_idx - window_radius)
    i_max = min(T - 1, center_idx + window_radius)

    ref_pts = all_points_history[center_idx]
    src_list = []
    tgt_list = []
    for idx in range(i_min, i_max + 1):
        if idx == center_idx:
            continue
        cur_pts = all_points_history[idx]
        src_list.append(ref_pts)
        tgt_list.append(cur_pts)
    if not src_list:
        return torch.eye(4, device=device)
    src_big = torch.cat(src_list, dim=0)
    tgt_big = torch.cat(tgt_list, dim=0)

    src_mean = src_big.mean(dim=0)
    tgt_mean = tgt_big.mean(dim=0)
    src_centered = src_big - src_mean
    tgt_centered = tgt_big - tgt_mean

    H = torch.einsum('ni,nj->ij', src_centered, tgt_centered)
    U, S, Vt = torch.linalg.svd(H)
    R_ = Vt.T @ U.T
    if torch.det(R_) < 0:
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T
    t_ = tgt_mean - R_ @ src_mean
    Tmat = torch.eye(4, device=device)
    Tmat[:3, :3] = R_
    Tmat[:3, 3] = t_
    return Tmat

def compute_motion_salience_batch_neighborhood(all_points_history, device='cuda', k=10):
    """
    Compute motion salience using neighbor-based average displacements.
    all_points_history: (T,N,3).
    """
    pts = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    T, N, _ = pts.shape
    if T < 2:
        return torch.zeros(N, device=device)
    B = T - 1
    pts_t = pts[:-1]
    pts_tp1 = pts[1:]
    neighbor_idx = find_neighbors_batch(pts_t, k)
    sum_disp = torch.zeros(N, device=device)
    for b in range(B):
        p_t = pts_t[b]
        p_tp1 = pts_tp1[b]
        nb_idx = neighbor_idx[b]
        p_t_nb = p_t[nb_idx]
        p_tp1_nb = p_tp1[nb_idx]
        disp = p_tp1_nb - p_t_nb
        disp_mean = disp.mean(dim=1)
        mag = torch.norm(disp_mean, dim=1)
        sum_disp += mag
    return sum_disp

def compute_motion_salience_batch(all_points_history, neighbor_k=400, device='cuda'):
    """Wrapper for neighbor-based motion salience with adjustable k."""
    pts = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    T, N, _ = pts.shape
    if T < 2:
        return torch.zeros(N, device=device)
    return compute_motion_salience_batch_neighborhood(all_points_history, device=device, k=neighbor_k)

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
    Compute linear and angular velocity for (T,N,3).
    Returns v_arr, w_arr => (T-1, N, 3).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not isinstance(all_points_history, torch.Tensor):
        all_points_history = torch.tensor(all_points_history, dtype=torch.float32, device=device)
    T, N, _ = all_points_history.shape
    if T < 2:
        return None, None

    if use_multi_frame:
        # Multi-frame rigid fit
        v_list = []
        w_list = []
        for t in range(T - 1):
            center_idx = t + 1
            Tmat = multi_frame_rigid_fit(all_points_history, center_idx, window_radius)
            Tmat_batch = Tmat.unsqueeze(0)
            se3_logs = se3_log_map_batch(Tmat_batch)
            se3_v = se3_logs[0, :3] / dt
            se3_w = se3_logs[0, 3:] / dt
            v_list.append(se3_v.unsqueeze(0).repeat(all_points_history.shape[1], 1))
            w_list.append(se3_w.unsqueeze(0).repeat(all_points_history.shape[1], 1))
        v_arr = torch.stack(v_list, dim=0).cpu().numpy()
        w_arr = torch.stack(w_list, dim=0).cpu().numpy()
    else:
        # Neighbor-based estimation
        pts_prev = all_points_history[:-1]
        pts_curr = all_points_history[1:]
        B = T - 1
        neighbor_idx_prev = find_neighbors_batch(pts_prev, num_neighbors)
        neighbor_idx_curr = find_neighbors_batch(pts_curr, num_neighbors)
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
        src_2d = src_batch.reshape(B * N, K, 3)
        tar_2d = tar_batch.reshape(B * N, K, 3)

        R_2d = estimate_rotation_matrix_batch(src_2d, tar_2d)
        c1_2d = src_2d.mean(dim=1)
        c2_2d = tar_2d.mean(dim=1)
        delta_p_2d = c2_2d - c1_2d

        eye_4 = torch.eye(4, device=device).unsqueeze(0).expand(B * N, -1, -1).clone()
        eye_4[:, :3, :3] = R_2d
        eye_4[:, :3, 3] = delta_p_2d
        transform_matrices_2d = eye_4
        se3_logs_2d = se3_log_map_batch(transform_matrices_2d)
        v_2d = se3_logs_2d[:, :3] / dt
        w_2d = se3_logs_2d[:, 3:] / dt

        v_arr = v_2d.reshape(B, N, 3).cpu().numpy()
        w_arr = w_2d.reshape(B, N, 3).cpu().numpy()

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

def compute_basic_scores(
    v_history, w_history, device='cuda',
    col_sigma=0.2, col_order=4.0,
    cop_sigma=0.2, cop_order=4.0,
    rad_sigma=0.2, rad_order=4.0,
    zp_sigma=0.2, zp_order=4.0
):
    """
    Compute the four main scores:
      - col: Collinearity
      - cop: Coplanarity
      - rad: Radius consistency
      - zp: Zero pitch
    Returns (N,) for each score.
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

    col_score = torch.ones(N, device=device)
    cop_score = torch.ones(N, device=device)
    if Tm1 >= 2:
        v_unit = normalize_vector_torch(v_clean)
        v_unit_all = v_unit.permute(1, 0, 2).contiguous()
        if Tm1 >= 3:
            U, S, _ = torch.linalg.svd(v_unit_all, full_matrices=False)
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
            U, S_, _ = torch.linalg.svd(v_unit_all, full_matrices=False)
            s1 = S_[:, 0]
            s2 = S_[:, 1] if S_.size(1) > 1 else torch.zeros_like(s1)
            ratio_col = torch.zeros_like(s1)
            mask_svd = (s1 > 1e-6)
            ratio_col[mask_svd] = s2[mask_svd] / s1[mask_svd]
            col_score = super_gaussian(ratio_col, col_sigma, col_order)

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

    zp_score = torch.ones(N, device=device)
    if Tm1 > 0:
        v_u = normalize_vector_torch(v_clean)
        w_u = normalize_vector_torch(w_clean)
        dot_val = torch.sum(v_u * w_u, dim=2).abs()
        mean_dot = torch.mean(dot_val, dim=0)
        zp_score = super_gaussian(mean_dot, zp_sigma, zp_order)

    return col_score, cop_score, rad_score, zp_score

def compute_joint_probability_new(col, cop, rad, zp, joint_type="prismatic", prob_sigma=0.1, prob_order=4.0):
    """
    Compute joint probability (0~1) based on the four fundamental scores.
    """
    if joint_type == "prismatic":
        # Prismatic: col->1, cop->0, rad->0, zp->1
        e = ((col - 1) ** 2 + (cop - 1) ** 2 + (rad - 0) ** 2 + (zp - 1) ** 2) / 4
        return super_gaussian(e, prob_sigma, prob_order)
    elif joint_type == "planar":
        # Planar: col->0, cop->1, rad->0, zp->1
        e = (col ** 2 + (cop - 1) ** 2 + rad ** 2 + (zp - 1) ** 2) / 4
        return super_gaussian(e, prob_sigma, prob_order)
    elif joint_type == "revolute":
        # Revolute: col->0, cop->1, rad->1, zp->1
        e = (col ** 2 + (cop - 1) ** 2 + (rad - 1) ** 2 + (zp - 1) ** 2) / 4
        return super_gaussian(e, prob_sigma, prob_order)
    elif joint_type == "screw":
        # Screw: col->0, cop->0, rad->1, zp->0
        e = (col ** 2 + cop ** 2 + (rad - 1) ** 2 + zp ** 2) / 4
        return super_gaussian(e, prob_sigma, prob_order)
    elif joint_type == "ball":
        # Ball: col->0, cop->0, rad->1, zp->1
        e = (col ** 2 + cop ** 2 + (rad - 1) ** 2 + (zp - 1) ** 2) / 4
        return super_gaussian(e, prob_sigma, prob_order)
    return torch.zeros_like(col)

planar_normal_reference = None
planar_axis1_reference = None
planar_axis2_reference = None
plane_is_fixed = False
screw_axis_reference = None
prismatic_axis_reference = None
revolute_axis_reference = None

def compute_planar_info(all_points_history, v_history, omega_history, w_i, device='cuda'):

    global planar_normal_reference, planar_axis1_reference, planar_axis2_reference
    global plane_is_fixed

    all_points_history = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    w_i = torch.as_tensor(w_i, dtype=torch.float32, device=device)
    T, N = all_points_history.shape[0], all_points_history.shape[1]
    if T < 3:
        return {"normal": np.array([0., 0., 0.]), "motion_limit": (0., 0.)}

    if not plane_is_fixed:
        # 第一次计算时，做加权PCA估计平面
        pts_torch = all_points_history.reshape(-1, 3)
        weights_expanded = w_i.repeat(T)  # 这里将每帧同一个 w_i 用作权重
        w_sum = weights_expanded.sum()
        sum_wx = torch.einsum('b,bj->j', weights_expanded, pts_torch)
        weighted_mean = sum_wx / (w_sum + 1e-9)
        centered = pts_torch - weighted_mean
        wc_ = centered * weights_expanded.unsqueeze(-1)
        cov_mat = torch.einsum('bi,bj->ij', wc_, centered) / (w_sum + 1e-9)
        cov_mat += 1e-9 * torch.eye(3, device=device)
        eigvals, eigvecs = torch.linalg.eigh(cov_mat)
        axis1_ = eigvecs[:, 2]
        axis2_ = eigvecs[:, 1]
        axis1_ = normalize_vector_torch(axis1_.unsqueeze(0))[0]
        axis2_ = normalize_vector_torch(axis2_.unsqueeze(0))[0]
        normal_ = torch.cross(axis1_, axis2_, dim=0)
        normal_ = normalize_vector_torch(normal_.unsqueeze(0))[0]
        planar_axis1_reference = axis1_.clone()
        planar_axis2_reference = axis2_.clone()
        planar_normal_reference = normal_.clone()
        plane_is_fixed = True
    else:
        axis1_ = planar_axis1_reference.clone()
        axis2_ = planar_axis2_reference.clone()
        normal_ = planar_normal_reference.clone()

    if T < 2:
        motion_limit = (0., 0.)
    else:
        pt0_0 = all_points_history[0, 0]
        pts = all_points_history[:, 0] - pt0_0
        proj1 = torch.matmul(pts, axis1_)
        proj2 = torch.matmul(pts, axis2_)
        max_p1 = proj1.max().item()
        max_p2 = proj2.max().item()
        motion_limit = (float(max_p1), float(max_p2))

    return {
        "normal": normal_.cpu().numpy(),
        "motion_limit": motion_limit
    }


def compute_ball_info(all_points_history, v_history, omega_history, w_i, device='cuda'):
    """
    Estimate rough center of rotation for a ball joint, plus a motion limit (approximate).
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

    # Rough maximum angle
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
    Estimate an approximate screw axis and pitch.
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

    EPS_W = 1e-3
    v_u = normalize_vector_torch(v_history)
    w_u = normalize_vector_torch(omega_history)

    v_norm = torch.norm(v_history, dim=2)
    w_norm = torch.norm(omega_history, dim=2)
    s_mat = torch.sum(v_history * w_u, dim=2)
    ratio = torch.zeros_like(s_mat)
    mask_w_ = (w_norm > EPS_W)
    ratio[mask_w_] = s_mat[mask_w_] / (v_norm[mask_w_] + 1e-9)
    pitch_all = torch.acos(torch.clamp(ratio, -1., 1.))
    pitch_sin = torch.sin(pitch_all)
    r_ = torch.zeros_like(v_norm)
    r_[mask_w_] = (v_norm[mask_w_] * pitch_sin[mask_w_]) / (w_norm[mask_w_] + 1e-9)
    dir_ = -torch.cross(v_u, w_u, dim=-1)
    dir_ = normalize_vector_torch(dir_)
    c_pos = all_points_history[:-1] + dir_ * r_.unsqueeze(-1)
    pitch_each = torch.mean(pitch_all, dim=0)
    pitch_sum = float(torch.sum(w_i * pitch_each).item())

    w_u_flat = w_u.reshape(-1, 3)
    T_actual = w_u.shape[0]
    w_i_flat = w_i.unsqueeze(0).expand(T_actual, N).reshape(-1)
    W = torch.sum(w_i_flat) + 1e-9
    weighted_mean = (w_u_flat * w_i_flat.unsqueeze(-1)).sum(dim=0) / W
    w_u_centered = w_u_flat - weighted_mean
    w_u_centered_weighted = w_u_centered * torch.sqrt(w_i_flat.unsqueeze(-1))
    cov_mat = (w_u_centered_weighted.transpose(0, 1) @ w_u_centered_weighted) / W
    cov_mat += 1e-9 * torch.eye(3, device=device)
    eigvals, eigvecs = torch.linalg.eigh(cov_mat)
    idx_max = torch.argmax(eigvals)
    axis_raw = eigvecs[:, idx_max]
    axis_sum = normalize_vector_torch(axis_raw.unsqueeze(0))[0]
    if screw_axis_reference is not None:
        dotval = torch.dot(axis_sum, screw_axis_reference)
        if dotval < 0:
            axis_sum = -axis_sum
    screw_axis_reference = axis_sum.clone()

    c_pos_flat = c_pos.reshape(-1, 3)
    if c_pos_flat.shape[0] == 0:
        return {
            "axis": axis_sum.cpu().numpy(),
            "origin": np.array([0., 0., 0.]),
            "pitch": pitch_sum,
            "motion_limit": (0., 0.)
        }
    median_cp = c_pos_flat.median(dim=0).values
    dev = torch.norm(c_pos_flat - median_cp, dim=1)
    med_dev = dev.median()
    if med_dev < 1e-9:
        origin_pts = c_pos_flat
    else:
        mask_in = (dev < 3.0 * med_dev)
        origin_pts = c_pos_flat[mask_in]
    if origin_pts.shape[0] == 0:
        origin_sum = median_cp
    else:
        origin_sum = origin_pts.mean(dim=0)

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
    Estimate the sliding axis for a prismatic joint. Also compute a rough motion range.
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
    pos_history_n_tc = all_points_history.permute(1, 0, 2).contiguous()
    mean_pos = pos_history_n_tc.mean(dim=1, keepdim=True)
    centered = pos_history_n_tc - mean_pos
    covs = torch.einsum('ntm,ntk->nmk', centered, centered)
    B_n = covs.shape[0]
    epsilon_eye = 1e-9 * torch.eye(3, device=device)
    for i in range(B_n):
        covs[i] += epsilon_eye
    eigvals, eigvecs = torch.linalg.eigh(covs)
    max_vecs = eigvecs[:, :, 2]
    max_vecs = normalize_vector_torch(max_vecs)
    weighted_dir = torch.sum(max_vecs * w_i.unsqueeze(-1), dim=0)
    axis_sum = normalize_vector_torch(weighted_dir.unsqueeze(0))[0]
    if prismatic_axis_reference is None:
        prismatic_axis_reference = axis_sum.clone()
    else:
        dot_val = torch.dot(axis_sum, prismatic_axis_reference)
        if dot_val < 0:
            axis_sum = -axis_sum

    base_pt = all_points_history[0, 0]
    pts = all_points_history[:, 0, :]
    vecs = pts - base_pt.unsqueeze(0)
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
    Estimate the rotation axis, origin, and motion range for a revolute joint.
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

    omega_nbc = omega_history.permute(1, 0, 2).contiguous()
    covs = torch.einsum('ibm,ibn->imn', omega_nbc, omega_nbc)
    covs += 1e-9 * torch.eye(3, device=device)
    eigvals, eigvecs = torch.linalg.eigh(covs)
    max_vecs = eigvecs[:, :, 2]
    max_vecs = normalize_vector_torch(max_vecs)
    revolve_axis = torch.sum(max_vecs * w_i.unsqueeze(-1), dim=0)
    revolve_axis = normalize_vector_torch(revolve_axis.unsqueeze(0))[0]
    if revolute_axis_reference is None:
        revolute_axis_reference = revolve_axis.clone()
    else:
        dot_val = torch.dot(revolve_axis, revolute_axis_reference)
        if dot_val < 0:
            revolve_axis = -revolve_axis

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
    c_pos_nbc = c_pos.permute(1, 0, 2).contiguous()
    c_pos_median = c_pos_nbc.median(dim=1).values
    dev = torch.norm(c_pos_nbc - c_pos_median.unsqueeze(1), dim=2)
    scale = dev.median(dim=1).values + 1e-6
    ratio = dev / scale.unsqueeze(-1)
    w_r = 1.0 / (1.0 + ratio * ratio)
    w_r_3d = w_r.unsqueeze(-1)
    c_pos_weighted = c_pos_nbc * w_r_3d
    sum_pos = c_pos_weighted.sum(dim=1)
    sum_w = w_r.sum(dim=1, keepdim=True) + 1e-6
    origin_each = sum_pos / sum_w
    origin_sum = torch.sum(origin_each * w_i.unsqueeze(-1), dim=0)

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
      Returns (joint_params_dict, best_joint, info_dict)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = all_points_history.shape[0]
    N = all_points_history.shape[1]
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

    ms_torch = compute_motion_salience_batch(all_points_history, neighbor_k=neighbor_k, device=device)
    ms = ms_torch.cpu().numpy()
    sum_ms = ms.sum()
    if sum_ms < 1e-6:
        w_i = np.ones_like(ms) / ms.shape[0]
    else:
        w_i = ms / sum_ms

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

    col, cop, rad, zp = compute_basic_scores(
        v_arr, w_arr, device=device,
        col_sigma=col_sigma, col_order=col_order,
        cop_sigma=cop_sigma, cop_order=cop_order,
        rad_sigma=rad_sigma, rad_order=rad_order,
        zp_sigma=zp_sigma, zp_order=zp_order
    )
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

    prismatic_prob = float(torch.sum(w_t * prismatic_pt).item())
    planar_prob = float(torch.sum(w_t * planar_pt).item())
    revolute_prob = float(torch.sum(w_t * revolute_pt).item())
    screw_prob = float(torch.sum(w_t * screw_pt).item())
    ball_prob = float(torch.sum(w_t * ball_pt).item())

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
        "planar": planar_prob,
        "revolute": revolute_prob,
        "screw": screw_prob,
        "ball": ball_prob
    }

    info_dict = {
        "basic_score_avg": basic_score_avg,
        "joint_probs": ret_probs,
        "v_arr": v_arr,
        "w_arr": w_arr,
        "w_i": w_i
    }
    return ret, best_joint, info_dict

# =====================================================================
# polyscope_viewer.py + additional sub-modes
# =====================================================================

running = False
current_mode = "Prismatic Door"
previous_mode = None
current_best_joint = "Unknown"
current_scores_info = None

modes = [
    "Prismatic Door", "Prismatic Door 2", "Prismatic Door 3",
    "Revolute Door",  "Revolute Door 2", "Revolute Door 3",
    "Planar Mouse",   "Planar Mouse 2",  "Planar Mouse 3",
    "Ball Joint",     "Ball Joint 2",    "Ball Joint 3",
    "Screw Joint",    "Screw Joint 2",   "Screw Joint 3",
    "Real Drawer Data"
]

point_cloud_history = {}
for m in modes:
    point_cloud_history[m.replace(" ", "_")] = []

file_counter = {}
velocity_profile = {m: [] for m in modes}
angular_velocity_profile = {m: [] for m in modes}
col_score_profile = {m: [] for m in modes}
cop_score_profile = {m: [] for m in modes}
rad_score_profile = {m: [] for m in modes}
zp_score_profile = {m: [] for m in modes}
error_profile = {m: [] for m in modes}
joint_prob_profile = {
    m: {
        "prismatic": [],
        "planar": [],
        "revolute": [],
        "screw": [],
        "ball": []
    } for m in modes
}
frame_count_per_mode = {m: 0 for m in modes}
TOTAL_FRAMES_PER_MODE = {}

for m in modes:
    if m == "Real Drawer Data":
        TOTAL_FRAMES_PER_MODE[m] = real_drawer_data_np.shape[0]
    else:
        TOTAL_FRAMES_PER_MODE[m] = 50

noise_sigma = 0.000
col_sigma = 0.2
col_order = 4.0
cop_sigma = 0.2
cop_order = 4.0
rad_sigma = 0.2
rad_order = 4.0
zp_sigma = 0.2
zp_order = 4.0
prob_sigma = 0.2
prob_order = 4.0
neighbor_k = 50
use_savgol_filter = False
savgol_window_length = 10
savgol_polyorder = 2
use_multi_frame_fit = False
multi_frame_radius = 20

output_dir = "exported_pointclouds"
os.makedirs(output_dir, exist_ok=True)

# ========== 1) Prismatic Door ==========
door_width, door_height, door_thickness = 2.0, 3.0, 0.2
num_points = 500

original_prismatic_door_points = np.random.rand(num_points, 3)
original_prismatic_door_points[:, 0] = original_prismatic_door_points[:, 0] * door_width - 0.5 * door_width
original_prismatic_door_points[:, 1] = original_prismatic_door_points[:, 1] * door_height
original_prismatic_door_points[:, 2] = original_prismatic_door_points[:, 2] * door_thickness - 0.5 * door_thickness
prismatic_door_points = original_prismatic_door_points.copy()
ps_prismatic_door = ps.register_point_cloud("Prismatic Door", prismatic_door_points)

original_prismatic_door_points_2 = original_prismatic_door_points.copy() + np.array([1.5, 0., 0.])
prismatic_door_points_2 = original_prismatic_door_points_2.copy()
ps_prismatic_door_2 = ps.register_point_cloud("Prismatic Door 2", prismatic_door_points_2, enabled=False)

original_prismatic_door_points_3 = original_prismatic_door_points.copy() + np.array([-1., 1., 0.])
prismatic_door_points_3 = original_prismatic_door_points_3.copy()
ps_prismatic_door_3 = ps.register_point_cloud("Prismatic Door 3", prismatic_door_points_3, enabled=False)

# ========== 2) Revolute Door ==========
original_revolute_door_points = prismatic_door_points.copy()
revolute_door_points = original_revolute_door_points.copy()
ps_revolute_door = ps.register_point_cloud("Revolute Door", revolute_door_points, enabled=False)
door_hinge_position = np.array([1.0, 1.5, 0.0])
door_hinge_axis = np.array([0.0, 1.0, 0.0])
revolute_door_gt_origin = door_hinge_position
revolute_door_gt_axis = door_hinge_axis
revolute_door_gt_axis_norm = revolute_door_gt_axis / np.linalg.norm(revolute_door_gt_axis)
dists_rev = [point_line_distance(p, revolute_door_gt_origin, revolute_door_gt_axis_norm)
             for p in original_revolute_door_points]
revolute_door_max_dist = max(dists_rev) if len(dists_rev) > 0 else 1.0

original_revolute_door_points_2 = original_revolute_door_points.copy() + np.array([0., 0., -1.])
revolute_door_points_2 = original_revolute_door_points_2.copy()
ps_revolute_door_2 = ps.register_point_cloud("Revolute Door 2", revolute_door_points_2, enabled=False)
door_hinge_position_2 = np.array([0.5, 2.0, -1.0])
door_hinge_axis_2 = np.array([1.0, 0.0, 0.0])
revolute_door_2_gt_origin = door_hinge_position_2
revolute_door_2_gt_axis = door_hinge_axis_2
revolute_door_2_gt_axis_norm = revolute_door_2_gt_axis / np.linalg.norm(revolute_door_2_gt_axis)
dists_rev_2 = [point_line_distance(p, revolute_door_2_gt_origin, revolute_door_2_gt_axis_norm)
               for p in original_revolute_door_points_2]
revolute_door_2_max_dist = max(dists_rev_2) if len(dists_rev_2) > 0 else 1.0

original_revolute_door_points_3 = original_revolute_door_points.copy() + np.array([0., -0.5, 1.0])
revolute_door_points_3 = original_revolute_door_points_3.copy()
ps_revolute_door_3 = ps.register_point_cloud("Revolute Door 3", revolute_door_points_3, enabled=False)
door_hinge_position_3 = np.array([2.0, 1.0, 1.0])
door_hinge_axis_3 = np.array([1.0, 1.0, 0.])
revolute_door_3_gt_origin = door_hinge_position_3
revolute_door_3_gt_axis = door_hinge_axis_3
revolute_door_3_gt_axis_norm = revolute_door_3_gt_axis / np.linalg.norm(revolute_door_3_gt_axis)
dists_rev_3 = [point_line_distance(p, revolute_door_3_gt_origin, revolute_door_3_gt_axis_norm)
               for p in original_revolute_door_points_3]
revolute_door_3_max_dist = max(dists_rev_3) if len(dists_rev_3) > 0 else 1.0

# ========== 3) Planar Mouse ==========
mouse_length, mouse_width, mouse_height = 1.0, 0.6, 0.3
original_mouse_points = np.zeros((num_points, 3))
original_mouse_points[:, 0] = np.random.rand(num_points) * mouse_length - 0.5 * mouse_length
original_mouse_points[:, 2] = np.random.rand(num_points) * mouse_width - 0.5 * mouse_width
original_mouse_points[:, 1] = np.random.rand(num_points) * mouse_height
mouse_points = original_mouse_points.copy()
ps_mouse = ps.register_point_cloud("Planar Mouse", mouse_points, enabled=False)
planar_mouse_gt_normal = np.array([0., 1., 0.])

original_mouse_points_2 = original_mouse_points.copy() + np.array([1., 0., 1.])
mouse_points_2 = original_mouse_points_2.copy()
ps_mouse_2 = ps.register_point_cloud("Planar Mouse 2", mouse_points_2, enabled=False)
planar_mouse_2_gt_normal = np.array([0., 1., 0.])

original_mouse_points_3 = original_mouse_points.copy() + np.array([-1., 0., 1.])
mouse_points_3 = original_mouse_points_3.copy()
ps_mouse_3 = ps.register_point_cloud("Planar Mouse 3", mouse_points_3, enabled=False)
planar_mouse_3_gt_normal = np.array([0., 1., 0.])

# ========== 4) Ball Joint ==========
sphere_radius = 0.3
rod_length = sphere_radius * 10.0
rod_radius = 0.05
def_points = generate_ball_joint_points(np.array([0., 0., 0.]), sphere_radius, rod_length, rod_radius, 250, 250)
original_joint_points = def_points.copy()
joint_points = original_joint_points.copy()
ps_joint = ps.register_point_cloud("Ball Joint", joint_points, enabled=False)
ball_joint_gt_center = np.array([0., 0., 0.])
ball_dists = np.linalg.norm(original_joint_points - ball_joint_gt_center, axis=1)
ball_joint_max_dist = np.max(ball_dists) if len(ball_dists) > 0 else 1.0

def_points_2 = generate_ball_joint_points(np.array([0., 0., 0.]), sphere_radius, rod_length, rod_radius, 250, 250)
original_joint_points_2 = def_points_2.copy()
joint_points_2 = original_joint_points_2.copy()
ps_joint_2 = ps.register_point_cloud("Ball Joint 2", joint_points_2, enabled=False)
ball_joint_2_gt_center = np.array([0., 0., 0.])
ball_dists_2 = np.linalg.norm(original_joint_points_2 - ball_joint_2_gt_center, axis=1)
ball_joint_2_max_dist = np.max(ball_dists_2) if len(ball_dists_2) > 0 else 1.0

def_points_3 = generate_ball_joint_points(np.array([0., 0., 0.]), sphere_radius, rod_length, rod_radius, 250, 250)
original_joint_points_3 = def_points_3.copy()
joint_points_3 = original_joint_points_3.copy()
ps_joint_3 = ps.register_point_cloud("Ball Joint 3", joint_points_3, enabled=False)
ball_joint_3_gt_center = np.array([0., 0., 0.])
ball_dists_3 = np.linalg.norm(original_joint_points_3 - ball_joint_3_gt_center, axis=1)
ball_joint_3_max_dist = np.max(ball_dists_3) if len(ball_dists_3) > 0 else 1.0

# ========== 5) Screw Joint ==========
screw_pitch = 0.5
original_cap_points = generate_hollow_cylinder(
    radius=0.4, height=0.2, thickness=0.05,
    num_points=500, cap_position="top", cap_points_ratio=0.2
)
cap_points = original_cap_points.copy()
ps_cap = ps.register_point_cloud("Screw Joint", cap_points, enabled=False)
screw_gt_axis = np.array([0., 1., 0.])
screw_gt_origin = np.array([0., 0., 0.])
screw_gt_axis_norm = screw_gt_axis / np.linalg.norm(screw_gt_axis)
dists_screw = [point_line_distance(p, screw_gt_origin, screw_gt_axis_norm) for p in original_cap_points]
screw_joint_max_dist = max(dists_screw) if len(dists_screw) > 0 else 1.0

original_cap_points_2 = original_cap_points.copy() + np.array([1., 0., 0.])
cap_points_2 = original_cap_points_2.copy()
ps_cap_2 = ps.register_point_cloud("Screw Joint 2", cap_points_2, enabled=False)
screw_2_gt_axis = np.array([1., 0., 0.])
screw_2_gt_origin = np.array([1., 0., 0.])
screw_2_gt_axis_norm = screw_2_gt_axis / np.linalg.norm(screw_2_gt_axis)
dists_screw_2 = [point_line_distance(p, screw_2_gt_origin, screw_2_gt_axis_norm) for p in original_cap_points_2]
screw_joint_2_max_dist = max(dists_screw_2) if len(dists_screw_2) > 0 else 1.0
screw_pitch_2 = 0.8

original_cap_points_3 = original_cap_points.copy() + np.array([-1., 0., 1.])
cap_points_3 = original_cap_points_3.copy()
ps_cap_3 = ps.register_point_cloud("Screw Joint 3", cap_points_3, enabled=False)
screw_3_gt_axis = np.array([1., 1., 0.])
screw_3_gt_origin = np.array([-1., 0., 1.])
screw_3_gt_axis_norm = screw_3_gt_axis / np.linalg.norm(screw_3_gt_axis)
dists_screw_3 = [point_line_distance(p, screw_3_gt_origin, screw_3_gt_axis_norm) for p in original_cap_points_3]
screw_joint_3_max_dist = max(dists_screw_3) if len(dists_screw_3) > 0 else 1.0
screw_pitch_3 = 0.6

# ========== 6) Real Drawer Data ==========
ps_real_drawer = ps.register_point_cloud("Real Drawer Data", real_drawer_data_np[0], enabled=False)

def store_point_cloud(points, joint_type):
    """Store the current frame's point cloud for a given joint type."""
    key = joint_type.replace(" ", "_")
    point_cloud_history[key].append(points.copy())

def save_all_to_npy(joint_type):
    """Save all recorded frames to .npy for a given joint type."""
    key = joint_type.replace(" ", "_")
    if len(point_cloud_history[key]) == 0:
        print("No data to save for", joint_type)
        return
    global file_counter
    if joint_type not in file_counter:
        file_counter[joint_type] = 0
    else:
        file_counter[joint_type] += 1
    all_points = np.stack(point_cloud_history[key], axis=0)
    joint_dir = os.path.join(output_dir, key)
    os.makedirs(joint_dir, exist_ok=True)
    filename = f"{key}_{file_counter[joint_type]}.npy"
    filepath = os.path.join(joint_dir, filename)
    np.save(filepath, all_points)
    print("Saved data to", filepath)

def remove_joint_visual():
    """Remove previously drawn joint lines from Polyscope."""
    for name in [
        "Planar Normal", "Ball Center", "Screw Axis", "Screw Axis Pitch",
        "Prismatic Axis", "Revolute Axis", "Revolute Origin", "Planar Axes"
    ]:
        if ps.has_curve_network(name):
            ps.remove_curve_network(name)
    if ps.has_point_cloud("BallCenterPC"):
        ps.remove_point_cloud("BallCenterPC")

def show_joint_visual(joint_type, joint_params):
    """Show lines/axes in Polyscope for the given joint type."""
    remove_joint_visual()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps = 1e-6

    def torch_normalize(vec_t):
        norm_ = torch.norm(vec_t)
        return vec_t if norm_ < eps else vec_t / norm_

    if joint_type == "planar":
        n_np = joint_params.get("normal", np.array([0., 0., 1.]))
        seg_nodes = np.array([[0, 0, 0], n_np])
        seg_edges = np.array([[0, 1]])
        planarnet = ps.register_curve_network("Planar Normal", seg_nodes, seg_edges)
        planarnet.set_color((1.0, 0.0, 0.0))
        planarnet.set_radius(0.02)
        n_t = torch.tensor(n_np, device=device)
        y_t = torch.tensor([0., 1., 0.], device=device)
        cross_1 = torch.cross(n_t, y_t, dim=0)
        cross_1 = torch_normalize(cross_1)
        if torch.norm(cross_1) < eps:
            cross_1 = torch.tensor([1., 0., 0.], device=device)
        cross_2 = torch.cross(n_t, cross_1, dim=0)
        cross_2 = torch_normalize(cross_2.unsqueeze(0))[0]
        seg_nodes2 = np.array([[0, 0, 0], cross_1.cpu().numpy(), [0, 0, 0], cross_2.cpu().numpy()], dtype=np.float32)
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
        seg_nodes = np.array([center_np, center_np + x_, center_np, center_np + y_, center_np, center_np + z_])
        seg_edges = np.array([[0, 1], [2, 3], [4, 5]])
        axisviz = ps.register_curve_network("Ball Center", seg_nodes, seg_edges)
        axisviz.set_radius(0.02)
        axisviz.set_color((1., 0., 1.))

    elif joint_type == "screw":
        axis_np = joint_params.get("axis", np.array([0., 1., 0.]))
        origin_np = joint_params.get("origin", np.array([0., 0., 0.]))
        pitch_ = joint_params.get("pitch", 0.0)
        seg_nodes = np.array([origin_np - axis_np * 0.5, origin_np + axis_np * 0.5])
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
        seg_nodes = np.array([origin_np - axis_np * 0.5, origin_np + axis_np * 0.5])
        seg_edges = np.array([[0, 1]])
        rvnet = ps.register_curve_network("Revolute Axis", seg_nodes, seg_edges)
        rvnet.set_radius(0.02)
        rvnet.set_color((1., 1., 0.))
        seg_nodes2 = np.array([origin_np, origin_np])
        seg_edges2 = np.array([[0, 1]])
        origin_net = ps.register_curve_network("Revolute Origin", seg_nodes2, seg_edges2)
        origin_net.set_radius(0.03)
        origin_net.set_color((1., 0., 0.))

def highlight_max_points(ps_cloud, current_points, prev_points):
    """Color the difference between consecutive frames."""
    if prev_points is None or current_points.shape != prev_points.shape:
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    curr_t = torch.tensor(current_points, dtype=torch.float32, device=device)
    prev_t = torch.tensor(prev_points, dtype=torch.float32, device=device)
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

def clear_data_for_mode(mode):
    """Clear all recorded point clouds and plot data for this mode."""
    key = mode.replace(" ", "_")
    point_cloud_history[key] = []
    velocity_profile[mode] = []
    angular_velocity_profile[mode] = []
    col_score_profile[mode] = []
    cop_score_profile[mode] = []
    rad_score_profile[mode] = []
    zp_score_profile[mode] = []
    error_profile[mode] = []
    for jt in joint_prob_profile[mode]:
        joint_prob_profile[mode][jt] = []

def restore_original_shape(mode):
    """Reset the object in a particular mode to its original shape."""
    global prismatic_door_points, prismatic_door_points_2, prismatic_door_points_3
    global revolute_door_points, revolute_door_points_2, revolute_door_points_3
    global mouse_points, mouse_points_2, mouse_points_3
    global joint_points, joint_points_2, joint_points_3
    global cap_points, cap_points_2, cap_points_3
    global ps_prismatic_door, ps_prismatic_door_2, ps_prismatic_door_3
    global ps_revolute_door, ps_revolute_door_2, ps_revolute_door_3
    global ps_mouse, ps_mouse_2, ps_mouse_3
    global ps_joint, ps_joint_2, ps_joint_3
    global ps_cap, ps_cap_2, ps_cap_3
    global ps_real_drawer

    ps_prismatic_door.set_enabled(mode == "Prismatic Door")
    ps_prismatic_door_2.set_enabled(mode == "Prismatic Door 2")
    ps_prismatic_door_3.set_enabled(mode == "Prismatic Door 3")

    ps_revolute_door.set_enabled(mode == "Revolute Door")
    ps_revolute_door_2.set_enabled(mode == "Revolute Door 2")
    ps_revolute_door_3.set_enabled(mode == "Revolute Door 3")

    ps_mouse.set_enabled(mode == "Planar Mouse")
    ps_mouse_2.set_enabled(mode == "Planar Mouse 2")
    ps_mouse_3.set_enabled(mode == "Planar Mouse 3")

    ps_joint.set_enabled(mode == "Ball Joint")
    ps_joint_2.set_enabled(mode == "Ball Joint 2")
    ps_joint_3.set_enabled(mode == "Ball Joint 3")

    ps_cap.set_enabled(mode == "Screw Joint")
    ps_cap_2.set_enabled(mode == "Screw Joint 2")
    ps_cap_3.set_enabled(mode == "Screw Joint 3")

    ps_real_drawer.set_enabled(mode == "Real Drawer Data")

    if mode == "Prismatic Door":
        prismatic_door_points = original_prismatic_door_points.copy()
        ps_prismatic_door.update_point_positions(prismatic_door_points)
    elif mode == "Prismatic Door 2":
        prismatic_door_points_2 = original_prismatic_door_points_2.copy()
        ps_prismatic_door_2.update_point_positions(prismatic_door_points_2)
    elif mode == "Prismatic Door 3":
        prismatic_door_points_3 = original_prismatic_door_points_3.copy()
        ps_prismatic_door_3.update_point_positions(prismatic_door_points_3)

    elif mode == "Revolute Door":
        revolute_door_points = original_revolute_door_points.copy()
        ps_revolute_door.update_point_positions(revolute_door_points)
    elif mode == "Revolute Door 2":
        revolute_door_points_2 = original_revolute_door_points_2.copy()
        ps_revolute_door_2.update_point_positions(revolute_door_points_2)
    elif mode == "Revolute Door 3":
        revolute_door_points_3 = original_revolute_door_points_3.copy()
        ps_revolute_door_3.update_point_positions(revolute_door_points_3)

    elif mode == "Planar Mouse":
        mouse_points = original_mouse_points.copy()
        ps_mouse.update_point_positions(mouse_points)
    elif mode == "Planar Mouse 2":
        mouse_points_2 = original_mouse_points_2.copy()
        ps_mouse_2.update_point_positions(mouse_points_2)
    elif mode == "Planar Mouse 3":
        mouse_points_3 = original_mouse_points_3.copy()
        ps_mouse_3.update_point_positions(mouse_points_3)

    elif mode == "Ball Joint":
        joint_points = original_joint_points.copy()
        ps_joint.update_point_positions(joint_points)
    elif mode == "Ball Joint 2":
        joint_points_2 = original_joint_points_2.copy()
        ps_joint_2.update_point_positions(joint_points_2)
    elif mode == "Ball Joint 3":
        joint_points_3 = original_joint_points_3.copy()
        ps_joint_3.update_point_positions(joint_points_3)

    elif mode == "Screw Joint":
        cap_points = original_cap_points.copy()
        ps_cap.update_point_positions(cap_points)
    elif mode == "Screw Joint 2":
        cap_points_2 = original_cap_points_2.copy()
        ps_cap_2.update_point_positions(cap_points_2)
    elif mode == "Screw Joint 3":
        cap_points_3 = original_cap_points_3.copy()
        ps_cap_3.update_point_positions(cap_points_3)

    elif mode == "Real Drawer Data":
        if real_drawer_data_np.shape[0] > 0:
            init_points = real_drawer_data_np[0]
        else:
            init_points = np.zeros((1, 3))
        ps_real_drawer.update_point_positions(init_points)

    clear_data_for_mode(mode)

def compute_error_for_mode(mode, param_dict):
    """Simple error function w.r.t. basic ground-truth. Returns 0 for real data."""
    if mode == "Real Drawer Data":
        return 0.0

    # Simple demonstration of checking the estimated axis/center vs. a ground truth
    # Prismatic Door 1,2,3
    if mode == "Prismatic Door":
        prismatic_door_gt_axis = np.array([1., 0., 0.])
        info = param_dict["prismatic"]
        est_axis = info["axis"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, prismatic_door_gt_axis)
        err = abs(1.0 - abs(dotv))
        return err
    elif mode == "Prismatic Door 2":
        prismatic_door_gt_axis_2 = np.array([0., 1., 0.])
        info = param_dict["prismatic"]
        est_axis = info["axis"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, prismatic_door_gt_axis_2)
        err = abs(1.0 - abs(dotv))
        return err
    elif mode == "Prismatic Door 3":
        prismatic_door_gt_axis_3 = np.array([0., 0., 1.])
        info = param_dict["prismatic"]
        est_axis = info["axis"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, prismatic_door_gt_axis_3)
        err = abs(1.0 - abs(dotv))
        return err

    # Revolute Doors
    elif mode == "Revolute Door":
        info = param_dict["revolute"]
        est_axis = info["axis"]
        est_origin = info["origin"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, revolute_door_gt_axis_norm)
        dir_err = abs(1.0 - abs(dotv))
        d12 = revolute_door_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, revolute_door_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            line_dist = np.linalg.norm(np.cross(d12, revolute_door_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            line_dist = abs(np.dot(d12, n_))
        line_err = line_dist / revolute_door_max_dist
        return 0.5 * (dir_err + line_err)

    elif mode == "Revolute Door 2":
        info = param_dict["revolute"]
        est_axis = info["axis"]
        est_origin = info["origin"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, revolute_door_2_gt_axis_norm)
        dir_err = abs(1.0 - abs(dotv))
        d12 = revolute_door_2_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, revolute_door_2_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            line_dist = np.linalg.norm(np.cross(d12, revolute_door_2_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            line_dist = abs(np.dot(d12, n_))
        line_err = line_dist / revolute_door_2_max_dist
        return 0.5 * (dir_err + line_err)

    elif mode == "Revolute Door 3":
        info = param_dict["revolute"]
        est_axis = info["axis"]
        est_origin = info["origin"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, revolute_door_3_gt_axis_norm)
        dir_err = abs(1.0 - abs(dotv))
        d12 = revolute_door_3_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, revolute_door_3_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            line_dist = np.linalg.norm(np.cross(d12, revolute_door_3_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            line_dist = abs(np.dot(d12, n_))
        line_err = line_dist / revolute_door_3_max_dist
        return 0.5 * (dir_err + line_err)

    # Planar Mice
    elif mode == "Planar Mouse":
        info = param_dict["planar"]
        est_normal = info["normal"]
        est_normal_norm = est_normal / (np.linalg.norm(est_normal) + 1e-9)
        dotv = np.dot(est_normal_norm, planar_mouse_gt_normal)
        err = abs(1.0 - abs(dotv))
        return err
    elif mode == "Planar Mouse 2":
        info = param_dict["planar"]
        est_normal = info["normal"]
        est_normal_norm = est_normal / (np.linalg.norm(est_normal) + 1e-9)
        dotv = np.dot(est_normal_norm, planar_mouse_2_gt_normal)
        err = abs(1.0 - abs(dotv))
        return err
    elif mode == "Planar Mouse 3":
        info = param_dict["planar"]
        est_normal = info["normal"]
        est_normal_norm = est_normal / (np.linalg.norm(est_normal) + 1e-9)
        dotv = np.dot(est_normal_norm, planar_mouse_3_gt_normal)
        err = abs(1.0 - abs(dotv))
        return err

    # Ball Joints
    elif mode == "Ball Joint":
        info = param_dict["ball"]
        est_center = info["center"]
        dist_ = np.linalg.norm(est_center - ball_joint_gt_center)
        err = dist_ / ball_joint_max_dist
        return err
    elif mode == "Ball Joint 2":
        info = param_dict["ball"]
        est_center = info["center"]
        dist_ = np.linalg.norm(est_center - ball_joint_2_gt_center)
        err = dist_ / ball_joint_2_max_dist
        return err
    elif mode == "Ball Joint 3":
        info = param_dict["ball"]
        est_center = info["center"]
        dist_ = np.linalg.norm(est_center - ball_joint_3_gt_center)
        err = dist_ / ball_joint_3_max_dist
        return err

    # Screw Joints
    elif mode == "Screw Joint":
        info = param_dict["screw"]
        est_axis = info["axis"]
        est_origin = info["origin"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, screw_gt_axis_norm)
        dir_err = abs(1.0 - abs(dotv))

        d12 = screw_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, screw_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            line_dist = np.linalg.norm(np.cross(d12, screw_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            line_dist = abs(np.dot(d12, n_))
        line_err = line_dist / screw_joint_max_dist
        return 0.5 * (dir_err + line_err)

    elif mode == "Screw Joint 2":
        info = param_dict["screw"]
        est_axis = info["axis"]
        est_origin = info["origin"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, screw_2_gt_axis_norm)
        dir_err = abs(1.0 - abs(dotv))

        d12 = screw_2_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, screw_2_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            line_dist = np.linalg.norm(np.cross(d12, screw_2_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            line_dist = abs(np.dot(d12, n_))
        line_err = line_dist / screw_joint_2_max_dist
        return 0.5 * (dir_err + line_err)

    elif mode == "Screw Joint 3":
        info = param_dict["screw"]
        est_axis = info["axis"]
        est_origin = info["origin"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, screw_3_gt_axis_norm)
        dir_err = abs(1.0 - abs(dotv))

        d12 = screw_3_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, screw_3_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            line_dist = np.linalg.norm(np.cross(d12, screw_3_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            line_dist = abs(np.dot(d12, n_))
        line_err = line_dist / screw_joint_3_max_dist
        return 0.5 * (dir_err + line_err)

    return 0.0

registered_gt_objects = set()
def show_ground_truth_visual(mode, enable=True):
    """Enable/disable a basic ground-truth visualization for each mode in Polyscope."""
    gt_name_axis = f"GT {mode} Axis"
    gt_name_origin = f"GT {mode} Origin"
    gt_name_center = f"GT {mode} Center"
    gt_name_normal = f"GT {mode} Normal"
    gt_name_axes_2d = f"GT {mode} PlaneAxes"
    gt_name_pitch = f"GT {mode} PitchArrow"

    if ps.has_curve_network(gt_name_axis):
        ps.remove_curve_network(gt_name_axis)
    if ps.has_curve_network(gt_name_origin):
        ps.remove_curve_network(gt_name_origin)
    if ps.has_curve_network(gt_name_normal):
        ps.remove_curve_network(gt_name_normal)
    if ps.has_curve_network(gt_name_axes_2d):
        ps.remove_curve_network(gt_name_axes_2d)
    if ps.has_curve_network(gt_name_pitch):
        ps.remove_curve_network(gt_name_pitch)
    if ps.has_point_cloud(gt_name_center):
        ps.remove_point_cloud(gt_name_center)

    if not enable:
        if mode in registered_gt_objects:
            registered_gt_objects.remove(mode)
        return

    # Very simplified GT demonstration
    if mode == "Prismatic Door":
        axis_np = np.array([1.0, 0., 0.])
        seg_nodes = np.array([[0, 0, 0], axis_np])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
        net.set_radius(0.02)
        net.set_color((1., 0., 0.))
        registered_gt_objects.add(mode)

    elif mode == "Prismatic Door 2":
        axis_np = np.array([0., 1., 0.])
        seg_nodes = np.array([[0, 0, 0], axis_np])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
        net.set_radius(0.02)
        net.set_color((0., 1., 0.))
        registered_gt_objects.add(mode)

    elif mode == "Prismatic Door 3":
        axis_np = np.array([0., 0., 1.])
        seg_nodes = np.array([[0, 0, 0], axis_np])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
        net.set_radius(0.02)
        net.set_color((0., 0., 1.))
        registered_gt_objects.add(mode)

    elif mode == "Revolute Door":
        axis_np = np.array([0., 1., 0.])
        origin_np = np.array([1.0, 1.5, 0.0])
        seg_nodes = np.array([origin_np - axis_np * 0.5, origin_np + axis_np * 0.5])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
        net.set_radius(0.02)
        net.set_color((0., 1., 0.))
        seg_nodes2 = np.array([origin_np, origin_np + 1e-5 * axis_np])
        seg_edges2 = np.array([[0, 1]])
        net2 = ps.register_curve_network(gt_name_origin, seg_nodes2, seg_edges2)
        net2.set_radius(0.03)
        net2.set_color((1., 0., 0.))
        registered_gt_objects.add(mode)

    elif mode == "Revolute Door 2":
        axis_np = np.array([1., 0., 0.])
        origin_np = np.array([0.5, 2.0, -1.0])
        seg_nodes = np.array([origin_np - axis_np * 0.5, origin_np + axis_np * 0.5])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
        net.set_radius(0.02)
        net.set_color((1., 0., 0.))
        seg_nodes2 = np.array([origin_np, origin_np + 1e-5 * axis_np])
        seg_edges2 = np.array([[0, 1]])
        net2 = ps.register_curve_network(gt_name_origin, seg_nodes2, seg_edges2)
        net2.set_radius(0.03)
        net2.set_color((1., 0., 0.))
        registered_gt_objects.add(mode)

    elif mode == "Revolute Door 3":
        axis_np = np.array([1., 1., 0.])
        origin_np = np.array([2., 1., 1.])
        seg_nodes = np.array([origin_np - axis_np * 0.2, origin_np + axis_np * 0.2])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
        net.set_radius(0.02)
        net.set_color((1., 0., 1.))
        seg_nodes2 = np.array([origin_np, origin_np + 1e-5 * axis_np])
        seg_edges2 = np.array([[0, 1]])
        net2 = ps.register_curve_network(gt_name_origin, seg_nodes2, seg_edges2)
        net2.set_radius(0.03)
        net2.set_color((1., 0., 0.))
        registered_gt_objects.add(mode)

    elif mode.startswith("Planar Mouse"):
        if mode == "Planar Mouse":
            normal_ = np.array([0., 1., 0.])
        elif mode == "Planar Mouse 2":
            normal_ = np.array([1., 0., 0.])
        else:
            normal_ = np.array([0., 0., 1.])
        seg_nodes = np.array([[0, 0, 0], normal_])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_normal, seg_nodes, seg_edges)
        net.set_color((0., 1., 1.))
        net.set_radius(0.02)
        seg_nodes2 = np.array([[0, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 1]])
        seg_edges2 = np.array([[0, 1], [2, 3]])
        net2 = ps.register_curve_network(gt_name_axes_2d, seg_nodes2, seg_edges2)
        net2.set_radius(0.02)
        net2.set_color((1., 1., 0.))
        registered_gt_objects.add(mode)

    elif mode == "Ball Joint":
        c_ = np.array([0.0, 0.0, 0.0])
        pc = ps.register_point_cloud(gt_name_center, c_.reshape(1, 3))
        pc.set_radius(0.05)
        pc.set_color((1., 0., 1.))
        registered_gt_objects.add(mode)

    elif mode == "Ball Joint 2":
        c_ = np.array([1., 0., 0.])
        pc = ps.register_point_cloud(gt_name_center, c_.reshape(1, 3))
        pc.set_radius(0.05)
        pc.set_color((1., 0., 1.))
        registered_gt_objects.add(mode)

    elif mode == "Ball Joint 3":
        c_ = np.array([1., 1., 0.])
        pc = ps.register_point_cloud(gt_name_center, c_.reshape(1, 3))
        pc.set_radius(0.05)
        pc.set_color((1., 0., 1.))
        registered_gt_objects.add(mode)

    elif mode == "Screw Joint":
        axis_np = np.array([0., 1., 0.])
        origin_np = np.array([0., 0., 0.])
        p_ = 0.5
        seg_nodes = np.array([origin_np - axis_np * 0.5, origin_np + axis_np * 0.5])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
        net.set_radius(0.02)
        net.set_color((1., 0., 0.))
        arrow_start = origin_np + axis_np * 0.6
        arrow_end = arrow_start + p_ * np.array([1., 0., 0.]) * 0.2
        seg_nodes2 = np.array([arrow_start, arrow_end])
        seg_edges2 = np.array([[0, 1]])
        net2 = ps.register_curve_network(gt_name_pitch, seg_nodes2, seg_edges2)
        net2.set_radius(0.02)
        net2.set_color((0., 1., 1.))
        registered_gt_objects.add(mode)

    elif mode == "Screw Joint 2":
        axis_np = np.array([1., 0., 0.])
        origin_np = np.array([1., 0., 0.])
        p_ = 0.8
        seg_nodes = np.array([origin_np - axis_np * 0.5, origin_np + axis_np * 0.5])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
        net.set_radius(0.02)
        net.set_color((1., 0., 0.))
        arrow_start = origin_np + axis_np * 0.6
        arrow_end = arrow_start + p_ * np.array([0., 1., 0.]) * 0.2
        seg_nodes2 = np.array([arrow_start, arrow_end])
        seg_edges2 = np.array([[0, 1]])
        net2 = ps.register_curve_network(gt_name_pitch, seg_nodes2, seg_edges2)
        net2.set_radius(0.02)
        net2.set_color((0., 1., 1.))
        registered_gt_objects.add(mode)

    elif mode == "Screw Joint 3":
        axis_np = np.array([1., 1., 0.])
        origin_np = np.array([-1., 0., 1.])
        p_ = 0.6
        seg_nodes = np.array([origin_np - axis_np * 0.2, origin_np + axis_np * 0.2])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
        net.set_radius(0.02)
        net.set_color((1., 0., 0.))
        arrow_start = origin_np + axis_np * 0.3
        arrow_end = arrow_start + p_ * np.array([1., 0., 0.]) * 0.2
        seg_nodes2 = np.array([arrow_start, arrow_end])
        seg_edges2 = np.array([[0, 1]])
        net2 = ps.register_curve_network(gt_name_pitch, seg_nodes2, seg_edges2)
        net2.set_radius(0.02)
        net2.set_color((0., 1., 1.))
        registered_gt_objects.add(mode)

def update_motion_and_store(mode):
    """
    Advance to the next frame of motion for the current mode, store the frame, and update Polyscope.
    """
    global frame_count_per_mode
    global prismatic_door_points, prismatic_door_points_2, prismatic_door_points_3
    global revolute_door_points, revolute_door_points_2, revolute_door_points_3
    global mouse_points, mouse_points_2, mouse_points_3
    global joint_points, joint_points_2, joint_points_3
    global cap_points, cap_points_2, cap_points_3
    global ps_prismatic_door, ps_prismatic_door_2, ps_prismatic_door_3
    global ps_revolute_door, ps_revolute_door_2, ps_revolute_door_3
    global ps_mouse, ps_mouse_2, ps_mouse_3
    global ps_joint, ps_joint_2, ps_joint_3
    global ps_cap, ps_cap_2, ps_cap_3
    global ps_real_drawer
    global noise_sigma, screw_pitch, screw_pitch_2, screw_pitch_3

    fidx = frame_count_per_mode[mode]
    limit = TOTAL_FRAMES_PER_MODE[mode]
    if fidx >= limit:
        return

    prev_points = None

    # -- Prismatic Door 1,2,3 --
    if mode == "Prismatic Door":
        prev_points = prismatic_door_points.copy()
        pos = (fidx / (limit - 1)) * 5.0
        prismatic_door_points = original_prismatic_door_points.copy()
        prismatic_door_points = translate_points(prismatic_door_points, pos, np.array([1., 0., 0.]))
        if noise_sigma > 1e-6:
            prismatic_door_points += np.random.normal(0, noise_sigma, prismatic_door_points.shape)
        ps_prismatic_door.update_point_positions(prismatic_door_points)
        store_point_cloud(prismatic_door_points, mode)
        highlight_max_points(ps_prismatic_door, prismatic_door_points, prev_points)

    elif mode == "Prismatic Door 2":
        prev_points = prismatic_door_points_2.copy()
        pos = (fidx / (limit - 1)) * 4.0
        prismatic_door_points_2 = original_prismatic_door_points_2.copy()
        prismatic_door_points_2 = translate_points(prismatic_door_points_2, pos, np.array([0., 1., 0.]))
        if noise_sigma > 1e-6:
            prismatic_door_points_2 += np.random.normal(0, noise_sigma, prismatic_door_points_2.shape)
        ps_prismatic_door_2.update_point_positions(prismatic_door_points_2)
        store_point_cloud(prismatic_door_points_2, mode)
        highlight_max_points(ps_prismatic_door_2, prismatic_door_points_2, prev_points)

    elif mode == "Prismatic Door 3":
        prev_points = prismatic_door_points_3.copy()
        pos = (fidx / (limit - 1)) * 3.0
        prismatic_door_points_3 = original_prismatic_door_points_3.copy()
        prismatic_door_points_3 = translate_points(prismatic_door_points_3, pos, np.array([0., 0., 1.]))
        if noise_sigma > 1e-6:
            prismatic_door_points_3 += np.random.normal(0, noise_sigma, prismatic_door_points_3.shape)
        ps_prismatic_door_3.update_point_positions(prismatic_door_points_3)
        store_point_cloud(prismatic_door_points_3, mode)
        highlight_max_points(ps_prismatic_door_3, prismatic_door_points_3, prev_points)

    # -- Revolute Door 1,2,3 --
    elif mode == "Revolute Door":
        prev_points = revolute_door_points.copy()
        angle_min = -math.radians(45.0)
        angle_max = math.radians(45.0)
        angle = angle_min + (angle_max - angle_min) * (fidx / (limit - 1))
        revolute_door_points = original_revolute_door_points.copy()
        revolute_door_points = rotate_points(revolute_door_points, angle, np.array([0., 1., 0.]),
                                             np.array([1., 1.5, 0.]))
        if noise_sigma > 1e-6:
            revolute_door_points += np.random.normal(0, noise_sigma, revolute_door_points.shape)
        ps_revolute_door.update_point_positions(revolute_door_points)
        store_point_cloud(revolute_door_points, mode)
        highlight_max_points(ps_revolute_door, revolute_door_points, prev_points)

    elif mode == "Revolute Door 2":
        prev_points = revolute_door_points_2.copy()
        angle_min = -math.radians(30.0)
        angle_max = math.radians(60.0)
        angle = angle_min + (angle_max - angle_min) * (fidx / (limit - 1))
        revolute_door_points_2 = original_revolute_door_points_2.copy()
        revolute_door_points_2 = rotate_points(revolute_door_points_2, angle, door_hinge_axis_2, door_hinge_position_2)
        if noise_sigma > 1e-6:
            revolute_door_points_2 += np.random.normal(0, noise_sigma, revolute_door_points_2.shape)
        ps_revolute_door_2.update_point_positions(revolute_door_points_2)
        store_point_cloud(revolute_door_points_2, mode)
        highlight_max_points(ps_revolute_door_2, revolute_door_points_2, prev_points)

    elif mode == "Revolute Door 3":
        prev_points = revolute_door_points_3.copy()
        angle_min = 0.0
        angle_max = math.radians(90.0)
        angle = angle_min + (angle_max - angle_min) * (fidx / (limit - 1))
        revolute_door_points_3 = original_revolute_door_points_3.copy()
        revolute_door_points_3 = rotate_points(revolute_door_points_3, angle, door_hinge_axis_3, door_hinge_position_3)
        if noise_sigma > 1e-6:
            revolute_door_points_3 += np.random.normal(0, noise_sigma, revolute_door_points_3.shape)
        ps_revolute_door_3.update_point_positions(revolute_door_points_3)
        store_point_cloud(revolute_door_points_3, mode)
        highlight_max_points(ps_revolute_door_3, revolute_door_points_3, prev_points)

    # # -- Planar Mouse 1,2,3 --
    elif mode == "Planar Mouse":
        prev_points = mouse_points.copy()
        if fidx < 20:
            alpha = fidx / 19.0
            tx = -1.0 + alpha * (0.0 - (-1.0))
            tz = 1.0 + alpha * (0.0 - 1.0)
            ry = 0.0
            mouse_points = original_mouse_points.copy()
            mouse_points = mouse_points + np.array([tx, 0., tz])
            mouse_points = rotate_points_y(mouse_points, ry, [0., 0., 0.])
        elif fidx < 30:
            alpha = (fidx - 20) / 9.0
            ry = math.radians(40.0) * alpha
            mouse_points = original_mouse_points.copy()
            mouse_points = rotate_points_y(mouse_points, ry, [0., 0., 0.])
        else:
            alpha = (fidx - 30) / 9.0
            tx = 0.0 + alpha * 1.0
            tz = 0.0 + alpha * (-1.0)
            ry = math.radians(40.0)
            mouse_points = original_mouse_points.copy()
            mouse_points = mouse_points + np.array([tx, 0., tz])
            mouse_points = rotate_points_y(mouse_points, ry, [0., 0., 0.])
        if noise_sigma > 1e-6:
            mouse_points += np.random.normal(0, noise_sigma, mouse_points.shape)
        ps_mouse.update_point_positions(mouse_points)
        store_point_cloud(mouse_points, mode)
        highlight_max_points(ps_mouse, mouse_points, prev_points)

    # elif mode == "Planar Mouse 2":
    #     prev_points = mouse_points_2.copy()
    #     if fidx < 20:
    #         alpha = fidx / 19.0
    #         dx = alpha * 1.0
    #         dy = 0.
    #         mouse_points_2 = original_mouse_points_2.copy()
    #         mouse_points_2 += np.array([dx, dy, 0.])
    #     elif fidx < 30:
    #         alpha = (fidx - 20) / 9.0
    #         r_ = math.radians(30.0) * alpha
    #         mouse_points_2 = original_mouse_points_2.copy()
    #         mouse_points_2 = rotate_points(mouse_points_2, r_, np.array([0., 0., 1.]), np.array([1.5, 0., 1.]))
    #     else:
    #         alpha = (fidx - 30) / 9.0
    #         dx = 1.0
    #         dy = alpha * 1.0
    #         mouse_points_2 = original_mouse_points_2.copy()
    #         mouse_points_2 += np.array([dx, dy, 0.])
    #     if noise_sigma > 1e-6:
    #         mouse_points_2 += np.random.normal(0, noise_sigma, mouse_points_2.shape)
    #     ps_mouse_2.update_point_positions(mouse_points_2)
    #     store_point_cloud(mouse_points_2, mode)
    #     highlight_max_points(ps_mouse_2, mouse_points_2, prev_points)
    #
    # elif mode == "Planar Mouse 3":
    #     prev_points = mouse_points_3.copy()
    #     if fidx < 20:
    #         alpha = fidx / 19.0
    #         dx = alpha * -1.0
    #         dz = alpha * 0.5
    #         mouse_points_3 = original_mouse_points_3.copy()
    #         mouse_points_3 += np.array([dx, 0., dz])
    #     else:
    #         alpha = (fidx - 20) / 19.0
    #         rx = math.radians(30.0) * alpha
    #         mouse_points_3 = original_mouse_points_3.copy()
    #         mouse_points_3 = rotate_points(mouse_points_3, rx, np.array([0., 0., 1.]), np.array([1.5, 0., 1.]))
    #     if noise_sigma > 1e-6:
    #         mouse_points_3 += np.random.normal(0, noise_sigma, mouse_points_3.shape)
    #     ps_mouse_3.update_point_positions(mouse_points_3)
    #     store_point_cloud(mouse_points_3, mode)
    #     highlight_max_points(ps_mouse_3, mouse_points_3, prev_points)

    ####################################
    #           Planar Mouse 2         #
    ####################################
    elif mode == "Planar Mouse 2":
        prev_points = mouse_points_2.copy()

        if fidx < 20:

            alpha = fidx / 19.0

            dy = alpha * 1.0
            dz = alpha * 1.0


            mouse_points_2 = original_mouse_points_2.copy()
            mouse_points_2 += np.array([0., dy, dz])

        elif fidx < 30:

            alpha = (fidx - 20) / 9.0

            mp = original_mouse_points_2.copy()
            mp += np.array([0., 1.0, 1.0])


            rx = math.radians(30.0) * alpha


            mouse_points_2 = rotate_points(mp, rx,
                                           np.array([1.0, 0., 0.]),
                                           np.array([0., 1.0, 1.0]))

        else:

            alpha = (fidx - 30) / 9.0

            mp = original_mouse_points_2.copy()
            mp += np.array([0., 1.0, 1.0])

            rx = math.radians(30.0)
            mp = rotate_points(mp, rx,
                               np.array([1.0, 0., 0.]),
                               np.array([0., 1.0, 1.0]))


            dy = alpha * 1.0
            dz = alpha * 0.5
            mp += np.array([0., dy, dz])

            mouse_points_2 = mp


        if noise_sigma > 1e-6:
            mouse_points_2 += np.random.normal(0, noise_sigma, mouse_points_2.shape)


        ps_mouse_2.update_point_positions(mouse_points_2)
        store_point_cloud(mouse_points_2, mode)
        highlight_max_points(ps_mouse_2, mouse_points_2, prev_points)

    ####################################
    #           Planar Mouse 3         #
    ####################################
    elif mode == "Planar Mouse 3":
        prev_points = mouse_points_3.copy()

        if fidx < 20:
            alpha = fidx / 19.0
            dx = alpha * 1.0
            dz = alpha * 0.5

            mouse_points_3 = original_mouse_points_3.copy()
            mouse_points_3 += np.array([dx, 0., dz])

        elif fidx < 30:

            alpha = (fidx - 20) / 9.0

            mp = original_mouse_points_3.copy()
            mp += np.array([1.0, 0., 0.5])


            ry = math.radians(30.0) * alpha
            mouse_points_3 = rotate_points(mp, ry,
                                           np.array([0., 1., 0.]),
                                           np.array([1.0, 0., 0.5]))

        else:

            alpha = (fidx - 30) / 9.0
            mp = original_mouse_points_3.copy()
            mp += np.array([1.0, 0., 0.5])

            ry = math.radians(30.0)
            mp = rotate_points(mp, ry,
                               np.array([0., 1., 0.]),
                               np.array([1.0, 0., 0.5]))


            dx = alpha * 1.0
            dz = alpha * 0.5
            mp += np.array([dx, 0., dz])

            mouse_points_3 = mp


        if noise_sigma > 1e-6:
            mouse_points_3 += np.random.normal(0, noise_sigma, mouse_points_3.shape)


        ps_mouse_3.update_point_positions(mouse_points_3)
        store_point_cloud(mouse_points_3, mode)
        highlight_max_points(ps_mouse_3, mouse_points_3, prev_points)


    # -- Ball Joint 1,2,3 --
    elif mode == "Ball Joint":
        prev_points = joint_points.copy()
        if fidx < 10:
            ax = math.radians(60.0) * (fidx / 10.0)
            ay = 0.0
            az = 0.0
        elif fidx < 20:
            alpha = (fidx - 10) / 10.0
            ax = math.radians(60.0)
            ay = math.radians(40.0) * alpha
            az = 0.0
        else:
            alpha = (fidx - 20) / 20.0
            ax = math.radians(60.0)
            ay = math.radians(40.0)
            az = math.radians(70.0) * alpha

        joint_points = original_joint_points.copy()
        joint_points = rotate_points_xyz(joint_points, ax, ay, az, np.array([0., 0., 0.]))
        if noise_sigma > 1e-6:
            joint_points += np.random.normal(0, noise_sigma, joint_points.shape)
        ps_joint.update_point_positions(joint_points)
        store_point_cloud(joint_points, mode)
        highlight_max_points(ps_joint, joint_points, prev_points)

    elif mode == "Ball Joint 2":
        prev_points = joint_points_2.copy()
        if fidx < 20:
            alpha = fidx / 19.0
            rx = math.radians(50.0) * alpha
            ry = math.radians(10.0) * alpha
            joint_points_2 = original_joint_points_2.copy()
            joint_points_2 = rotate_points_xyz(joint_points_2, rx, ry, 0.0, np.array([1., 0., 0.]))
        else:
            alpha = (fidx - 20) / 19.0
            rx = math.radians(50.0)
            ry = math.radians(10.0)
            rz = math.radians(45.0) * alpha
            joint_points_2 = original_joint_points_2.copy()
            joint_points_2 = rotate_points_xyz(joint_points_2, rx, ry, rz, np.array([1., 0., 0.]))
        if noise_sigma > 1e-6:
            joint_points_2 += np.random.normal(0, noise_sigma, joint_points_2.shape)
        ps_joint_2.update_point_positions(joint_points_2)
        store_point_cloud(joint_points_2, mode)
        highlight_max_points(ps_joint_2, joint_points_2, prev_points)

    elif mode == "Ball Joint 3":
        prev_points = joint_points_3.copy()
        if fidx < 10:
            ax = math.radians(30.0) * (fidx / 10.0)
            ay = 0.0
            az = 0.0
        elif fidx < 20:
            alpha = (fidx - 10) / 10.0
            ax = math.radians(30.0)
            ay = math.radians(50.0) * alpha
            az = 0.0
        else:
            alpha = (fidx - 20) / 20.0
            ax = math.radians(30.0)
            ay = math.radians(50.0)
            az = math.radians(80.0) * alpha

        joint_points_3 = original_joint_points_3.copy()
        joint_points_3 = rotate_points_xyz(joint_points_3, ax, ay, az, np.array([1., 1., 0.]))
        if noise_sigma > 1e-6:
            joint_points_3 += np.random.normal(0, noise_sigma, joint_points_3.shape)
        ps_joint_3.update_point_positions(joint_points_3)
        store_point_cloud(joint_points_3, mode)
        highlight_max_points(ps_joint_3, joint_points_3, prev_points)

    # -- Screw Joint 1,2,3 --
    elif mode == "Screw Joint":
        prev_points = cap_points.copy()
        angle = 2 * math.pi * (fidx / (limit - 1))
        cap_points = original_cap_points.copy()
        cap_points = apply_screw_motion(cap_points, angle, np.array([0., 1., 0.]), np.array([0., 0., 0.]), screw_pitch)
        if noise_sigma > 1e-6:
            cap_points += np.random.normal(0, noise_sigma, cap_points.shape)
        ps_cap.update_point_positions(cap_points)
        store_point_cloud(cap_points, mode)
        highlight_max_points(ps_cap, cap_points, prev_points)

    elif mode == "Screw Joint 2":
        prev_points = cap_points_2.copy()
        angle = 2 * math.pi * (fidx / (limit - 1))
        cap_points_2 = original_cap_points_2.copy()
        cap_points_2 = apply_screw_motion(cap_points_2, angle, np.array([1., 0., 0.]), np.array([1., 0., 0.]), screw_pitch_2)
        if noise_sigma > 1e-6:
            cap_points_2 += np.random.normal(0, noise_sigma, cap_points_2.shape)
        ps_cap_2.update_point_positions(cap_points_2)
        store_point_cloud(cap_points_2, mode)
        highlight_max_points(ps_cap_2, cap_points_2, prev_points)

    elif mode == "Screw Joint 3":
        prev_points = cap_points_3.copy()
        angle = 2 * math.pi * (fidx / (limit - 1))
        cap_points_3 = original_cap_points_3.copy()
        cap_points_3 = apply_screw_motion(cap_points_3, angle, np.array([1., 1., 0.]), np.array([-1., 0., 1.]), screw_pitch_3)
        if noise_sigma > 1e-6:
            cap_points_3 += np.random.normal(0, noise_sigma, cap_points_3.shape)
        ps_cap_3.update_point_positions(cap_points_3)
        store_point_cloud(cap_points_3, mode)
        highlight_max_points(ps_cap_3, cap_points_3, prev_points)

    # -- Real Drawer Data --
    elif mode == "Real Drawer Data":
        if fidx < real_drawer_data_np.shape[0]:
            prev_positions = None
            if fidx > 0:
                prev_positions = real_drawer_data_np[fidx - 1]
            current_positions = real_drawer_data_np[fidx]
            ps_real_drawer.update_point_positions(current_positions)
            store_point_cloud(current_positions, mode)
            highlight_max_points(ps_real_drawer, current_positions, prev_positions)

    frame_count_per_mode[mode] += 1

def callback():
    """
    Polyscope user callback for each frame, handling the UI and calls to update.
    """
    global current_mode, previous_mode
    global current_best_joint, current_scores_info
    global noise_sigma
    global col_sigma, col_order
    global cop_sigma, cop_order
    global rad_sigma, rad_order
    global zp_sigma, zp_order
    global prob_sigma, prob_order
    global neighbor_k
    global running
    global use_savgol_filter, savgol_window_length, savgol_polyorder
    global use_multi_frame_fit, multi_frame_radius

    changed = psim.BeginCombo("Object Mode", current_mode)
    if changed:
        for mode in modes:
            _, selected = psim.Selectable(mode, current_mode == mode)
            if selected and mode != current_mode:
                remove_joint_visual()
                if previous_mode is not None:
                    show_ground_truth_visual(previous_mode, enable=False)
                restore_original_shape(mode)
                frame_count_per_mode[mode] = 0
                current_mode = mode
                show_ground_truth_visual(mode, enable=True)
        psim.EndCombo()

    if previous_mode is None:
        restore_original_shape(current_mode)
        show_ground_truth_visual(current_mode, enable=True)
    previous_mode = current_mode

    psim.Separator()

    if psim.TreeNodeEx("Noise & SuperGaussian + SG Filter", flags=psim.ImGuiTreeNodeFlags_DefaultOpen):
        psim.Columns(2, "mycolumns", False)
        psim.SetColumnWidth(0, 230)
        changed_k, new_k = psim.InputInt("Neighbor K", neighbor_k, 10)
        if changed_k:
            neighbor_k = max(1, new_k)
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
            zp_order = max(1, new_zo)

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

        _, use_sg_filter_new = psim.Checkbox("Use SG Filter?", use_savgol_filter)
        if use_sg_filter_new != use_savgol_filter:
            use_savgol_filter = use_sg_filter_new

        _, use_mf_new = psim.Checkbox("Use Multi-Frame Fit?", use_multi_frame_fit)
        if use_mf_new != use_multi_frame_fit:
            use_multi_frame_fit = use_mf_new

        changed_mfr, new_mfr = psim.InputInt("MultiFrame Radius", multi_frame_radius, 1)
        if changed_mfr:
            multi_frame_radius = max(1, new_mfr)

        psim.NextColumn()
        psim.Columns(1)
        psim.TreePop()

    psim.Separator()

    if psim.Button("Start"):
        frame_count_per_mode[current_mode] = 0
        restore_original_shape(current_mode)
        running = True

    if running:
        limit = TOTAL_FRAMES_PER_MODE[current_mode]
        if frame_count_per_mode[current_mode] < limit:
            update_motion_and_store(current_mode)
        else:
            running = False

    key = current_mode.replace(" ", "_")
    all_frames = point_cloud_history.get(key, None)
    if all_frames is not None and len(all_frames) >= 2:
        all_points_history = np.stack(all_frames, axis=0)
        param_dict, best_type, scores_info = compute_joint_info_all_types(
            all_points_history,
            neighbor_k=neighbor_k,
            col_sigma=col_sigma, col_order=col_order,
            cop_sigma=cop_sigma, cop_order=cop_order,
            rad_sigma=rad_sigma, rad_order=rad_order,
            zp_sigma=zp_sigma, zp_order=zp_order,
            prob_sigma=prob_sigma, prob_order=prob_order,
            use_savgol=use_savgol_filter,
            savgol_window=savgol_window_length,
            savgol_poly=savgol_polyorder,
            use_multi_frame=use_multi_frame_fit,
            multi_frame_window_radius=multi_frame_radius
        )
        current_best_joint = best_type
        current_scores_info = scores_info

        if scores_info is not None:
            v_arr = scores_info.get("v_arr", None)
            w_arr = scores_info.get("w_arr", None)
            w_i = scores_info.get("w_i", None)
            if v_arr is not None and w_arr is not None and v_arr.shape[0] >= 1 and w_i is not None:
                idx = v_arr.shape[0] - 1
                v_last = v_arr[idx]
                w_last = w_arr[idx]
                w_sum = np.sum(w_i)
                if w_sum < 1e-9:
                    w_sum = 1e-9
                inst_v_3d = np.sum(v_last * w_i[:, None], axis=0) / w_sum
                inst_w_3d = np.sum(w_last * w_i[:, None], axis=0) / w_sum
                velocity_profile[current_mode].append(np.linalg.norm(inst_v_3d))
                angular_velocity_profile[current_mode].append(np.linalg.norm(inst_w_3d))

            col_m = scores_info["basic_score_avg"]["col_mean"]
            cop_m = scores_info["basic_score_avg"]["cop_mean"]
            rad_m = scores_info["basic_score_avg"]["rad_mean"]
            zp_m = scores_info["basic_score_avg"]["zp_mean"]
            col_score_profile[current_mode].append(col_m)
            cop_score_profile[current_mode].append(cop_m)
            rad_score_profile[current_mode].append(rad_m)
            zp_score_profile[current_mode].append(zp_m)

            joint_probs = scores_info["joint_probs"]
            for jt_name in joint_prob_profile[current_mode]:
                joint_prob_profile[current_mode][jt_name].append(joint_probs[jt_name])

            err_val = compute_error_for_mode(current_mode, param_dict)
            error_profile[current_mode].append(err_val)

        if scores_info is not None:
            col_m = scores_info["basic_score_avg"]["col_mean"]
            cop_m = scores_info["basic_score_avg"]["cop_mean"]
            rad_m = scores_info["basic_score_avg"]["rad_mean"]
            zp_m = scores_info["basic_score_avg"]["zp_mean"]
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

            gt_err = error_profile[current_mode][-1] if len(error_profile[current_mode]) > 0 else 0.0
            psim.TextUnformatted(f"GT Error => {gt_err:.4f}")

        if best_type in param_dict:
            jinfo = param_dict[best_type]
            show_joint_visual(best_type, jinfo)
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
        else:
            psim.TextUnformatted(f"Best Type: {best_type} (no param)")
    else:
        psim.TextUnformatted("Not enough frames to do joint classification.")

    if psim.Button("Save .npy for current joint type"):
        save_all_to_npy(current_mode)

stop_refresh = False

def stop_callback(sender, app_data):
    global stop_refresh
    stop_refresh = not stop_refresh
    if stop_refresh:
        dpg.set_item_label(sender, "Resume Refresh")
    else:
        dpg.set_item_label(sender, "Pause Refresh")

def clear_plots_callback():
    global velocity_profile, angular_velocity_profile
    global col_score_profile, cop_score_profile, rad_score_profile, zp_score_profile
    global error_profile, joint_prob_profile

    for m in modes:
        velocity_profile[m].clear()
        angular_velocity_profile[m].clear()
        col_score_profile[m].clear()
        cop_score_profile[m].clear()
        rad_score_profile[m].clear()
        zp_score_profile[m].clear()
        error_profile[m].clear()
        for jt_name in joint_prob_profile[m]:
            joint_prob_profile[m][jt_name].clear()

    refresh_dearpygui_plots()

def setup_dearpygui():
    dpg.create_context()

    with dpg.window(label="Plots", width=1200, height=800):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Pause Refresh", callback=stop_callback)
            dpg.add_button(label="Clear Plots", callback=clear_plots_callback)

        with dpg.tab_bar():
            with dpg.tab(label="Velocity and Omega"):
                with dpg.group(horizontal=True):
                    with dpg.plot(label="Weighted Linear Velocity vs Frame", height=300, width=450):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_vel")
                        y_axis_vel = dpg.add_plot_axis(dpg.mvYAxis, label="Velocity", tag="y_axis_vel")
                        for mode in modes:
                            tag_line = f"{mode}_vel_series"
                            dpg.add_line_series([], [], label=mode, parent=y_axis_vel, tag=tag_line)

                    with dpg.plot(label="Weighted Angular Velocity vs Frame", height=300, width=450):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_omega")
                        y_axis_omega = dpg.add_plot_axis(dpg.mvYAxis, label="Omega", tag="y_axis_omega")
                        for mode in modes:
                            tag_line = f"{mode}_omega_series"
                            dpg.add_line_series([], [], label=mode, parent=y_axis_omega, tag=tag_line)

            with dpg.tab(label="Basic Scores"):
                with dpg.group(horizontal=True):
                    with dpg.group():
                        with dpg.plot(label="Collinearity Score vs Frame", height=250, width=400):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_col")
                            y_axis_col = dpg.add_plot_axis(dpg.mvYAxis, label="Col Score", tag="y_axis_col")
                            dpg.set_axis_limits("y_axis_col", -1, 2)
                            for mode in modes:
                                tag_line = f"{mode}_col_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_col, tag=tag_line)

                        with dpg.plot(label="Radius Consistency Score vs Frame", height=250, width=400):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_rad")
                            y_axis_rad = dpg.add_plot_axis(dpg.mvYAxis, label="Rad Score", tag="y_axis_rad")
                            dpg.set_axis_limits("y_axis_rad", -1, 2)
                            for mode in modes:
                                tag_line = f"{mode}_rad_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_rad, tag=tag_line)

                    with dpg.group():
                        with dpg.plot(label="Coplanarity Score vs Frame", height=250, width=400):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_cop")
                            y_axis_cop = dpg.add_plot_axis(dpg.mvYAxis, label="Cop Score", tag="y_axis_cop")
                            dpg.set_axis_limits("y_axis_cop", -1, 2)
                            for mode in modes:
                                tag_line = f"{mode}_cop_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_cop, tag=tag_line)

                        with dpg.plot(label="Zero Pitch Score vs Frame", height=250, width=400):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_zp")
                            y_axis_zp = dpg.add_plot_axis(dpg.mvYAxis, label="ZP Score", tag="y_axis_zp")
                            dpg.set_axis_limits("y_axis_zp", -1, 2)
                            for mode in modes:
                                tag_line = f"{mode}_zp_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_zp, tag=tag_line)

            with dpg.tab(label="Error"):
                with dpg.plot(label="Error vs Frame", height=300, width=800):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_err")
                    y_axis_err = dpg.add_plot_axis(dpg.mvYAxis, label="Error", tag="y_axis_err")
                    for mode in modes:
                        tag_line = f"{mode}_err_series"
                        dpg.add_line_series([], [], label=mode, parent=y_axis_err, tag=tag_line)

            with dpg.tab(label="Joint Probabilities"):
                with dpg.group(horizontal=True):
                    with dpg.group():
                        with dpg.plot(label="Prismatic Probability vs Frame", height=250, width=400):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_prob_prismatic")
                            y_axis_prob_pm = dpg.add_plot_axis(dpg.mvYAxis, label="Prismatic Prob", tag="y_axis_prob_pm")
                            dpg.set_axis_limits("y_axis_prob_pm", -1, 2)
                            for mode in modes:
                                tag_ = f"{mode}_prob_prismatic_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_prob_pm, tag=tag_)

                        with dpg.plot(label="Planar Probability vs Frame", height=250, width=400):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_prob_planar")
                            y_axis_prob_pl = dpg.add_plot_axis(dpg.mvYAxis, label="Planar Prob", tag="y_axis_prob_pl")
                            dpg.set_axis_limits("y_axis_prob_pl", -1, 2)
                            for mode in modes:
                                tag_ = f"{mode}_prob_planar_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_prob_pl, tag=tag_)

                        with dpg.plot(label="Revolute Probability vs Frame", height=250, width=400):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_prob_revolute")
                            y_axis_prob_rv = dpg.add_plot_axis(dpg.mvYAxis, label="Revolute Prob", tag="y_axis_prob_rv")
                            dpg.set_axis_limits("y_axis_prob_rv", -1, 2)
                            for mode in modes:
                                tag_ = f"{mode}_prob_revolute_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_prob_rv, tag=tag_)

                    with dpg.group():
                        with dpg.plot(label="Screw Probability vs Frame", height=250, width=400):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_prob_screw")
                            y_axis_prob_sc = dpg.add_plot_axis(dpg.mvYAxis, label="Screw Prob", tag="y_axis_prob_sc")
                            dpg.set_axis_limits("y_axis_prob_sc", -1, 2)
                            for mode in modes:
                                tag_ = f"{mode}_prob_screw_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_prob_sc, tag=tag_)

                        with dpg.plot(label="Ball Probability vs Frame", height=250, width=400):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="Frame", tag="x_axis_prob_ball")
                            y_axis_prob_ba = dpg.add_plot_axis(dpg.mvYAxis, label="Ball Prob", tag="y_axis_prob_ba")
                            dpg.set_axis_limits("y_axis_prob_ba", -1, 2)
                            for mode in modes:
                                tag_ = f"{mode}_prob_ball_series"
                                dpg.add_line_series([], [], label=mode, parent=y_axis_prob_ba, tag=tag_)

    dpg.create_viewport(title='DearPyGUI - Plots', width=1250, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()

def refresh_dearpygui_plots():
    global stop_refresh
    if stop_refresh:
        return

    all_finished = True
    for m in modes:
        limit = TOTAL_FRAMES_PER_MODE[m]
        if frame_count_per_mode[m] < limit:
            all_finished = False
            break

    if all_finished:
        return

    for mode in modes:
        vel_data = velocity_profile[mode]
        x_data = list(range(len(vel_data)))
        dpg.set_value(f"{mode}_vel_series", [x_data, vel_data])

        omega_data = angular_velocity_profile[mode]
        dpg.set_value(f"{mode}_omega_series", [x_data, omega_data])

        col_data = col_score_profile[mode]
        dpg.set_value(f"{mode}_col_series", [list(range(len(col_data))), col_data])
        cop_data = cop_score_profile[mode]
        dpg.set_value(f"{mode}_cop_series", [list(range(len(cop_data))), cop_data])
        rad_data = rad_score_profile[mode]
        dpg.set_value(f"{mode}_rad_series", [list(range(len(rad_data))), rad_data])
        zp_data = zp_score_profile[mode]
        dpg.set_value(f"{mode}_zp_series", [list(range(len(zp_data))), zp_data])

        err_data = error_profile[mode]
        dpg.set_value(f"{mode}_err_series", [list(range(len(err_data))), err_data])

        p_ = joint_prob_profile[mode]["prismatic"]
        dpg.set_value(f"{mode}_prob_prismatic_series", [list(range(len(p_))), p_])
        p_ = joint_prob_profile[mode]["planar"]
        dpg.set_value(f"{mode}_prob_planar_series", [list(range(len(p_))), p_])
        p_ = joint_prob_profile[mode]["revolute"]
        dpg.set_value(f"{mode}_prob_revolute_series", [list(range(len(p_))), p_])
        p_ = joint_prob_profile[mode]["screw"]
        dpg.set_value(f"{mode}_prob_screw_series", [list(range(len(p_))), p_])
        p_ = joint_prob_profile[mode]["ball"]
        dpg.set_value(f"{mode}_prob_ball_series", [list(range(len(p_))), p_])

def dearpygui_thread_main():
    setup_dearpygui()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        refresh_dearpygui_plots()

    dpg.destroy_context()

# =====================================================================
# main.py (entry point)
# =====================================================================

def main():
    th = threading.Thread(target=dearpygui_thread_main, daemon=True)
    th.start()
    ps.set_ground_plane_mode("none")
    ps.set_user_callback(callback)
    ps.show()

if __name__ == "__main__":
    main()
