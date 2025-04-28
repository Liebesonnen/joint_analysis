"""
Scoring functions for joint type estimation.
"""

import torch
import numpy as np


def super_gaussian(x, sigma, order):
    """
    Super-Gaussian function: exp(-(|x|/sigma)^order).

    Args:
        x (Tensor): Input values
        sigma (float): Width parameter of the Gaussian
        order (float): Order of the super-Gaussian

    Returns:
        Tensor: Super-Gaussian values
    """
    return torch.exp(- (torch.abs(x) / sigma) ** order)


def normalize_vector_torch(v, eps=1e-3):
    """
    Normalize 3D vectors in a PyTorch tensor; if norm < eps, return zero vector.

    Args:
        v (Tensor): Input vectors
        eps (float): Small value to avoid division by zero

    Returns:
        Tensor: Normalized vectors
    """
    norm_v = torch.norm(v, dim=-1)
    mask = (norm_v > eps)
    out = torch.zeros_like(v)
    out[mask] = v[mask] / norm_v[mask].unsqueeze(-1)
    return out


def estimate_rotation_matrix_batch(pcd_src: torch.Tensor, pcd_tar: torch.Tensor):
    """
    Estimate rotation matrices via SVD in batch.

    Args:
        pcd_src (Tensor): Source point clouds of shape (B,N,3)
        pcd_tar (Tensor): Target point clouds of shape (B,N,3)

    Returns:
        Tensor: Rotation matrices of shape (B,3,3)
    """
    assert pcd_src.shape == pcd_tar.shape

    # Center the point clouds
    pcd_src_centered = pcd_src - pcd_src.mean(dim=1, keepdim=True)
    pcd_tar_centered = pcd_tar - pcd_tar.mean(dim=1, keepdim=True)

    # Compute the cross-covariance matrix
    cov_matrix = torch.einsum('bni,bnj->bij', pcd_src_centered, pcd_tar_centered)

    # SVD decomposition
    U, S, Vt = torch.linalg.svd(cov_matrix, full_matrices=False)

    # Compute rotation matrix
    R = torch.einsum('bij,bjk->bik', Vt.transpose(-1, -2), U.transpose(-1, -2))

    # Handle reflections (det(R) = -1)
    det_r = torch.det(R)
    flip_mask = det_r < 0
    if flip_mask.any():
        Vt[flip_mask, :, -1] *= -1
        R = torch.einsum('bij,bjk->bik', Vt.transpose(-1, -2), U.transpose(-1, -2))

    return R


def se3_log_map_batch(transform_matrices: torch.Tensor):
    """
    SE(3) log map in batch. Converts SE(3) transformation matrices to twist coordinates.

    Args:
        transform_matrices (Tensor): Transformation matrices of shape (B,4,4)

    Returns:
        Tensor: Twist coordinates of shape (B,6) => [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
    """
    B = transform_matrices.shape[0]
    device = transform_matrices.device

    # Extract rotation and translation components
    R = transform_matrices[:, :3, :3]
    t = transform_matrices[:, :3, 3]

    # Calculate rotation angle from trace
    trace = torch.einsum('bii->b', R)
    tmp = (trace - 1.0) / 2.0
    tmp = torch.clamp(tmp, min=-1.0, max=1.0)
    theta = torch.acos(tmp).unsqueeze(-1)

    # Initialize variables
    omega = torch.zeros_like(R)
    log_R = torch.zeros_like(R)
    mask = theta.squeeze(-1) > 1e-3

    # Handle non-identity rotations
    if mask.any():
        theta_masked = theta[mask].squeeze(-1)
        skew_symmetric = (R[mask] - R[mask].transpose(-1, -2)) / (
                2 * torch.sin(theta_masked).view(-1, 1, 1)
        )
        omega[mask] = theta_masked.view(-1, 1, 1) * skew_symmetric
        log_R[mask] = skew_symmetric * theta_masked.view(-1, 1, 1)

    # Compute matrix to recover translation component of the twist
    A_inv = (torch.eye(3, device=device).repeat(B, 1, 1) - 0.5 * log_R)
    if mask.any():
        theta_sq = (theta[mask] ** 2).squeeze(-1)
        A_inv[mask] += (
                               (1 - theta[mask].squeeze(-1) / (2 * torch.tan(theta[mask].squeeze(-1) / 2))) / theta_sq
                       ).view(-1, 1, 1) * (log_R[mask] @ log_R[mask])

    # Recover translation component of the twist
    v = torch.einsum('bij,bj->bi', A_inv, t)

    # Extract rotation vector from skew-symmetric matrix
    # skew(omega) = [[0, -w_z, w_y], [w_z, 0, -w_x], [-w_y, w_x, 0]]
    rotvec = torch.stack([-omega[:, 1, 2], omega[:, 0, 2], -omega[:, 0, 1]], dim=-1)

    # Concatenate to form twist coordinates
    return torch.cat([v, rotvec], dim=-1)


def find_neighbors_batch(pcd_batch: torch.Tensor, num_neighbor_pts: int):
    """
    Find nearest neighbors for each point in the point clouds.

    Args:
        pcd_batch (Tensor): Point clouds of shape (B,N,3)
        num_neighbor_pts (int): Number of neighbors to find

    Returns:
        Tensor: Indices of neighbors of shape (B,N,num_neighbor_pts)
    """
    dist = torch.cdist(pcd_batch, pcd_batch, p=2.0)
    neighbor_indices = torch.topk(dist, k=num_neighbor_pts, dim=-1, largest=False).indices
    return neighbor_indices


# def compute_velocity_angular_one_step_3d(pts_prev, pts_curr, dt, num_neighbors=50):
#     """
#     Compute linear and angular velocities between two consecutive frames.
#
#     Args:
#         pts_prev (Tensor): Points in previous frame of shape (N,3)
#         pts_curr (Tensor): Points in current frame of shape (N,3)
#         dt (float): Time step
#         num_neighbors (int): Number of neighbors to use for local transformation estimation
#
#     Returns:
#         tuple: (linear_velocity, angular_velocity) each of shape (3,)
#     """
#     device = pts_prev.device
#
#     # Find nearest neighbors
#     neighbor_idx_prev = find_neighbors_batch(pts_prev.unsqueeze(0), num_neighbors)[0]  # (N,k)
#     neighbor_idx_curr = find_neighbors_batch(pts_curr.unsqueeze(0), num_neighbors)[0]  # (N,k)
#
#     # Get neighbor points
#     src_batch = pts_prev[neighbor_idx_prev]  # (N,k,3)
#     tar_batch = pts_curr[neighbor_idx_curr]  # (N,k,3)
#
#     # Estimate rotation matrices
#     R_2d = estimate_rotation_matrix_batch(src_batch, tar_batch)
#     c1_2d = src_batch.mean(dim=1)
#     c2_2d = tar_batch.mean(dim=1)
#     delta_p_2d = c2_2d - c1_2d
#
#     # Create SE(3) transformation matrices
#     eye_4 = torch.eye(4, device=device).unsqueeze(0).expand(pts_prev.shape[0], -1, -1).clone()
#     eye_4[:, :3, :3] = R_2d
#     eye_4[:, :3, 3] = delta_p_2d
#
#     # Convert to twist coordinates
#     se3_logs = se3_log_map_batch(eye_4)  # (N,6)
#
#     # Extract linear and angular velocities
#     trans_local = se3_logs[:, :3] / dt  # (N,3)
#     rot_local = se3_logs[:, 3:] / dt  # (N,3)
#
#     # Average over all points
#     v_mean = trans_local.mean(dim=0)  # (3,)
#     w_mean = rot_local.mean(dim=0)  # (3,)
#
#     return v_mean.cpu().numpy(), w_mean.cpu().numpy()
def compute_velocity_angular_one_step_3d(pts_prev, pts_curr, dt=0.1, num_neighbors=50):
    """计算两帧之间的线速度和角速度

    Args:
        pts_prev: 前一帧点云 (N, 3)
        pts_curr: 当前帧点云 (N, 3)
        dt: 时间步长
        num_neighbors: 用于计算角速度的邻居点数

    Returns:
        v_3d: 线速度 (3,)
        w_3d: 角速度 (3,)
    """
    # 检查是否是PyTorch张量
    if hasattr(pts_prev, 'device'):
        # PyTorch张量的处理逻辑
        device = pts_prev.device
        # ...其他PyTorch相关代码
    else:
        # NumPy数组的处理逻辑
        # 计算质心
        centroid_prev = np.mean(pts_prev, axis=0)
        centroid_curr = np.mean(pts_curr, axis=0)

        # 线速度
        v_3d = (centroid_curr - centroid_prev) / dt

        # 角速度计算
        # (这里可以实现纯NumPy版本的角速度计算)
        # 示例实现:
        w_3d = np.zeros(3)
        if pts_prev.shape[0] > 3:
            # 寻找偏离最大的点
            displacements = pts_curr - pts_prev
            disp_norms = np.linalg.norm(displacements, axis=1)
            max_indices = np.argsort(disp_norms)[-3:]

            omegas = []
            for idx in max_indices:
                # 找邻居
                dists = np.linalg.norm(pts_prev - pts_prev[idx], axis=1)
                neighbor_indices = np.argsort(dists)[:num_neighbors]

                # 局部运动分析
                local_prev = pts_prev[neighbor_indices]
                local_curr = pts_curr[neighbor_indices]

                # 计算旋转
                H = (local_prev - np.mean(local_prev, axis=0)).T @ (local_curr - np.mean(local_curr, axis=0))
                U, _, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T

                # 旋转矩阵转角速度
                omega = np.array([
                    R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]
                ]) / (2 * dt)

                omegas.append(omega)

            if omegas:
                w_3d = np.mean(omegas, axis=0)

    return v_3d, w_3d

def compute_position_average_3d(pts):
    """
    Compute the average 3D position of points.

    Args:
        pts (ndarray): Points of shape (N,3)

    Returns:
        ndarray: Average position of shape (3,)
    """
    return pts.mean(axis=0)  # shape (3,)


def compute_basic_scores(
        v_history, w_history, device='cuda',
        col_sigma=0.2, col_order=4.0,
        cop_sigma=0.2, cop_order=4.0,
        rad_sigma=0.2, rad_order=4.0,
        zp_sigma=0.2, zp_order=4.0,
        omega_thresh=1e-5,
        eps_vec=1e-3
):
    """
    Compute the four main scores for N points over T-1 frames:
      - col: Collinearity
      - cop: Coplanarity
      - rad: Radius consistency
      - zp: Zero pitch

    Args:
        v_history (ndarray): Linear velocity history of shape (T-1, N, 3)
        w_history (ndarray): Angular velocity history of shape (T-1, N, 3)
        device (str): Device to use for computation
        col_sigma (float): Width parameter for collinearity score
        col_order (float): Order parameter for collinearity score
        cop_sigma (float): Width parameter for coplanarity score
        cop_order (float): Order parameter for coplanarity score
        rad_sigma (float): Width parameter for radius consistency score
        rad_order (float): Order parameter for radius consistency score
        zp_sigma (float): Width parameter for zero pitch score
        zp_order (float): Order parameter for zero pitch score
        omega_thresh (float): Threshold for angular velocity magnitude
        eps_vec (float): Small value to avoid division by zero

    Returns:
        tuple: (col_score, cop_score, rad_score, zp_score), each shape (N,)
    """
    # Convert inputs to tensors
    v_t = torch.as_tensor(v_history, dtype=torch.float32, device=device)
    w_t = torch.as_tensor(w_history, dtype=torch.float32, device=device)
    Tm1, N = v_t.shape[0], v_t.shape[1]

    # Clean up small velocities to avoid numerical instability
    v_norm = torch.norm(v_t, dim=2)
    w_norm = torch.norm(w_t, dim=2)
    mask_v = (v_norm > eps_vec)
    mask_w = (w_norm > eps_vec)
    v_clean = torch.zeros_like(v_t)
    w_clean = torch.zeros_like(w_t)
    v_clean[mask_v] = v_t[mask_v]
    w_clean[mask_w] = w_t[mask_w]

    # ----------------------------
    # (A) Collinearity and Coplanarity scores
    # ----------------------------
    col_score = torch.ones(N, device=device)
    cop_score = torch.ones(N, device=device)

    if Tm1 >= 2:
        v_unit = normalize_vector_torch(v_clean)
        v_unit_all = v_unit.permute(1, 0, 2).contiguous()

        if Tm1 >= 3:
            # Compute SVD of unit velocities
            U, S, _ = torch.linalg.svd(v_unit_all, full_matrices=False)
            s1 = S[:, 0]
            s2 = S[:, 1]
            s3 = S[:, 2]

            # Compute ratios for collinearity and coplanarity
            eps_svd = 1e-6
            mask_svd = (s1 > eps_svd)
            ratio_col = torch.zeros_like(s1)
            ratio_cop = torch.zeros_like(s1)
            ratio_col[mask_svd] = s2[mask_svd] / s1[mask_svd]
            ratio_cop[mask_svd] = s3[mask_svd] / s1[mask_svd]

            # Apply super-Gaussian to get scores
            col_score = super_gaussian(ratio_col, col_sigma, col_order)
            cop_score = super_gaussian(ratio_cop, cop_sigma, cop_order)
        else:
            # Special case with only 2 frames
            U, S_, _ = torch.linalg.svd(v_unit_all, full_matrices=False)
            s1 = S_[:, 0]
            s2 = S_[:, 1] if S_.size(1) > 1 else torch.zeros_like(s1)

            ratio_col = torch.zeros_like(s1)
            mask_svd = (s1 > 1e-6)
            ratio_col[mask_svd] = s2[mask_svd] / s1[mask_svd]
            col_score = super_gaussian(ratio_col, col_sigma, col_order)

    # ----------------------------
    # (B) Radius consistency score
    # ----------------------------
    rad_score = torch.zeros(N, device=device)

    if Tm1 > 0:
        # Calculate radius r = v / w for frames where w is significant
        v_mag = torch.norm(v_clean, dim=2)  # (T-1, N)
        w_mag = torch.norm(w_clean, dim=2)  # (T-1, N)

        # Identify frames with significant angular velocity
        valid_mask = (w_mag >= omega_thresh)
        r_mat = torch.zeros_like(v_mag)
        r_mat[valid_mask] = v_mag[valid_mask] / w_mag[valid_mask]

        # Compute radius consistency score for each point
        final_scores = torch.zeros(N, device=device)

        for i in range(N):
            # Find frames with valid angular velocity for this point
            valid_frames_i = valid_mask[:, i]  # shape (T-1,)
            # Get radius values for those frames
            r_vals_i = r_mat[valid_frames_i, i]

            # Skip points with insufficient valid frames
            if r_vals_i.numel() <= 1:
                final_scores[i] = 0.0
            else:
                # Compute variance of radius and convert to score
                var_r_i = torch.var(r_vals_i, unbiased=False)
                final_scores[i] = super_gaussian(var_r_i, rad_sigma, rad_order)

        rad_score = final_scores

    # ----------------------------
    # (C) Zero pitch score
    # ----------------------------
    zp_score = torch.ones(N, device=device)

    if Tm1 > 0:
        # Normalize velocity vectors
        v_u = normalize_vector_torch(v_clean)  # (T-1, N, 3)
        w_u = normalize_vector_torch(w_clean)  # (T-1, N, 3)

        # Compute absolute dot product between v and w
        dot_val = torch.sum(v_u * w_u, dim=2).abs()  # (T-1, N)

        # Average over time and apply super-Gaussian
        mean_dot = torch.mean(dot_val, dim=0)  # (N,)
        zp_score = super_gaussian(mean_dot, zp_sigma, zp_order)

    return col_score, cop_score, rad_score, zp_score


def compute_joint_probability_new(col, cop, rad, zp, joint_type="prismatic", prob_sigma=0.1, prob_order=4.0):
    """
    Compute joint probability (0~1) based on the four fundamental scores.

    Args:
        col (Tensor): Collinearity scores of shape (N,)
        cop (Tensor): Coplanarity scores of shape (N,)
        rad (Tensor): Radius consistency scores of shape (N,)
        zp (Tensor): Zero pitch scores of shape (N,)
        joint_type (str): Type of joint to compute probability for
        prob_sigma (float): Width parameter for probability function
        prob_order (float): Order parameter for probability function

    Returns:
        Tensor: Joint probability scores of shape (N,)
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


def compute_motion_salience_batch_neighborhood(all_points_history, device='cuda', k=10):
    """
    Compute motion salience using neighbor-based average displacements.

    Args:
        all_points_history (ndarray): Point history of shape (T,N,3)
        device (str): Device to use for computation
        k (int): Number of neighbors to consider

    Returns:
        Tensor: Motion salience scores of shape (N,)
    """
    pts = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    T, N, _ = pts.shape

    if T < 2:
        return torch.zeros(N, device=device)

    B = T - 1
    pts_t = pts[:-1]
    pts_tp1 = pts[1:]

    # Find neighbors for each point
    neighbor_idx = find_neighbors_batch(pts_t, k)

    # Compute total displacement
    sum_disp = torch.zeros(N, device=device)
    for b in range(B):
        p_t = pts_t[b]
        p_tp1 = pts_tp1[b]
        nb_idx = neighbor_idx[b]

        # Get neighbor points
        p_t_nb = p_t[nb_idx]
        p_tp1_nb = p_tp1[nb_idx]

        # Compute displacement and its magnitude
        disp = p_tp1_nb - p_t_nb
        disp_mean = disp.mean(dim=1)
        mag = torch.norm(disp_mean, dim=1)

        # Accumulate displacements
        sum_disp += mag

    return sum_disp


def compute_motion_salience_batch(all_points_history, neighbor_k=400, device='cuda'):
    """
    Wrapper for neighbor-based motion salience with adjustable k.

    Args:
        all_points_history (ndarray): Point history of shape (T,N,3)
        neighbor_k (int): Number of neighbors to consider
        device (str): Device to use for computation

    Returns:
        Tensor: Motion salience scores of shape (N,)
    """
    pts = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    T, N, _ = pts.shape

    if T < 2:
        return torch.zeros(N, device=device)

    return compute_motion_salience_batch_neighborhood(all_points_history, device=device, k=neighbor_k)