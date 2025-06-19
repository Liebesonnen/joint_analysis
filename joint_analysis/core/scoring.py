"""
Scoring functions for joint type estimation.

This module provides comprehensive scoring functions for estimating different types of mechanical joints
(prismatic, revolute, planar, screw, ball) based on motion analysis of point clouds. The scoring is based
on four fundamental geometric properties:
1. Collinearity - how well motion vectors align along a single direction
2. Coplanarity - how well motion vectors lie within a single plane
3. Radius consistency - how consistent the radius of rotation is over time
4. Zero pitch - how perpendicular linear and angular velocities are

The module uses PyTorch for efficient batch processing and implements various mathematical operations
including SE(3) transformations, SVD analysis, and super-Gaussian scoring functions.
"""

import torch
import numpy as np
from .type import JointType, ExpectedScore, JointExpectedScores
from .utils import get_data_path
from robot_utils.py.filesystem import get_validate_file
from omegaconf import OmegaConf

# Load pre-computed expected scores for different joint types from configuration file
joint_exp_scores: JointExpectedScores = JointExpectedScores.load(
    get_validate_file(get_data_path() / "joint_expected_scores.yaml")
)


def super_gaussian(x: torch.Tensor, sigma: float, order: float) -> torch.Tensor:
    """
    Compute super-Gaussian function: exp(-(|x|/sigma)^order).

    A super-Gaussian is a generalization of the Gaussian function that allows for different
    tail behaviors based on the order parameter. When order=2, it reduces to a standard Gaussian.
    Higher orders create sharper peaks and faster decay, while lower orders create broader distributions.

    This function is used throughout the scoring system to convert error measures into probability-like
    scores where smaller errors yield higher scores approaching 1.0.

    Args:
        x (torch.Tensor): Input values to evaluate the super-Gaussian on
        sigma (float): Width parameter controlling the spread of the function - larger values
                      create wider distributions
        order (float): Order parameter controlling the tail behavior - higher values create
                      sharper peaks and faster decay

    Returns:
        torch.Tensor: Super-Gaussian values in range [0, 1], same shape as input x
    """
    return torch.exp(- (torch.abs(x) / sigma) ** order)


def normalize_vector_torch(v: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Normalize 3D vectors in a PyTorch tensor; if norm < eps, return zero vector.

    This function safely normalizes vectors by checking if their magnitude is above a threshold
    to avoid division by zero or near-zero values that could cause numerical instability.
    Vectors with magnitude below the threshold are set to zero vectors.

    Args:
        v (torch.Tensor): Input vectors of shape (..., 3) where the last dimension represents
                         3D coordinates (x, y, z)
        eps (float): Small threshold value to avoid division by zero. Vectors with magnitude
                    below this value are set to zero vectors

    Returns:
        torch.Tensor: Normalized vectors of same shape as input. Unit vectors for input vectors
                     with magnitude >= eps, zero vectors otherwise
    """
    norm_v: torch.Tensor = torch.norm(v, dim=-1)  # Compute magnitude of each vector
    mask: torch.Tensor = (norm_v > eps)  # Boolean mask for vectors above threshold
    out: torch.Tensor = torch.zeros_like(v)  # Initialize output with zeros

    # Only normalize vectors that pass the threshold check
    out[mask] = v[mask] / norm_v[mask].unsqueeze(-1)
    return out


def estimate_rotation_matrix_batch(pcd_src: torch.Tensor, pcd_tar: torch.Tensor) -> torch.Tensor:
    """
    Estimate rotation matrices via SVD in batch using the Kabsch algorithm.

    This function implements the Kabsch algorithm to find the optimal rotation matrix that
    best aligns two sets of corresponding points. The algorithm works by:
    1. Centering both point clouds around their centroids
    2. Computing the cross-covariance matrix between the centered clouds
    3. Using SVD to decompose the covariance matrix
    4. Constructing the rotation matrix from the SVD components
    5. Handling reflection cases where determinant is negative

    This is commonly used in robotics for pose estimation and point cloud registration.

    Args:
        pcd_src (torch.Tensor): Source point clouds of shape (B, N, 3) where B is batch size,
                               N is number of points, and 3 represents x,y,z coordinates
        pcd_tar (torch.Tensor): Target point clouds of shape (B, N, 3) corresponding to
                               transformed versions of the source clouds

    Returns:
        torch.Tensor: Rotation matrices of shape (B, 3, 3) that best align source to target
                     point clouds for each batch element
    """
    assert pcd_src.shape == pcd_tar.shape, "Source and target point clouds must have same shape"

    # Center both point clouds by subtracting their respective centroids
    # This removes translation effects and focuses on rotation estimation
    pcd_src_centered: torch.Tensor = pcd_src - pcd_src.mean(dim=1, keepdim=True)
    pcd_tar_centered: torch.Tensor = pcd_tar - pcd_tar.mean(dim=1, keepdim=True)

    # Compute the cross-covariance matrix H = sum(src_i * tar_i^T)
    # This matrix captures the correlation between corresponding points
    cov_matrix: torch.Tensor = torch.einsum('bni,bnj->bij', pcd_src_centered, pcd_tar_centered)

    # Perform SVD decomposition: H = U * S * V^T
    # The optimal rotation is constructed from U and V matrices
    U: torch.Tensor
    S: torch.Tensor  # Singular values (not used in rotation calculation)
    Vt: torch.Tensor  # V transpose matrix
    U, S, Vt = torch.linalg.svd(cov_matrix, full_matrices=False)

    # Compute rotation matrix R = V * U^T
    R: torch.Tensor = torch.einsum('bij,bjk->bik', Vt.transpose(-1, -2), U.transpose(-1, -2))

    # Handle reflection cases where det(R) = -1
    # In these cases, we need to flip the last column of V to ensure proper rotation
    det_r: torch.Tensor = torch.det(R)
    flip_mask: torch.Tensor = det_r < 0

    if flip_mask.any():
        # Flip the sign of the last column of Vt for reflection cases
        Vt[flip_mask, :, -1] *= -1
        # Recompute rotation matrix with corrected V
        R = torch.einsum('bij,bjk->bik', Vt.transpose(-1, -2), U.transpose(-1, -2))

    return R


def se3_log_map_batch(transform_matrices: torch.Tensor) -> torch.Tensor:
    """
    SE(3) log map in batch. Converts SE(3) transformation matrices to twist coordinates.

    The SE(3) log map converts rigid body transformations (represented as 4x4 homogeneous
    transformation matrices) into their corresponding twist coordinates in the Lie algebra se(3).

    A twist coordinate is a 6-dimensional vector [v, ω] where:
    - v ∈ ℝ³ represents the linear velocity component
    - ω ∈ ℝ³ represents the angular velocity component (rotation vector)

    This transformation is fundamental in robotics for:
    - Motion analysis and control
    - Trajectory planning in joint space
    - Screw theory applications
    - Converting between different motion representations

    The algorithm implements the matrix logarithm for SE(3):
    1. Extract rotation matrix R and translation vector t from transformation matrix
    2. Compute rotation angle θ from the trace of R
    3. Handle special cases (identity rotation vs. general rotation)
    4. Compute the angular velocity vector ω from the skew-symmetric part of log(R)
    5. Recover the linear velocity v by inverting the SE(3) exponential map formula

    Args:
        transform_matrices (torch.Tensor): Transformation matrices of shape (B, 4, 4) where
                                         B is batch size. Each matrix is a homogeneous
                                         transformation matrix with rotation R (3x3) in top-left
                                         and translation t (3x1) in top-right

    Returns:
        torch.Tensor: Twist coordinates of shape (B, 6) where each row contains
                     [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
                     representing linear and angular velocity components
    """
    B: int = transform_matrices.shape[0]  # Batch size
    device: torch.device = transform_matrices.device

    # Extract rotation (top-left 3x3) and translation (top-right 3x1) components
    R: torch.Tensor = transform_matrices[:, :3, :3]  # Shape: (B, 3, 3)
    t: torch.Tensor = transform_matrices[:, :3, 3]  # Shape: (B, 3)

    # Calculate rotation angle θ from trace of rotation matrix
    # Using: cos(θ) = (trace(R) - 1) / 2
    trace: torch.Tensor = torch.einsum('bii->b', R)  # Sum of diagonal elements
    tmp: torch.Tensor = (trace - 1.0) / 2.0
    tmp = torch.clamp(tmp, min=-1.0, max=1.0)  # Clamp to valid acos domain
    theta: torch.Tensor = torch.acos(tmp).unsqueeze(-1)  # Shape: (B, 1)

    # Initialize matrices for batch processing
    omega: torch.Tensor = torch.zeros_like(R)  # Will store skew-symmetric matrices
    log_R: torch.Tensor = torch.zeros_like(R)  # Will store log(R) matrices

    # Identify non-identity rotations (angle > threshold)
    mask: torch.Tensor = theta.squeeze(-1) > 1e-3

    # Handle non-identity rotations using Rodrigues' rotation formula
    if mask.any():
        theta_masked: torch.Tensor = theta[mask].squeeze(-1)

        # Compute skew-symmetric matrix from (R - R^T) / (2*sin(θ))
        # This extracts the axis of rotation
        skew_symmetric: torch.Tensor = (R[mask] - R[mask].transpose(-1, -2)) / (
                2 * torch.sin(theta_masked).view(-1, 1, 1)
        )

        # Scale by angle to get ω (angular velocity vector in matrix form)
        omega[mask] = theta_masked.view(-1, 1, 1) * skew_symmetric
        log_R[mask] = skew_symmetric * theta_masked.view(-1, 1, 1)

    # Compute matrix A^(-1) to recover translation component of twist
    # This inverts the SE(3) exponential map formula: exp([ω]_×) * v = t
    A_inv: torch.Tensor = (torch.eye(3, device=device).repeat(B, 1, 1) - 0.5 * log_R)

    # For non-identity rotations, add correction term
    if mask.any():
        theta_sq: torch.Tensor = (theta[mask] ** 2).squeeze(-1)

        # Correction factor: (1 - θ/(2*tan(θ/2))) / θ²
        correction_factor: torch.Tensor = (
                                                  1 - theta[mask].squeeze(-1) / (
                                                      2 * torch.tan(theta[mask].squeeze(-1) / 2))
                                          ) / theta_sq

        # Add correction term: correction_factor * (log_R)²
        A_inv[mask] += correction_factor.view(-1, 1, 1) * (log_R[mask] @ log_R[mask])

    # Recover linear velocity component: v = A^(-1) * t
    v: torch.Tensor = torch.einsum('bij,bj->bi', A_inv, t)

    # Extract rotation vector from skew-symmetric matrix ω
    # For skew-symmetric matrix [[0, -w_z, w_y], [w_z, 0, -w_x], [-w_y, w_x, 0]]
    # the rotation vector is [w_x, w_y, w_z]
    rotvec: torch.Tensor = torch.stack([
        -omega[:, 1, 2],  # w_x from -ω[1,2]
        omega[:, 0, 2],  # w_y from ω[0,2]
        -omega[:, 0, 1]  # w_z from -ω[0,1]
    ], dim=-1)

    # Concatenate linear and angular components to form twist coordinates
    return torch.cat([v, rotvec], dim=-1)


def find_neighbors_batch(pcd_batch: torch.Tensor, num_neighbor_pts: int) -> torch.Tensor:
    """
    Find nearest neighbors for each point in the point clouds using Euclidean distance.

    This function computes k-nearest neighbors for each point in each point cloud within a batch.
    It uses pairwise Euclidean distances and returns the indices of the closest points.
    This is commonly used for:
    - Local feature computation
    - Surface normal estimation
    - Motion smoothing and filtering
    - Neighborhood-based analysis

    Args:
        pcd_batch (torch.Tensor): Point clouds of shape (B, N, 3) where B is batch size,
                                  N is number of points, and 3 represents coordinates
        num_neighbor_pts (int): Number of nearest neighbors to find for each point
                               (including the point itself)

    Returns:
        torch.Tensor: Indices of neighbors of shape (B, N, num_neighbor_pts) where each
                     entry contains the indices of the k nearest neighbors for that point
    """
    # Compute pairwise distances between all points using L2 norm
    # dist[b,i,j] = ||point_i - point_j||_2 for batch b
    dist: torch.Tensor = torch.cdist(pcd_batch, pcd_batch, p=2.0)

    # Find k smallest distances (nearest neighbors) for each point
    # largest=False ensures we get the smallest distances
    neighbor_indices: torch.Tensor = torch.topk(
        dist, k=num_neighbor_pts, dim=-1, largest=False
    ).indices

    return neighbor_indices


def compute_position_average_3d(pts: np.ndarray) -> np.ndarray:
    """
    Compute the average 3D position (centroid) of a set of points.

    This is a simple utility function that calculates the geometric center of a point cloud
    by averaging the coordinates along each axis. The centroid is often used as a
    representative point for the entire point cloud.

    Args:
        pts (np.ndarray): Points of shape (N, 3) where N is number of points and
                         3 represents x, y, z coordinates

    Returns:
        np.ndarray: Average position of shape (3,) representing the centroid coordinates
    """
    return pts.mean(axis=0)  # Average along the points dimension


def compute_basic_scores(
        v_history: np.ndarray,
        w_history: np.ndarray,
        device: str = 'cuda',
        col_sigma: float = 0.3,
        col_order: float = 3.0,
        cop_sigma: float = 0.3,
        cop_order: float = 3.0,
        rad_sigma: float = 0.3,
        rad_order: float = 3.0,
        zp_sigma: float = 0.3,
        zp_order: float = 3.0,
        omega_thresh: float = 1e-6,
        eps_vec: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the four fundamental scores for joint type classification.

    This function analyzes motion trajectories to compute four key geometric properties
    that characterize different types of mechanical joints:

    1. **Collinearity Score**: Measures how well velocity vectors align along a single direction.
       High for prismatic joints (pure translation), low for rotational motions.

    2. **Coplanarity Score**: Measures how well velocity vectors lie within a single plane.
       High for planar and revolute joints, low for 3D motions like ball joints.

    3. **Radius Consistency Score**: Measures consistency of the rotation radius over time.
       High for revolute joints with constant radius, low for variable or no rotation.

    4. **Zero Pitch Score**: Measures how perpendicular linear and angular velocities are.
       High when v ⟂ ω (pure rotation), low when v ∥ ω (screw motion).

    The algorithm uses Singular Value Decomposition (SVD) to analyze the structure of
    velocity vectors and applies super-Gaussian functions to convert geometric measures
    into normalized scores.

    Args:
        v_history (np.ndarray): Linear velocity history of shape (T-1, N, 3) where T is
                               number of time steps, N is number of points
        w_history (np.ndarray): Angular velocity history of shape (T-1, N, 3)
        device (str): PyTorch device to use for computation ('cuda' or 'cpu')
        col_sigma (float): Width parameter for collinearity score super-Gaussian
        col_order (float): Order parameter for collinearity score super-Gaussian
        cop_sigma (float): Width parameter for coplanarity score super-Gaussian
        cop_order (float): Order parameter for coplanarity score super-Gaussian
        rad_sigma (float): Width parameter for radius consistency score super-Gaussian
        rad_order (float): Order parameter for radius consistency score super-Gaussian
        zp_sigma (float): Width parameter for zero pitch score super-Gaussian
        zp_order (float): Order parameter for zero pitch score super-Gaussian
        omega_thresh (float): Threshold for meaningful angular velocity magnitude
        eps_vec (float): Small value to avoid division by zero in vector operations

    Returns:
        tuple: (col_score, cop_score, rad_score, zp_score) where each is a tensor of shape (N,)
               containing scores in range [0, 1] for each point
    """
    # Convert input arrays to PyTorch tensors for GPU acceleration
    v_t: torch.Tensor = torch.as_tensor(v_history, dtype=torch.float32, device=device)
    w_t: torch.Tensor = torch.as_tensor(w_history, dtype=torch.float32, device=device)
    Tm1: int = v_t.shape[0]  # Number of time intervals (T-1)
    N: int = v_t.shape[1]  # Number of points

    # Clean up small velocities to avoid numerical instability
    # This prevents noise from dominating the analysis when velocities are very small
    v_norm: torch.Tensor = torch.norm(v_t, dim=2)  # Magnitude of linear velocities
    w_norm: torch.Tensor = torch.norm(w_t, dim=2)  # Magnitude of angular velocities

    # Create masks for meaningful velocities above threshold
    mask_v: torch.Tensor = (v_norm > eps_vec)
    mask_w: torch.Tensor = (w_norm > eps_vec)

    # Initialize cleaned velocity arrays
    v_clean: torch.Tensor = torch.zeros_like(v_t)
    w_clean: torch.Tensor = torch.zeros_like(w_t)
    v_clean[mask_v] = v_t[mask_v]  # Keep only significant linear velocities
    w_clean[mask_w] = w_t[mask_w]  # Keep only significant angular velocities

    # ----------------------------
    # (A) Collinearity and Coplanarity Scores
    # ----------------------------
    # These scores analyze the geometric structure of velocity vectors using SVD

    # Initialize scores to perfect values (1.0 = ideal condition)
    col_score: torch.Tensor = torch.ones(N, device=device)
    cop_score: torch.Tensor = torch.ones(N, device=device)

    if Tm1 >= 2:  # Need at least 2 time steps for meaningful analysis
        # Normalize velocity vectors to focus on direction rather than magnitude
        v_unit: torch.Tensor = normalize_vector_torch(v_clean)
        # Reshape for SVD: (N, T-1, 3) - each point has a matrix of unit velocity vectors
        v_unit_all: torch.Tensor = v_unit.permute(1, 0, 2).contiguous()

        if Tm1 >= 3:  # Full SVD analysis requires at least 3 vectors
            # Perform SVD on velocity vector matrices for each point
            # U: left singular vectors, S: singular values, V: right singular vectors
            U: torch.Tensor
            S: torch.Tensor
            _: torch.Tensor  # V not used
            U, S, _ = torch.linalg.svd(v_unit_all, full_matrices=False)

            # Extract singular values (sorted in descending order)
            s1: torch.Tensor = S[:, 0]  # Largest singular value
            s2: torch.Tensor = S[:, 1]  # Second largest singular value
            s3: torch.Tensor = S[:, 2]  # Smallest singular value

            # Compute ratios for collinearity and coplanarity analysis
            eps_svd: float = 1e-6
            mask_svd: torch.Tensor = (s1 > eps_svd)  # Valid SVD decomposition

            # Initialize ratio arrays
            ratio_col: torch.Tensor = torch.zeros_like(s1)
            ratio_cop: torch.Tensor = torch.zeros_like(s1)

            # Collinearity ratio: s2/s1 (how much the second direction contributes)
            # Low ratio → vectors are collinear (align along single direction)
            ratio_col[mask_svd] = s2[mask_svd] / s1[mask_svd]

            # Coplanarity ratio: s3/s1 (how much the third direction contributes)
            # Low ratio → vectors are coplanar (lie within a plane)
            ratio_cop[mask_svd] = s3[mask_svd] / s1[mask_svd]

            # Convert ratios to scores using super-Gaussian functions
            # Lower ratios yield higher scores (approaching 1.0)
            col_score = super_gaussian(ratio_col, col_sigma, col_order)
            cop_score = super_gaussian(ratio_cop, cop_sigma, cop_order)

        else:  # Special case with only 2 time steps (T-1 = 2)
            # Limited SVD analysis with only 2 velocity vectors per point
            U, S_, _ = torch.linalg.svd(v_unit_all, full_matrices=False)
            s1 = S_[:, 0]
            s2 = S_[:, 1] if S_.size(1) > 1 else torch.zeros_like(s1)

            ratio_col = torch.zeros_like(s1)
            mask_svd = (s1 > 1e-6)
            ratio_col[mask_svd] = s2[mask_svd] / s1[mask_svd]
            col_score = super_gaussian(ratio_col, col_sigma, col_order)
            # Note: coplanarity analysis requires at least 3 vectors, so cop_score remains 1.0

    # ----------------------------
    # (B) Radius Consistency Score
    # ----------------------------
    # This score measures how consistent the rotation radius is over time
    # High consistency indicates revolute joint motion with fixed rotation center

    rad_score: torch.Tensor = torch.zeros(N, device=device)

    if Tm1 > 0:
        # Compute instantaneous radius for each time step: r = |v| / |ω|
        # This relationship comes from the kinematic equation for circular motion
        v_mag: torch.Tensor = torch.norm(v_clean, dim=2)  # Shape: (T-1, N)
        w_mag: torch.Tensor = torch.norm(w_clean, dim=2)  # Shape: (T-1, N)

        # Identify time steps with meaningful angular velocity
        valid_mask: torch.Tensor = (w_mag >= omega_thresh)
        r_mat: torch.Tensor = torch.zeros_like(v_mag)

        # Compute radius only where angular velocity is significant
        r_mat[valid_mask] = v_mag[valid_mask] / w_mag[valid_mask]

        # Analyze radius consistency for each point individually
        final_scores: torch.Tensor = torch.zeros(N, device=device)

        for i in range(N):
            # Find frames where this point has valid angular velocity
            valid_frames_i: torch.Tensor = valid_mask[:, i]  # Shape: (T-1,)
            r_vals_i: torch.Tensor = r_mat[valid_frames_i, i]  # Valid radius values

            # Need at least 2 valid measurements for variance calculation
            if r_vals_i.numel() <= 1:
                final_scores[i] = 0.0  # Insufficient data
            else:
                # Compute variance of radius values
                # Low variance → consistent radius → high score for revolute joints
                var_r_i: torch.Tensor = torch.var(r_vals_i, unbiased=False)
                final_scores[i] = super_gaussian(var_r_i, rad_sigma, rad_order)

        rad_score = final_scores

    # ----------------------------
    # (C) Zero Pitch Score
    # ----------------------------
    # This score measures how perpendicular linear and angular velocities are
    # High score indicates pure rotation (v ⟂ ω), low score indicates screw motion (v ∥ ω)

    zp_score: torch.Tensor = torch.ones(N, device=device)

    if Tm1 > 0:
        # Normalize velocity vectors to compare directions only
        v_u: torch.Tensor = normalize_vector_torch(v_clean)  # Shape: (T-1, N, 3)
        w_u: torch.Tensor = normalize_vector_torch(w_clean)  # Shape: (T-1, N, 3)

        # Compute absolute dot product between normalized linear and angular velocities
        # |v·ω| = 0 for perpendicular vectors (pure rotation)
        # |v·ω| = 1 for parallel vectors (pure translation or screw motion)
        dot_val: torch.Tensor = torch.sum(v_u * w_u, dim=2).abs()  # Shape: (T-1, N)

        # Average over time to get overall alignment measure
        mean_dot: torch.Tensor = torch.mean(dot_val, dim=0)  # Shape: (N,)

        # Convert to score: lower dot product → higher score
        zp_score = super_gaussian(mean_dot, zp_sigma, zp_order)

    return col_score, cop_score, rad_score, zp_score


def compute_joint_probability(
        col: torch.Tensor,
        cop: torch.Tensor,
        rad: torch.Tensor,
        zp: torch.Tensor,
        joint_exp_score: JointExpectedScores,
        joint_type: JointType = JointType.Prismatic,
        sigma: float = None,
        order: float = None,
        prob_sigma: float = 0.1,
        prob_order: float = 4.0,
        prismatic_sigma: float = 0.08,
        prismatic_order: float = 5.0,
        planar_sigma: float = 0.12,
        planar_order: float = 4.0,
        revolute_sigma: float = 0.08,
        revolute_order: float = 5.0,
        screw_sigma: float = 0.15,
        screw_order: float = 4.0,
        ball_sigma: float = 0.12,
        ball_order: float = 4.0
) -> torch.Tensor:
    """
    Compute joint probability based on the four fundamental scores and expected score patterns.

    This function evaluates how well the computed scores match the expected patterns for
    different joint types. Each joint type has a characteristic "signature" in the 4D
    score space:

    - **Prismatic**: col≈1, cop≈0, rad≈0, zp≈1 (linear motion along fixed axis)
    - **Revolute**: col≈0, cop≈1, rad≈1, zp≈1 (rotation around fixed axis)
    - **Planar**: col≈0, cop≈1, rad≈0, zp≈1 (motion constrained to plane)
    - **Screw**: col≈0, cop≈0, rad≈1, zp≈0 (rotation + translation along axis)
    - **Ball**: col≈0, cop≈0, rad≈1, zp≈1 (3DOF rotation around fixed point)

    The probability is computed by measuring the Euclidean distance from the observed
    scores to the expected pattern and converting it to a probability using a super-Gaussian.

    Args:
        col (torch.Tensor): Collinearity scores of shape (N,)
        cop (torch.Tensor): Coplanarity scores of shape (N,)
        rad (torch.Tensor): Radius consistency scores of shape (N,)
        zp (torch.Tensor): Zero pitch scores of shape (N,)
        joint_exp_score (JointExpectedScores): Configuration object containing expected
                                             score patterns for all joint types
        joint_type (JointType): Type of joint to compute probability for
        sigma (float): Unused parameter (overridden by joint-specific values)
        order (float): Unused parameter (overridden by joint-specific values)
        prob_sigma (float): Unused parameter
        prob_order (float): Unused parameter
        *_sigma/*_order: Joint-specific parameters for super-Gaussian probability mapping

    Returns:
        torch.Tensor: Joint probability scores of shape (N,) in range [0, 1] where
                     higher values indicate better match to the specified joint type

    Raises:
        KeyError: If the specified joint_type is not supported
    """
    if joint_type not in JointType:
        raise KeyError(f"joint type {joint_type} is not supported")

    # Retrieve expected scores for the specified joint type from configuration
    # The joint_exp_score object contains pre-computed ideal score patterns
    es_field = {
        JointType.Prismatic: joint_exp_score.prismatic,
        JointType.Revolute: joint_exp_score.revolute,
        JointType.Planar: joint_exp_score.planar,
        JointType.Screw: joint_exp_score.screw,
        JointType.Ball: joint_exp_score.ball,
    }[joint_type]

    # Convert OmegaConf Field to regular Python dictionary for easier access
    es: dict = OmegaConf.to_container(es_field)

    # Get sigma parameter for the specified joint type
    # These control the width of the probability distribution
    sigma: float = {
        JointType.Prismatic: prismatic_sigma,
        JointType.Revolute: revolute_sigma,
        JointType.Planar: planar_sigma,
        JointType.Screw: screw_sigma,
        JointType.Ball: ball_sigma,
    }[joint_type]

    # Get order parameter for the specified joint type
    # These control the sharpness of the probability distribution
    order: float = {
        JointType.Prismatic: prismatic_order,
        JointType.Revolute: revolute_order,
        JointType.Planar: planar_order,
        JointType.Screw: screw_order,
        JointType.Ball: ball_order,
    }[joint_type]

    # Compute squared Euclidean distance between observed and expected scores
    # Each component represents deviation from the ideal pattern for this joint type
    e: torch.Tensor = (
                              (col - es['col']) ** 2 +  # Collinearity deviation
                              (cop - es['cop']) ** 2 +  # Coplanarity deviation
                              (rad - es['radius']) ** 2 +  # Radius consistency deviation
                              (zp - es['zp']) ** 2  # Zero pitch deviation
                      ) / 4  # Normalize by number of components

    # Convert error distance to probability using super-Gaussian
    # Smaller errors yield higher probabilities (approaching 1.0)
    return super_gaussian(e, sigma, order)


def compute_motion_salience_batch_neighborhood(
        all_points_history: np.ndarray,
        device: str = 'cuda',
        k: int = 10
) -> torch.Tensor:
    """
    Compute motion salience using neighbor-based average displacements.

    Motion salience measures how much each point moves relative to its local neighborhood.
    Points that move significantly compared to their neighbors are considered more salient
    for motion analysis. This is useful for:
    - Identifying moving parts vs. static background
    - Focusing analysis on actively articulating regions
    - Filtering out noise and stationary components

    The algorithm works by:
    1. Finding k nearest neighbors for each point at each time step
    2. Computing average displacement of each neighborhood
    3. Accumulating displacement magnitudes over time
    4. Higher accumulated displacement = higher motion salience

    Args:
        all_points_history (np.ndarray): Point history of shape (T, N, 3) where T is
                                        number of time steps, N is number of points
        device (str): PyTorch device for computation ('cuda' or 'cpu')
        k (int): Number of nearest neighbors to consider for each point

    Returns:
        torch.Tensor: Motion salience scores of shape (N,) where higher values indicate
                     points with more significant motion relative to their neighborhoods
    """
    # Convert to PyTorch tensor for GPU acceleration
    pts: torch.Tensor = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    T: int = pts.shape[0]  # Number of time steps
    N: int = pts.shape[1]  # Number of points

    # Need at least 2 time steps to compute motion
    if T < 2:
        return torch.zeros(N, device=device)

    B: int = T - 1  # Number of time intervals
    pts_t: torch.Tensor = pts[:-1]  # Points at time t (shape: B, N, 3)
    pts_tp1: torch.Tensor = pts[1:]  # Points at time t+1 (shape: B, N, 3)

    # Find k nearest neighbors for each point at each time step
    # This creates neighborhoods for local motion analysis
    neighbor_idx: torch.Tensor = find_neighbors_batch(pts_t, k)

    # Accumulate displacement magnitudes over all time intervals
    sum_disp: torch.Tensor = torch.zeros(N, device=device)

    for b in range(B):  # For each time interval
        # Get point positions at current and next time steps
        p_t: torch.Tensor = pts_t[b]  # Shape: (N, 3)
        p_tp1: torch.Tensor = pts_tp1[b]  # Shape: (N, 3)
        nb_idx: torch.Tensor = neighbor_idx[b]  # Shape: (N, k)

        # Extract neighbor positions for both time steps
        p_t_nb: torch.Tensor = p_t[nb_idx]  # Shape: (N, k, 3)
        p_tp1_nb: torch.Tensor = p_tp1[nb_idx]  # Shape: (N, k, 3)

        # Compute displacement vectors for each neighborhood
        disp: torch.Tensor = p_tp1_nb - p_t_nb  # Shape: (N, k, 3)

        # Average displacement within each neighborhood
        disp_mean: torch.Tensor = disp.mean(dim=1)  # Shape: (N, 3)

        # Compute magnitude of average displacement
        mag: torch.Tensor = torch.norm(disp_mean, dim=1)  # Shape: (N,)

        # Accumulate displacement magnitudes over time
        sum_disp += mag

    return sum_disp


def compute_motion_salience_batch(
        all_points_history: np.ndarray,
        neighbor_k: int = 400,
        device: str = 'cuda'
) -> torch.Tensor:
    """
    Wrapper function for neighbor-based motion salience computation with configurable k.

    This is a convenience function that provides a simpler interface to the motion
    salience computation while allowing adjustment of the neighborhood size parameter.
    Larger values of k create smoother, more global motion analysis, while smaller
    values focus on more local motion patterns.

    Args:
        all_points_history (np.ndarray): Point history of shape (T, N, 3)
        neighbor_k (int): Number of neighbors to consider for motion analysis.
                         Default of 400 works well for dense point clouds
        device (str): PyTorch device for computation ('cuda' or 'cpu')

    Returns:
        torch.Tensor: Motion salience scores of shape (N,) indicating relative
                     motion significance for each point
    """
    # Convert to tensor and validate input
    pts: torch.Tensor = torch.as_tensor(all_points_history, dtype=torch.float32, device=device)
    T: int = pts.shape[0]
    N: int = pts.shape[1]

    # Early return for insufficient temporal data
    if T < 2:
        return torch.zeros(N, device=device)

    # Delegate to the main computation function
    return compute_motion_salience_batch_neighborhood(
        all_points_history, device=device, k=neighbor_k
    )