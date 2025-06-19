#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Articulation model fitting demonstration script.

This script demonstrates joint model fitting from the research paper.
It includes various joint models (rigid, prismatic, revolute, GP) and
provides 3D visualization using Polyscope.
"""

import math
import random
import numpy as np
import os
import threading

import polyscope as ps
import polyscope.imgui as psim
import dearpygui.dearpygui as dpg

from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import gpytorch
import torch


###############################################################################
#                          General Helper Functions                           #
#                        (Rotation, Translation, etc.)                        #
###############################################################################


def translate_points(points, displacement, axis):
    """Translate each point along axis direction by displacement amount."""
    return points + displacement * axis


def rotate_points(points, angle, axis, origin):
    """Rotate points around given axis and origin by given angle (radians)."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    pts = points - origin
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    x, y, z = axis
    R = np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c]
    ])
    rotated = pts.dot(R.T)
    return rotated + origin


def rotate_points_xyz(points, angle_x, angle_y, angle_z, center):
    """Rotate sequentially around X, Y, Z axes (angles in radians) with center."""
    pts = points - center
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(angle_x), -math.sin(angle_x)],
        [0, math.sin(angle_x), math.cos(angle_x)]
    ])
    Ry = np.array([
        [math.cos(angle_y), 0, math.sin(angle_y)],
        [0, 1, 0],
        [-math.sin(angle_y), 0, math.cos(angle_y)]
    ])
    Rz = np.array([
        [math.cos(angle_z), -math.sin(angle_z), 0],
        [math.sin(angle_z), math.cos(angle_z), 0],
        [0, 0, 1]
    ])
    R = Rz.dot(Ry).dot(Rx)
    rotated = pts.dot(R.T)
    return rotated + center


def apply_screw_motion(points, angle, axis, origin, pitch):
    """Apply screw motion: rotate around axis by angle while translating along axis by (pitch * angle/(2π))."""
    rotated = rotate_points(points, angle, axis, origin)
    translation = (angle / (2 * math.pi)) * pitch * (axis / (np.linalg.norm(axis) + 1e-12))
    return rotated + translation


def quaternion_to_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix (ensures quaternion normalization)."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx = qx * qx;
    yy = qy * qy;
    zz = qz * qz
    xy = qx * qy;
    xz = qx * qz;
    yz = qy * qz
    wx = qw * qx;
    wy = qw * qy;
    wz = qw * qz
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    return R


def rotation_angle_from_matrix(R):
    """Extract rotation angle from rotation matrix."""
    trace = np.trace(R)
    val = (trace - 1.0) / 2.0
    val = max(-1.0, min(1.0, val))
    return abs(math.acos(val))


###############################################################################
#                              Define Pose Class                              #
###############################################################################


class Pose:
    """Represents a 6DOF pose with position and orientation (quaternion)."""

    def __init__(self, x=0, y=0, z=0, qx=0, qy=0, qz=0, qw=1):
        self.x = x
        self.y = y
        self.z = z
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw


###############################################################################
#                   Define Joint Model Base Class (GenericLinkModel)         #
###############################################################################


def log_outlier_likelihood():
    """Return log likelihood for outlier observations."""
    return -math.log(1e5)


def log_inlier_likelihood(obs_pose, pred_pose, sigma_pos, sigma_ori):
    """Calculate log likelihood for inlier observations."""
    dp = np.array([obs_pose.x - pred_pose.x,
                   obs_pose.y - pred_pose.y,
                   obs_pose.z - pred_pose.z])
    pos_err = np.linalg.norm(dp)
    R1 = quaternion_to_matrix(obs_pose.qx, obs_pose.qy, obs_pose.qz, obs_pose.qw)
    R2 = quaternion_to_matrix(pred_pose.qx, pred_pose.qy, pred_pose.qz, pred_pose.qw)
    dR = R1.T.dot(R2)
    ang_err = rotation_angle_from_matrix(dR)
    sp, so = sigma_pos, sigma_ori
    val = - math.log(2 * math.pi * sp * so) - 0.5 * ((pos_err ** 2) / (sp ** 2) + (ang_err ** 2) / (so ** 2))
    return val


class GenericLinkModel:
    """Base class for all joint models."""

    def __init__(self):
        self._dofs = 0
        self.k = 6  # Number of parameters
        self.name = "generic"
        self.sigma_pos = 0.005
        self.sigma_ori = math.radians(360)
        self.gamma = 0.3  # Outlier ratio
        self.sac_iterations = 50
        self.optimizer_iterations = 10

        # Evaluation metrics
        self.loglikelihood = -1e10
        self.bic = 1e10
        self.avg_pos_err = 999.
        self.avg_ori_err = 999.

        self.state_params = None

    def dofs(self):
        """Return degrees of freedom."""
        return self._dofs

    def sample_size(self):
        """Return minimum sample size needed for model fitting."""
        return 1

    def guess_from_minimal_samples(self, samples):
        """Initialize model parameters from minimal sample set."""
        return False

    def refine_nonlinear(self, data):
        """Refine model parameters using nonlinear optimization."""
        pass

    def forward_kinematics(self, q):
        """Compute pose from joint configuration."""
        return Pose()

    def inverse_kinematics(self, pose):
        """Compute joint configuration from pose."""
        return np.array([])

    def log_inlier_ll_single(self, obs):
        """Calculate log likelihood for single observation."""
        q = self.inverse_kinematics(obs)
        pred = self.forward_kinematics(q)
        return log_inlier_likelihood(obs, pred, self.sigma_pos, self.sigma_ori)

    def compute_log_likelihood(self, data, estimate_gamma=False):
        """Compute total log likelihood of data given model."""
        if len(data) == 0:
            return -1e10
        outl_ll = log_outlier_likelihood()
        n = len(data)
        if estimate_gamma:
            self.gamma = 0.5
            gamma_i = torch.full((n,), 0.5, dtype=torch.float32)
            outl_ll_torch = torch.tensor(outl_ll, dtype=torch.float32)
            changed = True
            it = 0
            while changed and it < 10:
                changed = False
                inl_ll = torch.tensor(
                    [self.log_inlier_ll_single(obs) for obs in data],
                    dtype=torch.float32
                )
                pi = (1 - self.gamma) * torch.exp(inl_ll)
                po = self.gamma * torch.exp(outl_ll_torch)
                new_gamma_i = po / (pi + po + 1e-16)
                if torch.any(torch.abs(new_gamma_i - gamma_i) > 1e-3):
                    changed = True
                gamma_i = new_gamma_i
                new_gamma = gamma_i.mean().item()
                if abs(new_gamma - self.gamma) > 1e-3:
                    changed = True
                self.gamma = new_gamma
                it += 1

        inl_ll = torch.tensor(
            [self.log_inlier_ll_single(obs) for obs in data],
            dtype=torch.float32
        )
        pi = (1 - self.gamma) * torch.exp(inl_ll)
        po = self.gamma * torch.exp(torch.tensor(outl_ll, dtype=torch.float32))
        s = torch.log(pi + po + 1e-16).sum().item()
        return s

    def evaluate_model(self, data):
        """Evaluate model performance on given data."""
        n = len(data)
        if n < 1:
            return
        self.loglikelihood = self.compute_log_likelihood(data, estimate_gamma=False)
        with torch.no_grad():
            poses_obs = torch.tensor([[p.x, p.y, p.z, p.qx, p.qy, p.qz, p.qw] for p in data],
                                     dtype=torch.float32)
            poses_pred = []
            for obs in data:
                q = self.inverse_kinematics(obs)
                pf = self.forward_kinematics(q)
                poses_pred.append([pf.x, pf.y, pf.z, pf.qx, pf.qy, pf.qz, pf.qw])
            poses_pred = torch.tensor(poses_pred, dtype=torch.float32)
            dp = poses_obs[:, :3] - poses_pred[:, :3]
            pos_errs = torch.norm(dp, dim=1)

            def quat_to_mat(q):
                qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                norm = torch.sqrt(qx ** 2 + qy ** 2 + qz ** 2 + qw ** 2 + 1e-8)
                qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
                xx, yy, zz = qx * qx, qy * qy, qz * qz
                xy, xz, yz = qx * qy, qx * qz, qy * qz
                wx, wy, wz = qw * qx, qw * qy, qw * qz
                mats = torch.stack([
                    torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=1),
                    torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=1),
                    torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=1)
                ], dim=1)
                return mats

            R1 = quat_to_mat(poses_obs[:, 3:])
            R2 = quat_to_mat(poses_pred[:, 3:])
            dR = torch.bmm(R1.transpose(1, 2), R2)
            trace = dR[:, 0, 0] + dR[:, 1, 1] + dR[:, 2, 2]
            val = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
            ori_errs = torch.acos(val).abs()
            self.avg_pos_err = pos_errs.mean().item()
            self.avg_ori_err = ori_errs.mean().item()
        self.loglikelihood -= n * self.dofs() * math.log(n)
        self.bic = -2 * self.loglikelihood + self.k * math.log(n)

    def sample_consensus_fit(self, data):
        """Fit model using sample consensus (RANSAC-like) approach."""
        best_ll = -1e10
        best_state = None
        for _ in range(self.sac_iterations):
            N = self.sample_size()
            if len(data) < N:
                continue
            idxs = random.sample(range(len(data)), N)
            subset = [data[i] for i in idxs]
            if not self.guess_from_minimal_samples(subset):
                continue
            ll = self.compute_log_likelihood(data, estimate_gamma=True)
            if ll > best_ll:
                best_ll = ll
                best_state = self.save_state()
        if best_state is None:
            return False
        self.load_state(best_state)
        self.refine_nonlinear(data)
        return True

    def save_state(self):
        """Save current model state."""
        return {
            'sigma_pos': self.sigma_pos,
            'sigma_ori': self.sigma_ori,
            'gamma': self.gamma,
            'state_params': self.state_params
        }

    def load_state(self, st):
        """Load model state from saved dictionary."""
        self.sigma_pos = st['sigma_pos']
        self.sigma_ori = st['sigma_ori']
        self.gamma = st['gamma']
        self.state_params = st['state_params']


###############################################################################
#              Specific Models: Rigid, Prismatic, Revolute, GP               #
###############################################################################


class RigidModel(GenericLinkModel):
    """Model for rigid body motion (0 DOF)."""

    def __init__(self):
        super().__init__()
        self.name = "rigid"
        self._dofs = 0
        self.k = 6
        self.offset = Pose()

    def sample_size(self):
        return 1

    def guess_from_minimal_samples(self, samples):
        p = samples[0]
        self.offset = p
        self.state_params = (p.x, p.y, p.z, p.qx, p.qy, p.qz, p.qw)
        return True

    def forward_kinematics(self, q):
        return self.offset

    def inverse_kinematics(self, pose):
        return np.array([])

    def refine_nonlinear(self, data):
        init = self.save_state()

        def objective(x):
            px, py, pz, qx, qy, qz, qw = x
            n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if n < 1e-12:
                return 1e10
            qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
            self.offset = Pose(px, py, pz, qx, qy, qz, qw)
            ll = self.compute_log_likelihood(data, estimate_gamma=True)
            return -ll

        x0 = self.state_params
        res = minimize(objective, x0, method='BFGS', options={'maxiter': self.optimizer_iterations})
        x = res.x
        px, py, pz, qx, qy, qz, qw = x
        n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if n < 1e-12:
            qx, qy, qz, qw = 0, 0, 0, 1
        else:
            qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
        self.offset = Pose(px, py, pz, qx, qy, qz, qw)
        self.state_params = (px, py, pz, qx, qy, qz, qw)


class PrismaticModel(GenericLinkModel):
    """Model for prismatic (translational) joint (1 DOF)."""

    def __init__(self):
        super().__init__()
        self.name = "prismatic"
        self._dofs = 1
        self.k = 8
        self.rigid_position = np.array([0., 0., 0.])
        self.rigid_orientation_q = np.array([0., 0., 0., 1.])
        self.prismatic_dir = np.array([1., 0., 0.])
        self.state_params = None

    def sample_size(self):
        return 2

    def guess_from_minimal_samples(self, samples):
        if len(samples) < 2:
            return False
        p1, p2 = samples
        pos1 = np.array([p1.x, p1.y, p1.z])
        pos2 = np.array([p2.x, p2.y, p2.z])
        diff = pos2 - pos1
        norm_ = np.linalg.norm(diff)
        if norm_ < 1e-12:
            return False
        self.rigid_position = pos1
        self.rigid_orientation_q = np.array([p1.qx, p1.qy, p1.qz, p1.qw])
        self.prismatic_dir = diff / norm_
        self.state_params = (
            *self.rigid_position,
            *self.rigid_orientation_q,
            *self.prismatic_dir
        )
        return True

    def forward_kinematics(self, q):
        dist = q[0]
        px = self.rigid_position[0] + dist * self.prismatic_dir[0]
        py = self.rigid_position[1] + dist * self.prismatic_dir[1]
        pz = self.rigid_position[2] + dist * self.prismatic_dir[2]
        ox, oy, oz, ow = self.rigid_orientation_q
        return Pose(px, py, pz, ox, oy, oz, ow)

    def inverse_kinematics(self, pose):
        diff = np.array([pose.x, pose.y, pose.z]) - self.rigid_position
        val = np.dot(diff, self.prismatic_dir)
        return np.array([val])

    def batch_forward_kinematics(self, q_tensor):
        """Batch forward kinematics for tensor operations."""
        dist = q_tensor.squeeze(-1)
        base = torch.tensor(self.rigid_position, dtype=torch.float32)
        dir_ = torch.tensor(self.prismatic_dir, dtype=torch.float32)
        pos = base + dist.unsqueeze(1) * dir_.unsqueeze(0)
        orient = torch.tensor(self.rigid_orientation_q, dtype=torch.float32).unsqueeze(0).repeat(len(q_tensor), 1)
        return torch.cat([pos, orient], dim=1)

    def batch_inverse_kinematics(self, pose_tensor):
        """Batch inverse kinematics for tensor operations."""
        pos = pose_tensor[:, :3]
        base = torch.tensor(self.rigid_position, dtype=torch.float32).unsqueeze(0)
        dir_ = torch.tensor(self.prismatic_dir, dtype=torch.float32).unsqueeze(0)
        diff = pos - base
        q_val = (diff * dir_).sum(dim=1, keepdim=True)
        return q_val


class RevoluteModel(GenericLinkModel):
    """Model for revolute (rotational) joint (1 DOF)."""

    def __init__(self):
        super().__init__()
        self.name = "revolute"
        self._dofs = 1
        self.k = 12
        self.rot_mode = 0  # 0: position-based, 1: orientation-based
        self.rot_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self.rot_axis_q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self.rot_radius = 1.0
        self.rot_orientation_q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self.center = Pose()
        self.radius = 1.0
        self.offset = Pose()
        self.state_params = (self.rot_center[0].item(), self.rot_center[1].item(), self.rot_center[2].item(),
                             self.rot_axis_q[0].item(), self.rot_axis_q[1].item(), self.rot_axis_q[2].item(),
                             self.rot_axis_q[3].item(),
                             self.rot_radius,
                             self.rot_orientation_q[0].item(), self.rot_orientation_q[1].item(),
                             self.rot_orientation_q[2].item(), self.rot_orientation_q[3].item(),
                             float(self.rot_mode))
        self.optimizer_iterations = 100

    def sample_size(self):
        return 3

    def pose_to_matrix(self, pose: Pose) -> torch.Tensor:
        """Convert pose to 4x4 transformation matrix."""
        R = quaternion_to_matrix(pose.qx, pose.qy, pose.qz, pose.qw)
        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = torch.tensor(R, dtype=torch.float32)
        T[0, 3] = pose.x
        T[1, 3] = pose.y
        T[2, 3] = pose.z
        return T

    def matrix_to_pose(self, M: torch.Tensor) -> Pose:
        """Convert 4x4 transformation matrix to pose."""
        R = M[:3, :3].numpy()
        trace_val = R[0, 0] + R[1, 1] + R[2, 2]
        val = 1 + trace_val
        qw = math.sqrt(val) / 2 if val > 0 else 0.0
        if abs(qw) < 1e-12:
            return Pose(M[0, 3].item(), M[1, 3].item(), M[2, 3].item(), 0, 0, 0, 1)
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
        return Pose(M[0, 3].item(), M[1, 3].item(), M[2, 3].item(), qx, qy, qz, qw)

    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return torch.tensor([x, y, z, w], dtype=torch.float32)

    def inverse_pose(self, M: torch.Tensor) -> torch.Tensor:
        """Compute inverse of transformation matrix."""
        R = M[:3, :3]
        t = M[:3, 3]
        R_inv = R.t()
        t_inv = -torch.matmul(R_inv, t)
        Minv = torch.eye(4, dtype=torch.float32)
        Minv[:3, :3] = R_inv
        Minv[:3, 3] = t_inv
        return Minv

    def transform_multiply(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Multiply two transformation matrices."""
        return torch.matmul(A, B)

    def guess_from_minimal_samples(self, samples):
        if len(samples) < 3:
            return False
        self.rot_mode = 1 if random.random() < 0.5 else 0
        p1, p2, p3 = samples[0], samples[1], samples[2]
        M1 = self.pose_to_matrix(p1)
        M2 = self.pose_to_matrix(p2)
        M3 = self.pose_to_matrix(p3)
        if self.rot_mode == 1:
            # Orientation-based mode
            M1_inv = self.inverse_pose(M1)
            M12 = self.transform_multiply(M1_inv, M2)
            pose12 = self.matrix_to_pose(M12)
            angle_12 = 2 * math.acos(max(-1.0, min(1.0, pose12.qw)))
            dist_12 = torch.norm(torch.tensor([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z], dtype=torch.float32))
            axis_local = torch.tensor([pose12.qx, pose12.qy, pose12.qz], dtype=torch.float32)
            n_ = torch.norm(axis_local)
            if n_ < 1e-12:
                axis_local = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            else:
                axis_local = axis_local / n_
            if abs(math.sin(angle_12 * 0.5)) < 1e-12:
                return False
            self.rot_radius = (dist_12.item() * 0.5) / math.sin(angle_12 * 0.5)
            c1 = torch.tensor([p1.x, p1.y, p1.z], dtype=torch.float32)
            c2 = torch.tensor([p2.x, p2.y, p2.z], dtype=torch.float32)
            mid = 0.5 * (c1 + c2)
            v1_ = c2 - c1
            v1_norm = torch.norm(v1_)
            if v1_norm < 1e-12:
                return False
            v1_ = v1_ / v1_norm
            dot_ = torch.dot(axis_local, v1_)
            v1_perp = v1_ - dot_ * axis_local
            vnorm = torch.norm(v1_perp)
            if vnorm < 1e-12:
                v1_perp = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            else:
                v1_perp = v1_perp / vnorm
            # Specify dim=0 for cross product
            cross_ = torch.cross(v1_perp, axis_local, dim=0)
            offset_len = self.rot_radius * math.cos(angle_12 * 0.5)
            center_approx = mid + cross_ * offset_len
            self.rot_center = center_approx

            def axis_to_quaternion(axis_src, axis_dst):
                a = axis_src / (torch.norm(axis_src) + 1e-12)
                b = axis_dst / (torch.norm(axis_dst) + 1e-12)
                crossv = torch.cross(a, b, dim=0)
                dotv = torch.dot(a, b)
                w_ = math.sqrt((1 + dotv.item()) * 2) * 0.5
                if abs(w_) < 1e-12:
                    crossv = torch.cross(a, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32), dim=0)
                    if torch.norm(crossv) < 1e-12:
                        crossv = torch.cross(a, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32), dim=0)
                    crossv = crossv / (torch.norm(crossv) + 1e-12)
                else:
                    crossv = crossv / (2 * w_)
                return torch.tensor([crossv[0].item(), crossv[1].item(), crossv[2].item(), w_], dtype=torch.float32)

            self.rot_axis_q = axis_to_quaternion(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32), axis_local)
            q1 = torch.tensor([p1.qx, p1.qy, p1.qz, p1.qw], dtype=torch.float32)
            x_, y_, z_, w_ = self.rot_axis_q
            normq = math.sqrt(x_ * x_ + y_ * y_ + z_ * z_ + w_ * w_)
            if normq < 1e-12:
                inv_axis_q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
            else:
                inv_axis_q = torch.tensor([-x_ / normq, -y_ / normq, -z_ / normq, w_ / normq], dtype=torch.float32)
            self.rot_orientation_q = self.quaternion_multiply(inv_axis_q, q1)
        else:
            # Position-based mode
            c1 = torch.tensor([p1.x, p1.y, p1.z], dtype=torch.float32)
            c2 = torch.tensor([p2.x, p2.y, p2.z], dtype=torch.float32)
            c3 = torch.tensor([p3.x, p3.y, p3.z], dtype=torch.float32)
            v_12 = c2 - c1
            v_13 = c3 - c1
            n_ = torch.cross(v_12, v_13, dim=0)
            nn = torch.norm(n_)
            if nn < 1e-12:
                return False
            n_ = n_ / nn
            plane_x = v_12 / (torch.norm(v_12) + 1e-12)
            plane_y = v_13 - torch.dot(v_13, plane_x) * plane_x
            ny = torch.norm(plane_y)
            if ny < 1e-12:
                return False
            plane_y = plane_y / ny
            plane_z = torch.cross(plane_x, plane_y, dim=0)
            plane_rot = torch.eye(4, dtype=torch.float32)
            plane_rot[:3, 0] = plane_x
            plane_rot[:3, 1] = plane_y
            plane_rot[:3, 2] = plane_z
            plane_rot[:3, 3] = c1
            plane_inv = self.inverse_pose(plane_rot)
            p1_ = self.transform_multiply(plane_inv, M1)
            p2_ = self.transform_multiply(plane_inv, M2)
            p3_ = self.transform_multiply(plane_inv, M3)
            c1_ = 0.5 * (p1_[:3, 3] + p2_[:3, 3])
            c2_ = 0.5 * (p1_[:3, 3] + p3_[:3, 3])

            def rotateZ(vec, rad):
                return torch.tensor([
                    vec[0] * math.cos(rad) - vec[1] * math.sin(rad),
                    vec[0] * math.sin(rad) + vec[1] * math.cos(rad),
                    vec[2]
                ], dtype=torch.float32)

            p21 = rotateZ(p2_[:3, 3] - p1_[:3, 3], math.pi / 2.0)
            p43 = rotateZ(p3_[:3, 3] - p1_[:3, 3], math.pi / 2.0)
            A = torch.stack([p21[:2], -p43[:2]], dim=1)
            b = (c2_[:2] - c1_[:2])
            if torch.abs(torch.det(A)) < 1e-12:
                return False
            lam = torch.linalg.inv(A) @ b
            onplane_center = c1_[:2] + lam[0] * p21[:2]
            plane_center_4 = torch.tensor([onplane_center[0].item(), onplane_center[1].item(), c1_[2].item(), 1.0],
                                          dtype=torch.float32)
            center_world_4 = self.transform_multiply(plane_rot, plane_center_4.view(4, 1))
            center_world_4 = center_world_4.view(-1)
            self.rot_center = center_world_4[:3]
            plane_rot3 = plane_rot[:3, :3]
            qw_ = math.sqrt(max(0, 1 + plane_rot3[0, 0].item() + plane_rot3[1, 1].item() + plane_rot3[2, 2].item())) / 2
            if abs(qw_) < 1e-12:
                self.rot_axis_q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
            else:
                qx_ = (plane_rot3[2, 1] - plane_rot3[1, 2]).item() / (4 * qw_)
                qy_ = (plane_rot3[0, 2] - plane_rot3[2, 0]).item() / (4 * qw_)
                qz_ = (plane_rot3[1, 0] - plane_rot3[0, 1]).item() / (4 * qw_)
                self.rot_axis_q = torch.tensor([qx_, qy_, qz_, qw_], dtype=torch.float32)
            r_ = torch.norm(p1_[:3, 3] - plane_center_4[:3])
            self.rot_radius = r_.item()
            self.rot_orientation_q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
            q_of_p1 = self.inverse_kinematics(p1)
            p_pred_1 = self.forward_kinematics(q_of_p1)
            M_pred_1 = self.pose_to_matrix(p_pred_1)
            Minv_pred_1 = self.inverse_pose(M_pred_1)
            diff_ = self.transform_multiply(Minv_pred_1, M1)
            diff_pose = self.matrix_to_pose(diff_)
            diff_q = torch.tensor([diff_pose.qx, diff_pose.qy, diff_pose.qz, diff_pose.qw], dtype=torch.float32)
            self.rot_orientation_q = self.quaternion_multiply(self.rot_orientation_q, diff_q)
        self.center.x, self.center.y, self.center.z = self.rot_center[0].item(), self.rot_center[1].item(), \
        self.rot_center[2].item()
        self.radius = self.rot_radius
        xq, yq, zq, wq = self.rot_orientation_q
        self.offset = Pose(0, 0, 0, xq.item(), yq.item(), zq.item(), wq.item())
        self.state_params = (self.rot_center[0].item(), self.rot_center[1].item(), self.rot_center[2].item(),
                             self.rot_axis_q[0].item(), self.rot_axis_q[1].item(), self.rot_axis_q[2].item(),
                             self.rot_axis_q[3].item(),
                             self.rot_radius,
                             xq.item(), yq.item(), zq.item(), wq.item(),
                             float(self.rot_mode))
        return True

    def forward_kinematics(self, q):
        if isinstance(q, (list, tuple)):
            angle = -q[0]
        elif isinstance(q, np.ndarray):
            angle = -q[0]
        elif isinstance(q, torch.Tensor) and q.dim() == 1:
            angle = -q[0].item()
        else:
            angle = -q[:, 0]
        C = torch.eye(4, dtype=torch.float32)
        Rax = quaternion_to_matrix(self.rot_axis_q[0].item(), self.rot_axis_q[1].item(),
                                   self.rot_axis_q[2].item(), self.rot_axis_q[3].item())
        C[:3, :3] = torch.tensor(Rax, dtype=torch.float32)
        C[:3, 3] = self.rot_center
        Rz = torch.eye(4, dtype=torch.float32)
        if isinstance(angle, float):
            ca = math.cos(angle)
            sa = math.sin(angle)
        else:
            ca = torch.cos(angle)
            sa = torch.sin(angle)
        Rz[0, 0] = ca
        Rz[0, 1] = -sa
        Rz[1, 0] = sa
        Rz[1, 1] = ca
        T_r = torch.eye(4, dtype=torch.float32)
        T_r[0, 3] = self.rot_radius
        O_ = self.pose_to_matrix(self.offset)
        M = self.transform_multiply(C, Rz)
        M = self.transform_multiply(M, T_r)
        M = self.transform_multiply(M, O_)
        return self.matrix_to_pose(M)

    def inverse_kinematics(self, pose: Pose):
        Mobs = self.pose_to_matrix(pose)
        C = torch.eye(4, dtype=torch.float32)
        Rax = quaternion_to_matrix(self.rot_axis_q[0].item(), self.rot_axis_q[1].item(),
                                   self.rot_axis_q[2].item(), self.rot_axis_q[3].item())
        C[:3, :3] = torch.tensor(Rax, dtype=torch.float32)
        C[:3, 3] = self.rot_center
        C_inv = self.inverse_pose(C)
        Mrel = self.transform_multiply(C_inv, Mobs)
        if self.rot_mode == 1:
            T_r_inv = torch.eye(4, dtype=torch.float32)
            T_r_inv[0, 3] = -self.rot_radius
            offset_inv = self.inverse_pose(self.pose_to_matrix(self.offset))
            M_no = self.transform_multiply(Mrel, offset_inv)
            M_no = self.transform_multiply(M_no, T_r_inv)
            x_ = M_no[0, 0]
            y_ = M_no[1, 0]
            angle = math.atan2(y_.item(), x_.item())
            return torch.tensor([-angle], dtype=torch.float32)
        else:
            tx = Mrel[0, 3]
            ty = Mrel[1, 3]
            angle = -math.atan2(ty.item(), tx.item())
            return torch.tensor([angle], dtype=torch.float32)

    def compute_log_likelihood(self, data, estimate_gamma=True):
        return torch.tensor(0.0, dtype=torch.float32)

    def refine_nonlinear(self, data):
        init_state = self.state_params
        x0 = torch.tensor(self.state_params[:12], dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS([x0], max_iter=self.optimizer_iterations)

        def closure():
            optimizer.zero_grad()
            cx, cy, cz = x0[0], x0[1], x0[2]
            ax, ay, az, aw = x0[3], x0[4], x0[5], x0[6]
            radius_ = x0[7]
            ox, oy, oz, ow = x0[8], x0[9], x0[10], x0[11]
            nq1 = torch.sqrt(ax * ax + ay * ay + az * az + aw * aw)
            if nq1 < 1e-12:
                nq1 = torch.tensor(1.0, requires_grad=True)
            ax, ay, az, aw = ax / nq1, ay / nq1, az / nq1, aw / nq1
            nq2 = torch.sqrt(ox * ox + oy * oy + oz * oz + ow * ow)
            if nq2 < 1e-12:
                nq2 = torch.tensor(1.0, requires_grad=True)
            ox, oy, oz, ow = ox / nq2, oy / nq2, oz / nq2, ow / nq2
            self.rot_center = torch.tensor([cx, cy, cz], dtype=torch.float32)
            self.rot_axis_q = torch.tensor([ax, ay, az, aw], dtype=torch.float32)
            self.rot_radius = max(0, radius_.item())
            self.rot_orientation_q = torch.tensor([ox, oy, oz, ow], dtype=torch.float32)
            self.center.x, self.center.y, self.center.z = self.rot_center[0].item(), self.rot_center[1].item(), \
            self.rot_center[2].item()
            self.radius = self.rot_radius
            self.offset = Pose(0, 0, 0, ox.item(), oy.item(), oz.item(), ow.item())
            ll = self.compute_log_likelihood(data, estimate_gamma=True)
            loss = -ll + 0 * torch.sum(x0)  # Force dependency to give loss gradients
            loss.backward()
            return loss

        optimizer.step(closure)
        self.state_params = (self.rot_center[0].item(), self.rot_center[1].item(), self.rot_center[2].item(),
                             self.rot_axis_q[0].item(), self.rot_axis_q[1].item(), self.rot_axis_q[2].item(),
                             self.rot_axis_q[3].item(),
                             self.rot_radius,
                             self.rot_orientation_q[0].item(), self.rot_orientation_q[1].item(),
                             self.rot_orientation_q[2].item(), self.rot_orientation_q[3].item(),
                             self.state_params[-1])
        return


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact Gaussian Process model for GPyTorch."""

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel(GenericLinkModel):
    """Gaussian Process model for complex joint motions."""

    def __init__(self):
        super().__init__()
        self.name = "gp"
        self._dofs = 1
        self.k = 5
        self.initialized = False
        self.n_down = 20  # Downsampling factor
        self.anchor = Pose()
        self.axis = torch.tensor([1., 0., 0.], dtype=torch.float32)
        self.X = None
        self.Y = None
        self.gps = []
        self.likelihoods = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit_model(self, data):
        """Fit GP models to the data."""
        n = len(data)
        if n < 2:
            return False
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        while j == i:
            j = random.randint(0, n - 1)
        p1 = data[i]
        p2 = data[j]
        diff = torch.tensor([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z], dtype=torch.float32)
        nm = torch.norm(diff)
        if nm < 1e-12:
            return False
        self.anchor = p1
        self.axis = diff / nm
        idxs = np.round(np.linspace(0, n - 1, min(n, self.n_down))).astype(int)
        idxs = list(set(idxs))
        idxs.sort()
        self.X = []
        self.Y = []
        for idx in idxs:
            obs = data[idx]
            q = self._inv(obs)
            self.X.append([q])
            self.Y.append([obs.x, obs.y, obs.z])
        self.X = torch.tensor(self.X, dtype=torch.float32).to(self.device)
        self.Y = torch.tensor(self.Y, dtype=torch.float32).to(self.device)
        self.gps = []
        self.likelihoods = []
        for c in range(3):
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            model = ExactGPModel(self.X, self.Y[:, c], likelihood).to(self.device)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for _ in range(50):
                optimizer.zero_grad()
                output = model(self.X)
                loss = -mll(output, self.Y[:, c])
                loss.backward()
                optimizer.step()
            model.eval()
            likelihood.eval()
            self.gps.append(model)
            self.likelihoods.append(likelihood)
        self.initialized = True
        return True

    def sample_size(self):
        return 2

    def guess_from_minimal_samples(self, samples):
        return True

    def sample_consensus_fit(self, data):
        return self.fit_model(data)

    def refine_nonlinear(self, data):
        pass

    def _inv(self, pose):
        """Project pose onto the axis to get scalar parameter."""
        dx = torch.tensor([pose.x - self.anchor.x, pose.y - self.anchor.y, pose.z - self.anchor.z],
                          dtype=torch.float32)
        return torch.dot(dx, self.axis).item()

    def inverse_kinematics(self, pose):
        val = self._inv(pose)
        return np.array([val])

    def forward_kinematics(self, q):
        if not self.initialized or len(self.gps) == 0:
            px = self.anchor.x + q[0] * self.axis[0].item()
            py = self.anchor.y + q[0] * self.axis[1].item()
            pz = self.anchor.z + q[0] * self.axis[2].item()
            return Pose(px, py, pz, 0, 0, 0, 1)
        xTest = torch.tensor([[q[0]]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            px = self.gps[0](xTest).mean.item()
            py = self.gps[1](xTest).mean.item()
            pz = self.gps[2](xTest).mean.item()
        return Pose(px, py, pz, 0, 0, 0, 1)

    def evaluate_model(self, data):
        if not self.initialized:
            super().evaluate_model(data)
            return
        self.loglikelihood = self.compute_log_likelihood(data, estimate_gamma=False)
        n = len(data)
        sum_p = 0
        sum_o = 0
        for obs in data:
            q = self.inverse_kinematics(obs)
            pf = self.forward_kinematics(q)
            dp = np.array([obs.x - pf.x, obs.y - pf.y, obs.z - pf.z])
            sum_p += np.linalg.norm(dp)
            R1 = quaternion_to_matrix(obs.qx, obs.qy, obs.qz, obs.qw)
            R2 = quaternion_to_matrix(pf.qx, pf.qy, pf.qz, pf.qw)
            sum_o += rotation_angle_from_matrix(np.dot(R1.T, R2))
        self.avg_pos_err = sum_p / n
        self.avg_ori_err = sum_o / n
        self.k = 6 * len(self.X) + 2
        self.loglikelihood -= n * self.dofs() * math.log(n)
        self.bic = -2 * self.loglikelihood + self.k * math.log(n)

    def compute_log_likelihood(self, data, estimate_gamma=False):
        return 0.0


###############################################################################
#     Package Four Models and Evaluate => Select Optimal BIC Model           #
###############################################################################


def fit_and_select_best_model(data):
    """Fit all candidate models and select the best one based on BIC."""
    candidates = [RigidModel(), PrismaticModel(), RevoluteModel(), GPModel()]
    best = None
    best_bic = 1e20
    for c in candidates:
        ok = c.sample_consensus_fit(data)
        if ok:
            c.evaluate_model(data)
            if c.bic < best_bic:
                best_bic = c.bic
                best = c
    return best


###############################################################################
#         Generate Synthetic Data (Different Modes) and Record to             #
#                           Historical Data                                   #
###############################################################################


# Define all motion modes
modes = [
    "Prismatic Door", "Revolute Door", "Planar Mouse",
    "Ball Joint", "Screw Joint",
    "Prismatic Door 2", "Revolute Door 2", "Planar Mouse 2",
    "Ball Joint 2", "Screw Joint 2",
    "Prismatic Door 3", "Revolute Door 3", "Planar Mouse 3",
    "Ball Joint 3", "Screw Joint 3",
]

# Initialize data structures
point_cloud_history = {m: [] for m in modes}
frame_count = {m: 0 for m in modes}
prev_positions = {m: None for m in modes}

# Template parameters
num_points = 500
door_width, door_height, door_thickness = 2.0, 3.0, 0.2

# Generate point cloud templates
prismatic_template = np.random.rand(num_points, 3)
prismatic_template[:, 0] = prismatic_template[:, 0] * door_width - 0.5 * door_width
prismatic_template[:, 1] = prismatic_template[:, 1] * door_height
prismatic_template[:, 2] = prismatic_template[:, 2] * door_thickness - 0.5 * door_thickness

revolute_template = prismatic_template.copy()

mouse_length, mouse_width, mouse_height = 1.0, 0.6, 0.3
mouse_template = np.zeros((num_points, 3))
mouse_template[:, 0] = np.random.rand(num_points) * mouse_length - 0.5 * mouse_length
mouse_template[:, 2] = np.random.rand(num_points) * mouse_width - 0.5 * mouse_width
mouse_template[:, 1] = np.random.rand(num_points) * mouse_height


def generate_ball_sphere_rod(center, sphere_radius, rod_length, rod_radius, n_sphere=250, n_rod=250):
    """Generate a compound shape with sphere and cylindrical rod."""

    def random_sphere(c, r, n):
        pts = []
        for _ in range(n):
            phi = 2 * math.pi * random.random()
            costh = 2 * random.random() - 1
            sinth = math.sqrt(1 - costh * costh)
            rr = r * (random.random() ** (1 / 3))
            x = rr * sinth * math.cos(phi)
            y = rr * sinth * math.sin(phi)
            z = rr * costh
            pts.append([c[0] + x, c[1] + y, c[2] + z])
        return np.array(pts)

    def random_cylinder(r, h, n):
        pts = []
        for _ in range(n):
            z_ = (random.random() - 0.5) * h
            phi = 2 * math.pi * random.random()
            rr = r * math.sqrt(random.random())
            x = rr * math.cos(phi)
            y = rr * math.sin(phi)
            pts.append([x, z_, y])
        return np.array(pts)

    spts = random_sphere(center, sphere_radius, n_sphere)
    rpts = random_cylinder(rod_radius, rod_length, n_rod)
    rpts[:, 1] += center[1]
    rpts[:, 0] += center[0]
    rpts[:, 2] += center[2]
    return np.concatenate([spts, rpts], axis=0)


ball_template = generate_ball_sphere_rod(np.array([0, 0, 0]), 0.3, 3.0, 0.05, 250, 250)


def generate_hollow_cylinder(radius, height, thickness, n=500):
    """Generate hollow cylinder point cloud."""
    data = []
    for _ in range(n):
        if random.random() < 0.2:  # Top/bottom surface
            r_ = random.random() * radius
            phi = 2 * math.pi * random.random()
            x = r_ * math.cos(phi)
            y = height
            z = r_ * math.sin(phi)
            data.append([x, y, z])
        else:  # Cylindrical wall
            ph = 2 * math.pi * random.random()
            rr = radius - thickness * random.random()
            y_ = random.random() * height
            x_ = rr * math.cos(ph)
            z_ = rr * math.sin(ph)
            data.append([x_, y_, z_])
    return np.array(data)


screw_template = generate_hollow_cylinder(0.4, 0.2, 0.05, 500)

# Generate additional templates with offsets
prismatic_template_2 = prismatic_template + np.array([1., 0., 0.])
revolute_template_2 = revolute_template + np.array([0., 0., -1.])
mouse_template_2 = mouse_template + np.array([1., 0., 1.])
ball_template_2 = ball_template + np.array([1., 0., 0.])
screw_template_2 = screw_template + np.array([1., 0., -0.5])

prismatic_template_3 = prismatic_template + np.array([-1., 1., 0.])
revolute_template_3 = revolute_template + np.array([0., -0.5, 1.])
mouse_template_3 = mouse_template + np.array([-1., 0., 1.])
ball_template_3 = ball_template + np.array([1., 1., 0.])
screw_template_3 = screw_template + np.array([-1., 0., 1.])

# Initialize Polyscope
ps.init()

# Register point clouds in Polyscope
ps_prismatic_door = ps.register_point_cloud("Prismatic Door", prismatic_template.copy())
ps_revolute_door = ps.register_point_cloud("Revolute Door", revolute_template.copy(), enabled=False)
ps_mouse = ps.register_point_cloud("Planar Mouse", mouse_template.copy(), enabled=False)
ps_ball = ps.register_point_cloud("Ball Joint", ball_template.copy(), enabled=False)
ps_screw = ps.register_point_cloud("Screw Joint", screw_template.copy(), enabled=False)

ps_prismatic_door_2 = ps.register_point_cloud("Prismatic Door 2", prismatic_template_2, enabled=False)
ps_revolute_door_2 = ps.register_point_cloud("Revolute Door 2", revolute_template_2, enabled=False)
ps_mouse_2 = ps.register_point_cloud("Planar Mouse 2", mouse_template_2, enabled=False)
ps_ball_2 = ps.register_point_cloud("Ball Joint 2", ball_template_2, enabled=False)
ps_screw_2 = ps.register_point_cloud("Screw Joint 2", screw_template_2, enabled=False)

ps_prismatic_door_3 = ps.register_point_cloud("Prismatic Door 3", prismatic_template_3, enabled=False)
ps_revolute_door_3 = ps.register_point_cloud("Revolute Door 3", revolute_template_3, enabled=False)
ps_mouse_3 = ps.register_point_cloud("Planar Mouse 3", mouse_template_3, enabled=False)
ps_ball_3 = ps.register_point_cloud("Ball Joint 3", ball_template_3, enabled=False)
ps_screw_3 = ps.register_point_cloud("Screw Joint 3", screw_template_3, enabled=False)

# Noise parameters
noise_sigma = 0.006


def restore_shape(mode):
    """Reset shapes to initial state and enable only the specified mode."""
    # Enable/disable appropriate point clouds
    ps_prismatic_door.set_enabled(mode == "Prismatic Door")
    ps_revolute_door.set_enabled(mode == "Revolute Door")
    ps_mouse.set_enabled(mode == "Planar Mouse")
    ps_ball.set_enabled(mode == "Ball Joint")
    ps_screw.set_enabled(mode == "Screw Joint")

    ps_prismatic_door_2.set_enabled(mode == "Prismatic Door 2")
    ps_revolute_door_2.set_enabled(mode == "Revolute Door 2")
    ps_mouse_2.set_enabled(mode == "Planar Mouse 2")
    ps_ball_2.set_enabled(mode == "Ball Joint 2")
    ps_screw_2.set_enabled(mode == "Screw Joint 2")

    ps_prismatic_door_3.set_enabled(mode == "Prismatic Door 3")
    ps_revolute_door_3.set_enabled(mode == "Revolute Door 3")
    ps_mouse_3.set_enabled(mode == "Planar Mouse 3")
    ps_ball_3.set_enabled(mode == "Ball Joint 3")
    ps_screw_3.set_enabled(mode == "Screw Joint 3")

    # Reset positions to initial templates
    ps_prismatic_door.update_point_positions(prismatic_template.copy())
    ps_revolute_door.update_point_positions(revolute_template.copy())
    ps_mouse.update_point_positions(mouse_template.copy())
    ps_ball.update_point_positions(ball_template.copy())
    ps_screw.update_point_positions(screw_template.copy())

    ps_prismatic_door_2.update_point_positions(prismatic_template_2.copy())
    ps_revolute_door_2.update_point_positions(revolute_template_2.copy())
    ps_mouse_2.update_point_positions(mouse_template_2.copy())
    ps_ball_2.update_point_positions(ball_template_2.copy())
    ps_screw_2.update_point_positions(screw_template_2.copy())

    ps_prismatic_door_3.update_point_positions(prismatic_template_3.copy())
    ps_revolute_door_3.update_point_positions(revolute_template_3.copy())
    ps_mouse_3.update_point_positions(mouse_template_3.copy())
    ps_ball_3.update_point_positions(ball_template_3.copy())
    ps_screw_3.update_point_positions(screw_template_3.copy())

    # Clear history
    point_cloud_history[mode] = []
    frame_count[mode] = 0
    prev_positions[mode] = None


# Error tracking for axis estimation
best_axis_pos_err = 999.  # Position error for axis
best_axis_ang_err = 999.  # Angular error for axis


def compute_axis_errors(model, mode):
    """Calculate position and angular errors between estimated axis and ground truth.

    Returns:
        tuple: (position_error, angular_error)
    """
    position_error = 999.
    angular_error = 999.

    if model is None:
        return position_error, angular_error

    # Prismatic Door
    if mode == "Prismatic Door":
        gt_axis = np.array([1., 0., 0.])
        gt_origin = np.array([0., 0., 0.])
        if model.name == "prismatic":
            est_axis = model.prismatic_dir
            est_origin = model.rigid_position

            # Angular error - angle between axes
            dot_val = np.dot(est_axis, gt_axis) / (np.linalg.norm(est_axis) * np.linalg.norm(gt_axis))
            angular_error = np.arccos(np.clip(abs(dot_val), 0, 1))

            # Position error - closest distance between lines
            v = est_origin - gt_origin
            cross = np.cross(est_axis, gt_axis)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:  # Parallel lines
                position_error = np.linalg.norm(np.cross(v, gt_axis)) / np.linalg.norm(gt_axis)
            else:
                n = cross / cross_norm
                position_error = abs(np.dot(v, n))

    # Prismatic Door 2
    elif mode == "Prismatic Door 2":
        gt_axis = np.array([0., 1., 0.])
        gt_origin = np.array([1., 0., 0.])
        if model.name == "prismatic":
            est_axis = model.prismatic_dir
            est_origin = model.rigid_position

            # Angular error
            dot_val = np.dot(est_axis, gt_axis) / (np.linalg.norm(est_axis) * np.linalg.norm(gt_axis))
            angular_error = np.arccos(np.clip(abs(dot_val), 0, 1))

            # Position error
            v = est_origin - gt_origin
            cross = np.cross(est_axis, gt_axis)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:
                position_error = np.linalg.norm(np.cross(v, gt_axis)) / np.linalg.norm(gt_axis)
            else:
                n = cross / cross_norm
                position_error = abs(np.dot(v, n))

    # Prismatic Door 3
    elif mode == "Prismatic Door 3":
        gt_axis = np.array([0., 0., 1.])
        gt_origin = np.array([-1., 1., 0.])
        if model.name == "prismatic":
            est_axis = model.prismatic_dir
            est_origin = model.rigid_position

            # Angular error
            dot_val = np.dot(est_axis, gt_axis) / (np.linalg.norm(est_axis) * np.linalg.norm(gt_axis))
            angular_error = np.arccos(np.clip(abs(dot_val), 0, 1))

            # Position error
            v = est_origin - gt_origin
            cross = np.cross(est_axis, gt_axis)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:
                position_error = np.linalg.norm(np.cross(v, gt_axis)) / np.linalg.norm(gt_axis)
            else:
                n = cross / cross_norm
                position_error = abs(np.dot(v, n))

    # Revolute Door
    elif mode == "Revolute Door":
        gt_axis = np.array([0., 1., 0.])
        gt_origin = np.array([1., 1.5, 0.])
        if model.name == "revolute":
            q = model.rot_axis_q
            R = quaternion_to_matrix(q[0], q[1], q[2], q[3])
            est_axis = R.dot(np.array([0, 0, 1]))
            est_axis = est_axis / (np.linalg.norm(est_axis) + 1e-12)
            est_origin = model.rot_center

            # Angular error
            dot_val = np.dot(est_axis, gt_axis) / (np.linalg.norm(est_axis) * np.linalg.norm(gt_axis))
            angular_error = np.arccos(np.clip(abs(dot_val), 0, 1))

            # Position error - closest distance between lines
            v = est_origin - gt_origin
            cross = np.cross(est_axis, gt_axis)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:  # Parallel lines
                position_error = np.linalg.norm(np.cross(v, gt_axis)) / np.linalg.norm(gt_axis)
            else:
                n = cross / cross_norm
                position_error = abs(np.dot(v, n))

    # Revolute Door 2
    elif mode == "Revolute Door 2":
        gt_axis = np.array([1., 0., 0.])
        gt_origin = np.array([0.5, 2., -1.])
        if model.name == "revolute":
            q = model.rot_axis_q
            R = quaternion_to_matrix(q[0], q[1], q[2], q[3])
            est_axis = R.dot(np.array([0, 0, 1]))
            est_axis = est_axis / (np.linalg.norm(est_axis) + 1e-12)
            est_origin = model.rot_center

            # Angular error
            dot_val = np.dot(est_axis, gt_axis) / (np.linalg.norm(est_axis) * np.linalg.norm(gt_axis))
            angular_error = np.arccos(np.clip(abs(dot_val), 0, 1))

            # Position error
            v = est_origin - gt_origin
            cross = np.cross(est_axis, gt_axis)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:
                position_error = np.linalg.norm(np.cross(v, gt_axis)) / np.linalg.norm(gt_axis)
            else:
                n = cross / cross_norm
                position_error = abs(np.dot(v, n))

    # Revolute Door 3
    elif mode == "Revolute Door 3":
        gt_axis = np.array([1., 1., 0.])
        gt_axis = gt_axis / np.linalg.norm(gt_axis)  # Normalize
        gt_origin = np.array([2., 1., 1.])
        if model.name == "revolute":
            q = model.rot_axis_q
            R = quaternion_to_matrix(q[0], q[1], q[2], q[3])
            est_axis = R.dot(np.array([0, 0, 1]))
            est_axis = est_axis / (np.linalg.norm(est_axis) + 1e-12)
            est_origin = model.rot_center

            # Angular error
            dot_val = np.dot(est_axis, gt_axis) / (np.linalg.norm(est_axis) * np.linalg.norm(gt_axis))
            angular_error = np.arccos(np.clip(abs(dot_val), 0, 1))

            # Position error
            v = est_origin - gt_origin
            cross = np.cross(est_axis, gt_axis)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:
                position_error = np.linalg.norm(np.cross(v, gt_axis)) / np.linalg.norm(gt_axis)
            else:
                n = cross / cross_norm
                position_error = abs(np.dot(v, n))

    # Screw Joint
    elif mode == "Screw Joint":
        gt_axis = np.array([0., 1., 0.])
        gt_origin = np.array([0., 0., 0.])
        if model.name == "revolute" or model.name == "prismatic":
            if model.name == "revolute":
                q = model.rot_axis_q
                R = quaternion_to_matrix(q[0], q[1], q[2], q[3])
                est_axis = R.dot(np.array([0, 0, 1]))
                est_axis = est_axis / (np.linalg.norm(est_axis) + 1e-12)
                est_origin = model.rot_center
            else:  # prismatic
                est_axis = model.prismatic_dir
                est_origin = model.rigid_position

            # Angular error
            dot_val = np.dot(est_axis, gt_axis) / (np.linalg.norm(est_axis) * np.linalg.norm(gt_axis))
            angular_error = np.arccos(np.clip(abs(dot_val), 0, 1))

            # Position error
            v = est_origin - gt_origin
            cross = np.cross(est_axis, gt_axis)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:
                position_error = np.linalg.norm(np.cross(v, gt_axis)) / np.linalg.norm(gt_axis)
            else:
                n = cross / cross_norm
                position_error = abs(np.dot(v, n))

    # Screw Joint 2
    elif mode == "Screw Joint 2":
        gt_axis = np.array([1., 0., 0.])
        gt_origin = np.array([1., 0., -0.5])
        if model.name == "revolute" or model.name == "prismatic":
            if model.name == "revolute":
                q = model.rot_axis_q
                R = quaternion_to_matrix(q[0], q[1], q[2], q[3])
                est_axis = R.dot(np.array([0, 0, 1]))
                est_axis = est_axis / (np.linalg.norm(est_axis) + 1e-12)
                est_origin = model.rot_center
            else:  # prismatic
                est_axis = model.prismatic_dir
                est_origin = model.rigid_position

            # Angular error
            dot_val = np.dot(est_axis, gt_axis) / (np.linalg.norm(est_axis) * np.linalg.norm(gt_axis))
            angular_error = np.arccos(np.clip(abs(dot_val), 0, 1))

            # Position error
            v = est_origin - gt_origin
            cross = np.cross(est_axis, gt_axis)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:
                position_error = np.linalg.norm(np.cross(v, gt_axis)) / np.linalg.norm(gt_axis)
            else:
                n = cross / cross_norm
                position_error = abs(np.dot(v, n))

    # Screw Joint 3
    elif mode == "Screw Joint 3":
        gt_axis = np.array([1., 1., 0.])
        gt_axis = gt_axis / np.linalg.norm(gt_axis)  # Normalize
        gt_origin = np.array([-1., 0., 1.])
        if model.name == "revolute" or model.name == "prismatic":
            if model.name == "revolute":
                q = model.rot_axis_q
                R = quaternion_to_matrix(q[0], q[1], q[2], q[3])
                est_axis = R.dot(np.array([0, 0, 1]))
                est_axis = est_axis / (np.linalg.norm(est_axis) + 1e-12)
                est_origin = model.rot_center
            else:  # prismatic
                est_axis = model.prismatic_dir
                est_origin = model.rigid_position

            # Angular error
            dot_val = np.dot(est_axis, gt_axis) / (np.linalg.norm(est_axis) * np.linalg.norm(gt_axis))
            angular_error = np.arccos(np.clip(abs(dot_val), 0, 1))

            # Position error
            v = est_origin - gt_origin
            cross = np.cross(est_axis, gt_axis)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:
                position_error = np.linalg.norm(np.cross(v, gt_axis)) / np.linalg.norm(gt_axis)
            else:
                n = cross / cross_norm
                position_error = abs(np.dot(v, n))

    # For Planar Mouse and Ball Joint, we don't calculate axis errors
    # since their parameters are different (normal for planar, center for ball)

    return position_error, angular_error


def highlight_deviation(ps_cloud, curr_pts, prev_pts):
    """Highlight point deviations from previous frame using color coding."""
    if prev_pts is None or curr_pts.shape != prev_pts.shape:
        return
    dp = curr_pts - prev_pts
    dist = np.linalg.norm(dp, axis=1)
    dmax = dist.max()
    if dmax < 1e-12:
        c = np.full((len(dist), 3), 0.7)
        ps_cloud.add_color_quantity("Deviation Highlight", c, enabled=True)
        return
    ratio = dist / dmax
    c = np.stack([ratio, 1 - ratio, 0 * ratio], axis=1)
    ps_cloud.add_color_quantity("Deviation Highlight", c, enabled=True)


def store_points(points, mode):
    """Store point cloud data in history."""
    point_cloud_history[mode].append(points.copy())


def build_track_from_points_history(frames):
    """Build trajectory data from point cloud history."""
    tdata = []
    for f in frames:
        center = np.mean(f, axis=0)
        p = Pose(center[0], center[1], center[2], 0, 0, 0, 1)
        tdata.append(p)
    return tdata


# Global variables for visualization
fitted_points_viz = None
joint_axis_viz = None


def remove_fitted():
    """Remove fitted model visualizations."""
    global fitted_points_viz, joint_axis_viz
    if fitted_points_viz is not None:
        if ps.has_point_cloud("Fitted Model"):
            ps.remove_point_cloud("Fitted Model")
        fitted_points_viz = None
    if joint_axis_viz is not None:
        if ps.has_curve_network(joint_axis_viz):
            ps.remove_curve_network(joint_axis_viz)
        joint_axis_viz = None


# Motion parameters
TOTAL_FRAMES_PER_MODE = {m: 50 for m in modes}


def update_motion_store(mode):
    """Update motion for the current mode and store frame data."""
    fidx = frame_count[mode]
    limit = TOTAL_FRAMES_PER_MODE[mode]
    if fidx >= limit:
        return

    prev = prev_positions[mode]
    newp = None

    # Synthetic motion modes:
    if mode == "Prismatic Door":
        alpha = fidx / (limit - 1)
        shift = alpha * 5.0
        newpts = translate_points(prismatic_template.copy(), shift, np.array([1., 0., 0.]))
        if noise_sigma > 0:
            newpts += np.random.normal(0, noise_sigma, newpts.shape)
        ps_prismatic_door.update_point_positions(newpts)
        newp = newpts

    elif mode == "Revolute Door":
        alpha = fidx / (limit - 1)
        angle = -math.radians(45) + alpha * math.radians(90)
        newpts = rotate_points(revolute_template.copy(), angle, np.array([0, 1, 0]), np.array([1, 1.5, 0]))
        if noise_sigma > 0:
            newpts += np.random.normal(0, noise_sigma, newpts.shape)
        ps_revolute_door.update_point_positions(newpts)
        newp = newpts

    elif mode == "Planar Mouse":
        limit_pm = 50
        if fidx < 20:
            a = fidx / 19.0
            tx = a * 1.0
            tz = a * 1.0
            mp = mouse_template.copy()
            mp += np.array([tx, 0., tz])
            newp = mp
        else:
            a = (fidx - 20) / (limit_pm - 1 - 20)
            ang = math.radians(40) * a
            mp = mouse_template.copy()
            mp = rotate_points(mp, ang, np.array([0, 1, 0]), np.array([0, 0, 0]))
            newp = mp
        if noise_sigma > 0:
            newp += np.random.normal(0, noise_sigma, newp.shape)
        ps_mouse.update_point_positions(newp)

    elif mode == "Ball Joint":
        alpha = min(1.0, fidx / (limit - 1))
        rx = alpha * math.radians(60)
        ry = alpha * math.radians(30)
        rz = alpha * math.radians(80)
        mp = ball_template.copy()
        mp = rotate_points_xyz(mp, rx, ry, rz, np.array([0, 0, 0]))
        if noise_sigma > 0:
            mp += np.random.normal(0, noise_sigma, mp.shape)
        ps_ball.update_point_positions(mp)
        newp = mp

    elif mode == "Screw Joint":
        alpha = min(1.0, fidx / (limit - 1))
        angle = alpha * (2 * math.pi)
        newpts = apply_screw_motion(screw_template.copy(), angle, np.array([0, 1, 0]),
                                    np.array([0, 0, 0]), 0.5)
        if noise_sigma > 0:
            newpts += np.random.normal(0, noise_sigma, newpts.shape)
        ps_screw.update_point_positions(newpts)
        newp = newpts

    elif mode == "Prismatic Door 2":
        alpha = min(1.0, fidx / (limit - 1))
        shift = alpha * 4.0
        newpts = translate_points(prismatic_template_2.copy(), shift, np.array([0, 1, 0]))
        if noise_sigma > 0:
            newpts += np.random.normal(0, noise_sigma, newpts.shape)
        ps_prismatic_door_2.update_point_positions(newpts)
        newp = newpts

    elif mode == "Revolute Door 2":
        alpha = min(1.0, fidx / (limit - 1))
        angle = alpha * math.radians(90)
        hinge = np.array([0.5, 2., -1.])
        axis = np.array([1., 0., 0.])
        newpts = rotate_points(revolute_template_2.copy(), angle, axis, hinge)
        if noise_sigma > 0:
            newpts += np.random.normal(0, noise_sigma, newpts.shape)
        ps_revolute_door_2.update_point_positions(newpts)
        newp = newpts

    elif mode == "Planar Mouse 2":
        alpha = min(1.0, fidx / (limit - 1))
        shift = alpha * 2.0
        angle = alpha * math.radians(45)
        mp = mouse_template_2.copy()
        mp += np.array([0, shift, 0])
        mp = rotate_points_xyz(mp, 0, angle, 0, np.array([1, 0, 1]))
        if noise_sigma > 0:
            mp += np.random.normal(0, noise_sigma, mp.shape)
        ps_mouse_2.update_point_positions(mp)
        newp = mp

    elif mode == "Ball Joint 2":
        alpha = min(1.0, fidx / (limit - 1))
        rx = alpha * math.radians(40)
        ry = alpha * math.radians(20)
        mp = ball_template_2.copy()
        mp = rotate_points_xyz(mp, rx, ry, 0, np.array([1, 0, 0]))
        if noise_sigma > 0:
            mp += np.random.normal(0, noise_sigma, mp.shape)
        ps_ball_2.update_point_positions(mp)
        newp = mp

    elif mode == "Screw Joint 2":
        alpha = min(1.0, fidx / (limit - 1))
        angle = alpha * (2 * math.pi)
        newpts = apply_screw_motion(screw_template_2.copy(), angle, np.array([1, 0, 0]),
                                    np.array([1, 0, -0.5]), 0.7)
        if noise_sigma > 0:
            newpts += np.random.normal(0, noise_sigma, newpts.shape)
        ps_screw_2.update_point_positions(newpts)
        newp = newpts

    elif mode == "Prismatic Door 3":
        alpha = min(1.0, fidx / (limit - 1))
        shift = alpha * 3.0
        newpts = translate_points(prismatic_template_3.copy(), shift, np.array([0, 0, 1]))
        if noise_sigma > 0:
            newpts += np.random.normal(0, noise_sigma, newpts.shape)
        ps_prismatic_door_3.update_point_positions(newpts)
        newp = newpts

    elif mode == "Revolute Door 3":
        alpha = min(1.0, fidx / (limit - 1))
        angle = alpha * math.radians(80)
        hinge = np.array([2., 1., 1.])
        axis = np.array([1., 1., 0.])
        axis = axis / np.linalg.norm(axis)
        newpts = rotate_points(revolute_template_3.copy(), angle, axis, hinge)
        if noise_sigma > 0:
            newpts += np.random.normal(0, noise_sigma, newpts.shape)
        ps_revolute_door_3.update_point_positions(newpts)
        newp = newpts

    elif mode == "Planar Mouse 3":
        alpha = min(1.0, fidx / (limit - 1))
        shift = alpha * 2.0
        ang = alpha * math.radians(60)
        mp = mouse_template_3.copy()
        mp += np.array([shift, 0, 0.5 * shift])
        mp = rotate_points_xyz(mp, 0, ang, 0, np.array([-1, 0, 1]))
        if noise_sigma > 0:
            mp += np.random.normal(0, noise_sigma, mp.shape)
        ps_mouse_3.update_point_positions(mp)
        newp = mp

    elif mode == "Ball Joint 3":
        alpha = min(1.0, fidx / (limit - 1))
        rx = alpha * math.radians(40)
        ry = alpha * math.radians(60)
        rz = alpha * math.radians(20)
        mp = ball_template_3.copy()
        mp = rotate_points_xyz(mp, rx, ry, rz, np.array([1, 1, 0]))
        if noise_sigma > 0:
            mp += np.random.normal(0, noise_sigma, mp.shape)
        ps_ball_3.update_point_positions(mp)
        newp = mp

    elif mode == "Screw Joint 3":
        alpha = min(1.0, fidx / (limit - 1))
        angle = alpha * (2 * math.pi)
        newpts = apply_screw_motion(screw_template_3.copy(), angle, np.array([1, 1, 0]),
                                    np.array([-1, 0, 1]), 0.6)
        if noise_sigma > 0:
            newpts += np.random.normal(0, noise_sigma, newpts.shape)
        ps_screw_3.update_point_positions(newpts)
        newp = newpts

    frame_count[mode] += 1
    if newp is not None:
        store_points(newp, mode)
        if prev is not None:
            highlight_deviation(ps.get_point_cloud(mode), newp, prev)
        prev_positions[mode] = newp.copy()


def remove_joint_visual():
    """Remove previously drawn joint lines from Polyscope."""
    for name in [
        "Joint Axis", "Joint Origin", "Prismatic Axis", "Revolute Axis",
        "Revolute Origin", "GP Axis"
    ]:
        if ps.has_curve_network(name):
            ps.remove_curve_network(name)
    if ps.has_point_cloud("JointOriginPC"):
        ps.remove_point_cloud("JointOriginPC")


def show_joint_visual(model, length_scale=1.0, radius_axis=0.02, radius_origin=0.03):
    """Visualize the joint parameters in Polyscope.

    Args:
        model: The joint model instance
        length_scale: Scale factor for axis length visualization
        radius_axis: Radius for the axis curve
        radius_origin: Radius for the origin point
    """
    # First remove any existing visualizations
    remove_joint_visual()

    # Handle different model types
    if model.name == "prismatic":
        # Extract parameters
        pos = model.rigid_position
        dir_ = model.prismatic_dir

        # Create a segment for visualization
        seg_nodes = np.vstack([
            pos,
            pos + dir_ * length_scale
        ])
        seg_edges = np.array([[0, 1]])

        net = ps.register_curve_network("Prismatic Axis", seg_nodes, seg_edges)
        net.set_radius(radius_axis)
        net.set_color((0., 1., 1.))  # Cyan color for prismatic

    elif model.name == "revolute":
        # Extract parameters
        axis_q = model.rot_axis_q
        R = quaternion_to_matrix(axis_q[0], axis_q[1], axis_q[2], axis_q[3])
        axis_ = R.dot(np.array([0, 0, 1]))
        axis_ = axis_ / (np.linalg.norm(axis_) + 1e-12)
        center_ = model.rot_center

        # Create line segment along the axis
        seg_nodes = np.vstack([
            center_ - axis_ * length_scale,
            center_ + axis_ * length_scale
        ])
        seg_edges = np.array([[0, 1]])

        # Draw the axis
        axis_net = ps.register_curve_network("Revolute Axis", seg_nodes, seg_edges)
        axis_net.set_radius(radius_axis)
        axis_net.set_color((1., 1., 0.))  # Yellow color for revolute axis

        # Draw a point at the origin
        origin_pc = ps.register_point_cloud("JointOriginPC", center_.reshape(1, 3))
        origin_pc.set_radius(radius_origin)
        origin_pc.set_color((1., 0., 0.))  # Red color for revolute origin

    elif model.name == "gp":
        # Extract parameters
        anchor = np.array([model.anchor.x, model.anchor.y, model.anchor.z])
        axis = model.axis.cpu().numpy() if hasattr(model.axis, 'cpu') else model.axis

        # Create a segment for visualization
        seg_nodes = np.vstack([
            anchor,
            anchor + axis * length_scale
        ])
        seg_edges = np.array([[0, 1]])

        net = ps.register_curve_network("GP Axis", seg_nodes, seg_edges)
        net.set_radius(radius_axis)
        net.set_color((0., 0.5, 1.))  # Light blue for GP


def show_ground_truth_visual(mode, enable=True):
    """Enable/disable ground-truth visualization for synthetic test cases."""
    gt_name_axis = f"GT Axis"
    gt_name_origin = f"GT Origin"

    # Clear any existing GT visualizations
    if ps.has_curve_network(gt_name_axis):
        ps.remove_curve_network(gt_name_axis)
    if ps.has_curve_network(gt_name_origin):
        ps.remove_curve_network(gt_name_origin)
    if ps.has_point_cloud(gt_name_origin):
        ps.remove_point_cloud(gt_name_origin)

    if not enable:
        return

    # Add GT visualizations based on mode
    if mode == "Prismatic Door":
        axis_np = np.array([1., 0., 0.])
        origin_np = np.array([0., 0., 0.])
        seg_nodes = np.vstack([origin_np, origin_np + axis_np])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
        net.set_radius(0.02)
        net.set_color((0., 0.7, 0.7))  # Light cyan

    elif mode == "Revolute Door":
        axis_np = np.array([0., 1., 0.])
        origin_np = np.array([1., 1.5, 0.])
        seg_nodes = np.vstack([origin_np - axis_np * 0.5, origin_np + axis_np * 0.5])
        seg_edges = np.array([[0, 1]])
        net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
        net.set_radius(0.02)
        net.set_color((0.7, 0.7, 0.))  # Light yellow

        origin_pc = ps.register_point_cloud(gt_name_origin, origin_np.reshape(1, 3))
        origin_pc.set_radius(0.03)
        origin_pc.set_color((0.7, 0., 0.))  # Light red


# Global state variables
running = False
current_mode = "Prismatic Door"
best_model_name = ""
best_model_bic = 1e9
best_pos_err = 999.
best_ori_err = 999.
all_model_info = []


def callback():
    """Main UI callback function for Polyscope interface."""
    global current_mode, running
    global best_model_name, best_model_bic, best_pos_err, best_ori_err
    global fitted_points_viz, all_model_info
    global best_axis_pos_err, best_axis_ang_err  # New global variables

    # Mode selection combo box
    changed = psim.BeginCombo("Object Mode", current_mode)
    if changed:
        for m in modes:
            _, sel = psim.Selectable(m, current_mode == m)
            if sel and m != current_mode:
                remove_fitted()
                remove_joint_visual()
                show_ground_truth_visual(current_mode, enable=False)
                restore_shape(m)
                current_mode = m
                show_ground_truth_visual(m, enable=True)
        psim.EndCombo()

    psim.Separator()

    # Noise parameter control
    changed_noise, new_n = psim.InputFloat("Noise Sigma", noise_sigma, 0.001)
    if changed_noise:
        globals()['noise_sigma'] = max(0, new_n)

    # Start button
    if psim.Button("Start"):
        restore_shape(current_mode)
        remove_fitted()
        remove_joint_visual()
        show_ground_truth_visual(current_mode, enable=True)
        running = True

    # Update motion if running
    if running:
        if frame_count[current_mode] < TOTAL_FRAMES_PER_MODE[current_mode]:
            update_motion_store(current_mode)
        else:
            running = False

    # Model fitting and evaluation
    frames = point_cloud_history[current_mode]
    if len(frames) >= 2:
        data = build_track_from_points_history(frames)
        model = fit_and_select_best_model(data)
        if model is not None:
            best_model_name = model.name
            best_model_bic = model.bic
            best_pos_err = model.avg_pos_err
            best_ori_err = model.avg_ori_err

            # Calculate separate errors for axes
            axis_pos_err, axis_ang_err = compute_axis_errors(model, current_mode)
            best_axis_pos_err = axis_pos_err
            best_axis_ang_err = axis_ang_err

            # Visualize fitted model
            remove_fitted()
            fitted = []
            for p in data:
                q = model.inverse_kinematics(p)
                pf = model.forward_kinematics(q)
                fitted.append([pf.x, pf.y, pf.z])
            fitted = np.array(fitted)
            pc = ps.register_point_cloud("Fitted Model", fitted)
            pc.set_enabled(True)

            # Show joint visualization for the best model
            show_joint_visual(model)

            all_model_info = [(model.name, model.bic, model.avg_pos_err, model.avg_ori_err,
                               axis_pos_err, axis_ang_err)]
        else:
            all_model_info = []

    # Display model information
    psim.TextUnformatted("Models BIC / Errors:")
    for info in all_model_info:
        if len(info) >= 6:  # Ensure all values are available
            nm, bic, pe, oe, ax_pos, ax_ang = info
            psim.TextUnformatted(f"  {nm} => BIC={bic:.3f}")
            psim.TextUnformatted(f"    PosErr={pe:.3f}, OriErr={oe:.3f}")
            psim.TextUnformatted(f"    Axis Position Error={ax_pos:.3f}")
            psim.TextUnformatted(f"    Axis Angular Error={ax_ang:.3f} rad ({np.degrees(ax_ang):.1f}°)")
        else:
            nm, bic, pe, oe = info[:4]
            psim.TextUnformatted(f"  {nm} => BIC={bic:.3f}, PosErr={pe:.3f}, OriErr={oe:.3f}")

    psim.Separator()

    # Display best model summary
    psim.TextUnformatted(f"Best Model = {best_model_name}")
    if best_model_bic < 1e9:
        psim.TextUnformatted(f"BIC = {best_model_bic:.3f}")
        psim.TextUnformatted(f"General: PosErr = {best_pos_err:.3f}, OriErr = {best_ori_err:.3f}")
        psim.TextUnformatted(
            f"Axis: Position Error = {best_axis_pos_err:.3f}, Angular Error = {best_axis_ang_err:.3f} rad ({np.degrees(best_axis_ang_err):.1f}°)")
        psim.TextUnformatted("Error calculation method: ")
        psim.TextUnformatted("  General position error = Euclidean distance (pose vs fit)")
        psim.TextUnformatted("  General angular error = Relative rotation angle between two pose rotation matrices")
        psim.TextUnformatted("  Axis position error = Closest distance between estimated axis and ground truth axis")
        psim.TextUnformatted(
            "  Axis angular error = Angular difference between estimated axis and ground truth axis (radians)")


def dearpygui_thread():
    """Run DearPyGUI interface in separate thread."""
    dpg.create_context()
    with dpg.window(label="Articulation Demo", width=600, height=300):
        dpg.add_text(
            "This example demonstrates joint model fitting from the paper.\nPolyscope window is used for 3D visualization.")
    dpg.create_viewport(title='DearPyGUI - minimal', width=600, height=300)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


def main():
    """Main function to run the demonstration."""
    th = threading.Thread(target=dearpygui_thread, daemon=True)
    th.start()
    ps.set_user_callback(callback)
    ps.set_ground_plane_mode("none")
    # Initialize ground truth visualization for the first mode
    show_ground_truth_visual(modes[0], enable=True)
    ps.show()


if __name__ == "__main__":
    main()