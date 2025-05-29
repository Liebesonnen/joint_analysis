import numpy as np
import os
import json
import re
import math
import random
import torch
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from datetime import datetime
import sys
from collections import defaultdict
import glob


###############################################################################
#                          通用辅助函数 (旋转、平移、等)                        #
###############################################################################

def quaternion_to_matrix(qx, qy, qz, qw):
    """将四元数转换为 3x3 旋转矩阵（保证四元数归一化）"""
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
    """从旋转矩阵中提取旋转角度"""
    trace = np.trace(R)
    val = (trace - 1.0) / 2.0
    val = max(-1.0, min(1.0, val))
    return abs(math.acos(val))


###############################################################################
#                          定义 Pose 类                                      #
###############################################################################

class Pose:
    def __init__(self, x=0, y=0, z=0, qx=0, qy=0, qz=0, qw=1):
        self.x = x
        self.y = y
        self.z = z
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw


###############################################################################
#                    定义关节模型基类 (GenericLinkModel)                      #
###############################################################################

def log_outlier_likelihood():
    return -math.log(1e5)


def log_inlier_likelihood(obs_pose, pred_pose, sigma_pos, sigma_ori):
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
    def __init__(self):
        self._dofs = 0
        self.k = 6
        self.name = "generic"
        self.sigma_pos = 0.005
        self.sigma_ori = math.radians(360)
        self.gamma = 0.3
        self.sac_iterations = 50
        self.optimizer_iterations = 10

        self.loglikelihood = -1e10
        self.bic = 1e10
        self.avg_pos_err = 999.
        self.avg_ori_err = 999.

        self.state_params = None

    def dofs(self):
        return self._dofs

    def sample_size(self):
        return 1

    def guess_from_minimal_samples(self, samples):
        return False

    def refine_nonlinear(self, data):
        pass

    def forward_kinematics(self, q):
        return Pose()

    def inverse_kinematics(self, pose):
        return np.array([])

    def log_inlier_ll_single(self, obs):
        q = self.inverse_kinematics(obs)
        pred = self.forward_kinematics(q)
        return log_inlier_likelihood(obs, pred, self.sigma_pos, self.sigma_ori)

    def compute_log_likelihood(self, data, estimate_gamma=False):
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
        return {
            'sigma_pos': self.sigma_pos,
            'sigma_ori': self.sigma_ori,
            'gamma': self.gamma,
            'state_params': self.state_params
        }

    def load_state(self, st):
        self.sigma_pos = st['sigma_pos']
        self.sigma_ori = st['sigma_ori']
        self.gamma = st['gamma']
        self.state_params = st['state_params']


###############################################################################
#                    各具体模型：Rigid, Prismatic, Revolute                   #
###############################################################################

class RigidModel(GenericLinkModel):
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


class RevoluteModel(GenericLinkModel):
    def __init__(self):
        super().__init__()
        self.name = "revolute"
        self._dofs = 1
        self.k = 12
        self.rot_mode = 0  # 0:基于位置, 1:基于姿态
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
        R = quaternion_to_matrix(pose.qx, pose.qy, pose.qz, pose.qw)
        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = torch.tensor(R, dtype=torch.float32)
        T[0, 3] = pose.x
        T[1, 3] = pose.y
        T[2, 3] = pose.z
        return T

    def matrix_to_pose(self, M: torch.Tensor) -> Pose:
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
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return torch.tensor([x, y, z, w], dtype=torch.float32)

    def inverse_pose(self, M: torch.Tensor) -> torch.Tensor:
        R = M[:3, :3]
        t = M[:3, 3]
        R_inv = R.t()
        t_inv = -torch.matmul(R_inv, t)
        Minv = torch.eye(4, dtype=torch.float32)
        Minv[:3, :3] = R_inv
        Minv[:3, 3] = t_inv
        return Minv

    def transform_multiply(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
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
            loss = -ll + 0 * torch.sum(x0)
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


###############################################################################
#          模型选择函数                                                       #
###############################################################################

def fit_and_select_best_model(data):
    candidates = [RigidModel(), PrismaticModel(), RevoluteModel()]
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


def convert_point_history_to_poses(point_history):
    """将点云历史转换为位姿数据"""
    poses = []
    for frame in point_history:
        # 计算质心作为位置
        center = np.mean(frame, axis=0)
        # 创建Pose对象，姿态为单位四元数
        pose = Pose(center[0], center[1], center[2], 0, 0, 0, 1)
        poses.append(pose)
    return poses


class ImprovedJointAnalysisEvaluator:
    def __init__(self, directory_path=None, noise_std=0.0):
        """
        Initialize the evaluator

        Args:
            directory_path: Path to directory containing .npy files
            noise_std: Standard deviation for Gaussian noise (0.0 means no noise)
        """
        self.directory_path = directory_path
        self.noise_std = noise_std

        # Load ground truth joint data
        self.ground_truth_json = "/common/homes/all/uksqc_chen/projects/control/ParaHome/all_scenes_transformed_axis_pivot_data.json"
        self.ground_truth_data = self.load_ground_truth()

        # Define expected joint types for each object
        self.expected_joint_types = {
            "drawer": "prismatic",
            "microwave": "revolute",
            "refrigerator": "revolute",
            "washingmachine": "revolute",
            "trashbin": "revolute",
            "chair": "planar"
        }

        # Target objects for evaluation
        self.target_objects = ["drawer", "washingmachine", "trashbin", "microwave", "refrigerator", "chair"]

        # Store all processed data
        self.processed_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "noise_std": self.noise_std,
                "directory_path": self.directory_path,
                "ground_truth_json": self.ground_truth_json
            },
            "files": []
        }

    def load_ground_truth(self):
        """Load ground truth data from JSON file"""
        try:
            with open(self.ground_truth_json, 'r') as f:
                data = json.load(f)
                print(f"Successfully loaded ground truth data from: {self.ground_truth_json}")
                return data
        except FileNotFoundError:
            print(f"Warning: Ground truth file not found at {self.ground_truth_json}")
            return {}
        except Exception as e:
            print(f"Error loading ground truth data: {e}")
            return {}

    def add_gaussian_noise(self, data):
        """Add Gaussian noise to the data"""
        if self.noise_std <= 0:
            return data

        noise = np.random.normal(0, self.noise_std, data.shape)
        return data + noise

    def extract_scene_info_from_filename(self, filename):
        """Extract scene and object information from filename"""
        # Pattern to match filenames like "s204_drawer_part2_1320_1440"
        pattern = r'(s\d+)_([^_]+)_(part\d+|base)_(\d+)_(\d+)'
        match = re.match(pattern, filename)

        if match:
            scene, object_type, part, start_frame, end_frame = match.groups()
            return {
                "scene": scene,
                "object": object_type,
                "part": part,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "frame_range": f"{start_frame}_{end_frame}"
            }

        print(f"Warning: Could not extract scene info from filename: {filename}")
        return None

    def group_files_by_scene_and_frames(self):
        """Group files by scene and frame range"""
        if not self.directory_path:
            print("No directory path specified")
            return {}

        # Get all .npy files in directory
        file_paths = glob.glob(os.path.join(self.directory_path, "*.npy"))

        # Group files by scene + frame_range + object_type
        groups = defaultdict(list)

        for file_path in file_paths:
            filename = os.path.basename(file_path).split('.')[0]
            scene_info = self.extract_scene_info_from_filename(filename)

            if scene_info and scene_info["object"] in self.target_objects:
                # Create group key: scene_object_framerange
                group_key = f"{scene_info['scene']}_{scene_info['object']}_{scene_info['frame_range']}"
                groups[group_key].append({
                    "file_path": file_path,
                    "filename": filename,
                    "scene_info": scene_info
                })

        print(f"Found {len(groups)} unique scene-object-frame combinations")
        return groups

    def perform_joint_analysis(self, point_history):
        """使用新的关节模型进行分析"""
        # 将点云历史转换为位姿数据
        poses = convert_point_history_to_poses(point_history)

        # 使用新的模型选择方法
        best_model = fit_and_select_best_model(poses)

        if best_model is not None:
            # 提取关节参数
            joint_params = {}
            if best_model.name == "prismatic":
                joint_params = {
                    "axis": best_model.prismatic_dir.tolist(),
                    "origin": best_model.rigid_position.tolist(),
                    "orientation": best_model.rigid_orientation_q.tolist()
                }
            elif best_model.name == "revolute":
                # 从四元数获取旋转轴
                axis_q = best_model.rot_axis_q.numpy() if hasattr(best_model.rot_axis_q,
                                                                  'numpy') else best_model.rot_axis_q
                R = quaternion_to_matrix(axis_q[0], axis_q[1], axis_q[2], axis_q[3])
                axis = R.dot(np.array([0, 0, 1]))
                center = best_model.rot_center.numpy() if hasattr(best_model.rot_center,
                                                                  'numpy') else best_model.rot_center

                joint_params = {
                    "axis": axis.tolist(),
                    "origin": center.tolist(),
                    "radius": best_model.rot_radius,
                    "axis_quaternion": axis_q.tolist()
                }
            elif best_model.name == "rigid":
                joint_params = {
                    "position": [best_model.offset.x, best_model.offset.y, best_model.offset.z],
                    "orientation": [best_model.offset.qx, best_model.offset.qy, best_model.offset.qz,
                                    best_model.offset.qw]
                }

            return joint_params, best_model.name, {
                "bic": best_model.bic,
                "loglikelihood": best_model.loglikelihood,
                "avg_pos_err": best_model.avg_pos_err,
                "avg_ori_err": best_model.avg_ori_err
            }
        else:
            print("No suitable joint model found")
            return None, "unknown", {}

    def calculate_angle_between_vectors(self, v1, v2):
        """Calculate angle between two vectors in degrees"""
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return float('inf')

        v1_normalized = v1 / v1_norm
        v2_normalized = v2 / v2_norm

        dot_product = np.clip(np.abs(np.dot(v1_normalized, v2_normalized)), 0.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        return min(angle_deg, 180 - angle_deg)

    def calculate_distance_between_lines(self, axis1, origin1, axis2, origin2):
        """Calculate minimum distance between two lines in 3D space"""
        axis1_norm = np.linalg.norm(axis1)
        axis2_norm = np.linalg.norm(axis2)

        if axis1_norm < 1e-6 or axis2_norm < 1e-6:
            return float('inf')

        axis1 = axis1 / axis1_norm
        axis2 = axis2 / axis2_norm

        cross_product = np.cross(axis1, axis2)
        cross_norm = np.linalg.norm(cross_product)

        if cross_norm < 1e-6:
            vec_to_point = origin1 - origin2
            proj_on_axis = np.dot(vec_to_point, axis2) * axis2
            perpendicular = vec_to_point - proj_on_axis
            return np.linalg.norm(perpendicular)
        else:
            vec_between = origin2 - origin1
            distance = abs(np.dot(vec_between, cross_product)) / cross_norm
            return distance

    def get_ground_truth_for_object(self, scene_info):
        """Get ground truth data for a specific object"""
        if not scene_info:
            return None

        scene = scene_info["scene"]
        object_type = scene_info["object"]
        part = scene_info["part"]

        # Special case for chair - always use (0,0,1) as ground truth normal
        if object_type == "chair":
            return {
                "axis": [0.0, 0.0, 1.0],
                "pivot": [0.0, 0.0, 0.0]
            }

        if scene not in self.ground_truth_data:
            return None
        if object_type not in self.ground_truth_data[scene]:
            return None
        if part not in self.ground_truth_data[scene][object_type]:
            return None

        return self.ground_truth_data[scene][object_type][part]

    def calculate_joint_errors(self, joint_type, estimated_params, ground_truth):
        """Calculate errors based on joint type"""
        errors = {}

        if joint_type == "prismatic":
            est_axis = np.array(estimated_params.get("axis", [0, 0, 0]))
            gt_axis = np.array(ground_truth["axis"])
            angle_error = self.calculate_angle_between_vectors(est_axis, gt_axis)
            errors["axis_angle_error_degrees"] = angle_error

        elif joint_type == "revolute":
            est_axis = np.array(estimated_params.get("axis", [0, 0, 0]))
            est_origin = np.array(estimated_params.get("origin", [0, 0, 0]))
            gt_axis = np.array(ground_truth["axis"])

            # Handle ground truth pivot
            gt_pivot = ground_truth["pivot"]
            if isinstance(gt_pivot, list):
                if len(gt_pivot) == 1:
                    gt_origin = np.array([0., 0., 0.]) + float(gt_pivot[0]) * gt_axis
                elif len(gt_pivot) == 3:
                    gt_origin = np.array(gt_pivot)
                else:
                    gt_origin = np.array([0., 0., 0.])
            else:
                gt_origin = np.array([0., 0., 0.]) + float(gt_pivot) * gt_axis

            angle_error = self.calculate_angle_between_vectors(est_axis, gt_axis)
            axis_distance = self.calculate_distance_between_lines(est_axis, est_origin, gt_axis, gt_origin)

            errors["axis_angle_error_degrees"] = angle_error
            errors["axis_distance_meters"] = axis_distance

        elif joint_type == "planar":
            est_normal = np.array(estimated_params.get("normal", [0, 0, 0]))
            gt_normal = np.array(ground_truth["axis"])
            angle_error = self.calculate_angle_between_vectors(est_normal, gt_normal)
            errors["normal_angle_error_degrees"] = angle_error

        return errors

    def evaluate_group(self, group_files):
        """Evaluate a group of files (same scene, object, frame range)"""
        group_key = f"{group_files[0]['scene_info']['scene']}_{group_files[0]['scene_info']['object']}_{group_files[0]['scene_info']['frame_range']}"
        object_type = group_files[0]['scene_info']['object']
        expected_joint_type = self.expected_joint_types[object_type]

        print(f"\nEvaluating group: {group_key}")
        print(f"Files in group: {[f['filename'] for f in group_files]}")

        group_result = {
            "group_key": group_key,
            "object_type": object_type,
            "expected_joint_type": expected_joint_type,
            "files_in_group": len(group_files),
            "file_results": [],
            "group_classification_correct": False,
            "best_result": None
        }

        # Evaluate each file in the group
        for file_info in group_files:
            file_result = self.evaluate_single_file(file_info)
            group_result["file_results"].append(file_result)

            # If any file in the group has correct classification, mark group as correct
            if file_result.get("classification_correct", False):
                group_result["group_classification_correct"] = True
                if group_result["best_result"] is None:
                    group_result["best_result"] = file_result

        return group_result

    def evaluate_single_file(self, file_info):
        """Evaluate a single file"""
        file_path = file_info["file_path"]
        filename = file_info["filename"]
        scene_info = file_info["scene_info"]

        object_type = scene_info["object"]
        expected_joint_type = self.expected_joint_types[object_type]

        # Load and process data
        try:
            data = np.load(file_path)

            # Add Gaussian noise if specified
            data = self.add_gaussian_noise(data)

            # Apply Savitzky-Golay filter to smooth data
            data_filter = savgol_filter(
                x=data, window_length=21, polyorder=2, deriv=0, axis=0, delta=0.1
            )

        except Exception as e:
            result = {
                "filename": filename,
                "scene_info": scene_info,
                "error": f"Could not load or process file: {e}"
            }
            self.processed_data["files"].append(result)
            return result

        # Perform joint analysis using new models
        try:
            joint_params, best_joint, info_dict = self.perform_joint_analysis(data_filter)
        except Exception as e:
            result = {
                "filename": filename,
                "scene_info": scene_info,
                "error": f"Joint analysis failed: {e}"
            }
            self.processed_data["files"].append(result)
            return result

        # Initialize result
        result = {
            "filename": filename,
            "scene_info": scene_info,
            "expected_joint_type": expected_joint_type,
            "detected_joint_type": best_joint,
            "classification_correct": best_joint == expected_joint_type,
            "joint_params": joint_params if joint_params else {},
            "analysis_info": info_dict
        }

        # Calculate errors if classification is correct
        if result["classification_correct"]:
            ground_truth = self.get_ground_truth_for_object(scene_info)
            if ground_truth:
                result["ground_truth"] = ground_truth
                result["errors"] = self.calculate_joint_errors(best_joint, joint_params, ground_truth)

        # Store processed data
        self.processed_data["files"].append(result)
        return result

    def evaluate_all_groups(self):
        """Evaluate all groups and return comprehensive results"""
        print("Starting improved joint analysis evaluation...")
        print(f"Using Gaussian noise with std = {self.noise_std}")

        # Group files
        file_groups = self.group_files_by_scene_and_frames()

        if not file_groups:
            print("No valid file groups found!")
            return None

        print(f"Total groups to evaluate: {len(file_groups)}")

        all_group_results = []

        # Statistics for each object type
        object_stats = {}
        for obj_type in self.target_objects:
            object_stats[obj_type] = {
                "total_groups": 0,
                "successful_groups": 0,
                "success_rate": 0.0,
                "errors": {
                    "angle_errors": [],
                    "distance_errors": []  # For revolute joints only
                }
            }

        # Evaluate each group
        for group_key, group_files in file_groups.items():
            group_result = self.evaluate_group(group_files)
            all_group_results.append(group_result)

            object_type = group_result["object_type"]
            object_stats[object_type]["total_groups"] += 1

            if group_result["group_classification_correct"]:
                object_stats[object_type]["successful_groups"] += 1

                # Collect errors from the best result
                best_result = group_result["best_result"]
                if best_result and "errors" in best_result:
                    errors = best_result["errors"]

                    # Collect angle errors
                    if "axis_angle_error_degrees" in errors:
                        object_stats[object_type]["errors"]["angle_errors"].append(errors["axis_angle_error_degrees"])
                    elif "normal_angle_error_degrees" in errors:
                        object_stats[object_type]["errors"]["angle_errors"].append(errors["normal_angle_error_degrees"])

                    # Collect distance errors (revolute joints only)
                    if "axis_distance_meters" in errors:
                        object_stats[object_type]["errors"]["distance_errors"].append(errors["axis_distance_meters"])

        # Calculate success rates and average errors
        for obj_type in self.target_objects:
            stats = object_stats[obj_type]
            if stats["total_groups"] > 0:
                stats["success_rate"] = stats["successful_groups"] / stats["total_groups"] * 100

            # Calculate average errors
            if stats["errors"]["angle_errors"]:
                stats["average_angle_error"] = np.mean(stats["errors"]["angle_errors"])
                stats["std_angle_error"] = np.std(stats["errors"]["angle_errors"])
            else:
                stats["average_angle_error"] = None
                stats["std_angle_error"] = None

            if stats["errors"]["distance_errors"]:
                stats["average_distance_error"] = np.mean(stats["errors"]["distance_errors"])
                stats["std_distance_error"] = np.std(stats["errors"]["distance_errors"])
            else:
                stats["average_distance_error"] = None
                stats["std_distance_error"] = None

        # Overall statistics
        total_groups = len(all_group_results)
        successful_groups = sum(1 for gr in all_group_results if gr["group_classification_correct"])
        overall_success_rate = (successful_groups / total_groups * 100) if total_groups > 0 else 0

        final_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "noise_std": self.noise_std,
            "overall_statistics": {
                "total_groups": total_groups,
                "successful_groups": successful_groups,
                "overall_success_rate": overall_success_rate
            },
            "object_statistics": object_stats,
            "detailed_group_results": all_group_results
        }

        return final_results

    def convert_numpy_to_python(self, obj):
        """Convert numpy arrays to Python lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_python(item) for item in obj]
        else:
            return obj

    def save_results(self, results, output_file):
        """Save evaluation results to JSON file"""
        try:
            serializable_results = self.convert_numpy_to_python(results)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

    def save_processed_data(self, output_file):
        """Save all processed data to JSON file"""
        try:
            serializable_data = self.convert_numpy_to_python(self.processed_data)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            print(f"\nProcessed data saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Error saving processed data: {e}")
            return False

    def print_summary(self, results):
        """Print a summary of the evaluation results"""
        print("\n" + "=" * 80)
        print("IMPROVED EVALUATION SUMMARY")
        print("=" * 80)

        overall_stats = results["overall_statistics"]
        print(f"Noise level (std): {results['noise_std']}")
        print(f"Total groups evaluated: {overall_stats['total_groups']}")
        print(f"Successful groups: {overall_stats['successful_groups']}")
        print(f"Overall success rate: {overall_stats['overall_success_rate']:.1f}%")

        print(f"\nDetailed Results by Object Type:")
        print("-" * 50)

        object_stats = results["object_statistics"]
        for obj_type in self.target_objects:
            stats = object_stats[obj_type]
            print(f"\n{obj_type.upper()}:")
            print(
                f"  Groups: {stats['successful_groups']}/{stats['total_groups']} (Success rate: {stats['success_rate']:.1f}%)")

            if stats["average_angle_error"] is not None:
                print(f"  Average angle error: {stats['average_angle_error']:.2f}° (±{stats['std_angle_error']:.2f}°)")
            else:
                print(f"  Average angle error: N/A")

            if stats["average_distance_error"] is not None:
                print(
                    f"  Average axis distance error: {stats['average_distance_error']:.4f}m (±{stats['std_distance_error']:.4f}m)")
            else:
                print(f"  Average axis distance error: N/A")


def main():
    """Main function to run the evaluation"""
    # Configuration
    directory_path = "/common/homes/all/uksqc_chen/projects/control/ParaHome/output_specific_actions"
    noise_levels = [0.01]  # Different noise levels to test

    for noise_std in noise_levels:
        print(f"\n{'=' * 60}")
        print(f"Running evaluation with noise std = {noise_std}")
        print(f"{'=' * 60}")

        # Create evaluator
        evaluator = ImprovedJointAnalysisEvaluator(directory_path, noise_std=noise_std)

        # Perform evaluation
        results = evaluator.evaluate_all_groups()

        if results is None:
            print("Evaluation failed!")
            continue

        # Print summary
        evaluator.print_summary(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        noise_str = f"noise_{noise_std:g}".replace('.', 'p')

        # Save evaluation results
        output_file = f"joint_analysis_evaluation_{noise_str}_{timestamp}.json"
        success = evaluator.save_results(results, output_file)

        # Save processed data
        processed_data_file = f"processed_joint_data_{noise_str}_{timestamp}.json"
        processed_success = evaluator.save_processed_data(processed_data_file)

        if success and processed_success:
            print(f"\n✓ Evaluation completed successfully!")
            print(f"✓ Results saved to: {output_file}")
            print(f"✓ Processed data saved to: {processed_data_file}")
        else:
            print(f"\n✗ Failed to save some files")


if __name__ == "__main__":
    main()