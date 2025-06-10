import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from io import BytesIO
import os
import json
import math
import random
from datetime import datetime
from robot_utils.viz.polyscope import PolyscopeUtils, ps, psim, register_point_cloud, draw_frame_3d
from scipy.optimize import minimize
import torch


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


class EnhancedViz:
    def __init__(self, file_paths=None):
        # Initialize default file paths list if none provided
        if file_paths is None:
            self.file_paths = ["./demo_data/s1_gasstove_part2_1110_1170.npy"]
        else:
            self.file_paths = file_paths

        # Create Polyscope utils instance
        self.pu = PolyscopeUtils()
        # Initialize current time frame index and change flag
        self.t, self.t_changed = 0, False
        # Initialize current point index and change flag
        self.idx_point, self.idx_point_changed = 0, False
        # Current selected dataset index
        self.current_dataset = 0
        self.noise_sigma = 0.00

        # Load ground truth joint data
        self.ground_truth_json = "./parahome_data_thesis/joint_info.json"
        self.ground_truth_data = self.load_ground_truth()
        self.show_ground_truth = False
        self.ground_truth_scale = 0.5
        # Track ground truth visualization elements
        self.gt_curve_networks = []
        self.gt_point_clouds = []

        # Store multiple datasets dictionary
        self.datasets = {}
        # Load all files
        for i, file_path in enumerate(self.file_paths):
            # Give each dataset a name identifier
            dataset_name = f"dataset_{i}"
            # Load data
            data = np.load(file_path)
            # Create dataset name from filename for possible relative paths
            display_name = os.path.basename(file_path).split('.')[0]
            if self.noise_sigma > 0:
                data += np.random.randn(*data.shape) * self.noise_sigma
            # Store dataset information
            self.datasets[dataset_name] = {
                "path": file_path,
                "display_name": display_name,
                "data": data,
                "T": data.shape[0],  # Number of time frames
                "N": data.shape[1],  # Number of points per frame
                "visible": True  # Whether to display
            }

            # Apply Savitzky-Golay filter to smooth data
            self.datasets[dataset_name]["data_filter"] = savgol_filter(
                x=data, window_length=21, polyorder=2, deriv=0, axis=0, delta=0.1
            )

            # Use Savitzky-Golay filter to calculate velocity (first derivative)
            self.datasets[dataset_name]["dv_filter"] = savgol_filter(
                x=data, window_length=21, polyorder=2, deriv=1, axis=0, delta=0.1
            )

        # Use first dataset as current dataset
        dataset_keys = list(self.datasets.keys())
        if dataset_keys:
            self.current_dataset_key = dataset_keys[0]
            current_data = self.datasets[self.current_dataset_key]
            # Get dimensions of current dataset
            self.T = current_data["T"]
            self.N = current_data["N"]
            # Current active data
            self.d = current_data["data"]
            self.d_filter = current_data["data_filter"]
            self.dv_filter = current_data["dv_filter"]
        else:
            print("No datasets loaded.")
            return

        # Average time step
        self.dt_mean = 0
        # Number of neighbors for SVD computation
        self.num_neighbors = 50

        # Create folder for saving images
        self.output_dir = "visualization_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Coordinate system settings
        self.coord_scale = 0.1

        # Calculate angular velocity
        for dataset_key in self.datasets:
            dataset = self.datasets[dataset_key]
            angular_velocity_raw, angular_velocity_filtered = self.calculate_angular_velocity(
                dataset["data_filter"], dataset["N"]
            )
            dataset["angular_velocity_raw"] = angular_velocity_raw
            dataset["angular_velocity_filtered"] = angular_velocity_filtered

            # Perform joint analysis using new models
            joint_model, best_joint_type = self.perform_joint_analysis(dataset["data_filter"])
            dataset["joint_model"] = joint_model
            dataset["best_joint_type"] = best_joint_type

            # Try to map dataset to ground truth entry
            dataset["ground_truth_key"] = self.map_dataset_to_ground_truth(dataset["display_name"])

        # Current dataset's angular velocity
        self.angular_velocity_raw = self.datasets[self.current_dataset_key]["angular_velocity_raw"]
        self.angular_velocity_filtered = self.datasets[self.current_dataset_key]["angular_velocity_filtered"]

        # Current dataset's joint information
        self.current_joint_model = self.datasets[self.current_dataset_key]["joint_model"]
        self.current_best_joint_type = self.datasets[self.current_dataset_key]["best_joint_type"]

        # Draw origin coordinate frame
        draw_frame_3d(np.zeros(6), label="origin", scale=0.1)

        # Register initial point cloud for all datasets and display
        for dataset_key, dataset in self.datasets.items():
            if dataset["visible"]:
                register_point_cloud(
                    dataset["display_name"],
                    dataset["data_filter"][self.t],
                    radius=0.01,
                    enabled=True
                )

        # Reset view bounds to include all point clouds
        visible_point_clouds = [dataset["data_filter"][self.t] for dataset in self.datasets.values()
                                if dataset["visible"]]
        if visible_point_clouds:
            self.pu.reset_bbox_from_pcl_list(visible_point_clouds)

        # Visualize joint parameters for current dataset
        self.visualize_joint_parameters()

        # Set user interaction callback function
        ps.set_user_callback(self.callback)
        # Show Polyscope interface
        ps.show()

    def load_ground_truth(self):
        """Load ground truth data from JSON file"""
        try:
            with open(self.ground_truth_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading ground truth data: {e}")
            return {}

    def map_dataset_to_ground_truth(self, display_name):
        """Map dataset name to ground truth key"""
        # Extract object type from dataset name (assuming format like "s1_refrigerator_part1_1380_1470")
        parts = display_name.split('_')
        if len(parts) >= 2:
            # Try to find the object type in the ground truth data
            object_type = parts[1].lower()  # e.g., "refrigerator"
            if object_type in self.ground_truth_data:
                # Find part number if present
                part_info = None
                if len(parts) >= 3 and parts[2].startswith("part"):
                    part_num = parts[2]  # e.g., "part1"
                    if part_num in self.ground_truth_data[object_type]:
                        part_info = part_num

                return {"object": object_type, "part": part_info}

        return None

    def perform_joint_analysis(self, point_history):
        """使用新的关节模型进行分析"""
        # 将点云历史转换为位姿数据
        poses = convert_point_history_to_poses(point_history)

        # 使用新的模型选择方法
        best_model = fit_and_select_best_model(poses)

        if best_model is not None:
            # 打印分析结果
            print("\n" + "=" * 80)
            print(f"Joint Type: {best_model.name}")
            print("=" * 80)
            print(f"BIC Score: {best_model.bic:.6f}")
            print(f"Log Likelihood: {best_model.loglikelihood:.6f}")
            print(f"Average Position Error: {best_model.avg_pos_err:.6f}")
            print(f"Average Orientation Error: {best_model.avg_ori_err:.6f}")

            # 打印关节参数
            if best_model.name == "prismatic":
                print(f"Prismatic Joint Parameters:")
                print(
                    f"Position: [{best_model.rigid_position[0]:.6f}, {best_model.rigid_position[1]:.6f}, {best_model.rigid_position[2]:.6f}]")
                print(
                    f"Direction: [{best_model.prismatic_dir[0]:.6f}, {best_model.prismatic_dir[1]:.6f}, {best_model.prismatic_dir[2]:.6f}]")
                print(
                    f"Orientation: [{best_model.rigid_orientation_q[0]:.6f}, {best_model.rigid_orientation_q[1]:.6f}, {best_model.rigid_orientation_q[2]:.6f}, {best_model.rigid_orientation_q[3]:.6f}]")

            elif best_model.name == "revolute":
                print(f"Revolute Joint Parameters:")
                print(
                    f"Center: [{best_model.rot_center[0]:.6f}, {best_model.rot_center[1]:.6f}, {best_model.rot_center[2]:.6f}]")
                print(
                    f"Axis Quaternion: [{best_model.rot_axis_q[0]:.6f}, {best_model.rot_axis_q[1]:.6f}, {best_model.rot_axis_q[2]:.6f}, {best_model.rot_axis_q[3]:.6f}]")
                print(f"Radius: {best_model.rot_radius:.6f}")
                print(
                    f"Orientation: [{best_model.rot_orientation_q[0]:.6f}, {best_model.rot_orientation_q[1]:.6f}, {best_model.rot_orientation_q[2]:.6f}, {best_model.rot_orientation_q[3]:.6f}]")

            elif best_model.name == "rigid":
                print(f"Rigid Joint Parameters:")
                print(f"Position: [{best_model.offset.x:.6f}, {best_model.offset.y:.6f}, {best_model.offset.z:.6f}]")
                print(
                    f"Orientation: [{best_model.offset.qx:.6f}, {best_model.offset.qy:.6f}, {best_model.offset.qz:.6f}, {best_model.offset.qw:.6f}]")

            print("=" * 80 + "\n")

            return best_model, best_model.name
        else:
            print("No suitable joint model found")
            return None, "unknown"

    def visualize_joint_parameters(self):
        """Visualize estimated joint parameters in Polyscope"""
        # Remove any existing joint visualizations
        self.remove_joint_visualization()

        joint_model = self.current_joint_model
        if joint_model is not None:
            self.show_joint_visualization(joint_model)

    def remove_joint_visualization(self):
        """Remove all joint visualization elements."""
        # Remove curve networks
        for name in [
            "Prismatic Axis", "Revolute Axis", "Revolute Origin", "Rigid Position"
        ]:
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)

        # Remove point clouds used for visualization
        for name in ["RevoluteCenterPC", "RigidPositionPC"]:
            if ps.has_point_cloud(name):
                ps.remove_point_cloud(name)

    def show_joint_visualization(self, joint_model):
        """Show visualization for a specific joint type."""
        if joint_model.name == "prismatic":
            # Extract parameters
            pos = joint_model.rigid_position
            dir_ = joint_model.prismatic_dir

            # Visualize axis
            seg_nodes = np.array([pos, pos + dir_])
            seg_edges = np.array([[0, 1]])
            name = "Prismatic Axis"
            prisviz = ps.register_curve_network(name, seg_nodes, seg_edges)
            prisviz.set_color((0., 1., 1.))
            prisviz.set_radius(0.02)

        elif joint_model.name == "revolute":
            # Extract parameters
            center = joint_model.rot_center.numpy() if hasattr(joint_model.rot_center,
                                                               'numpy') else joint_model.rot_center
            axis_q = joint_model.rot_axis_q.numpy() if hasattr(joint_model.rot_axis_q,
                                                               'numpy') else joint_model.rot_axis_q

            # Get rotation axis from quaternion
            R = quaternion_to_matrix(axis_q[0], axis_q[1], axis_q[2], axis_q[3])
            axis = R.dot(np.array([0, 0, 1]))
            axis = axis / (np.linalg.norm(axis) + 1e-12)

            # Visualize axis
            seg_nodes = np.array([center - axis * 0.5, center + axis * 0.5])
            seg_edges = np.array([[0, 1]])

            name = "Revolute Axis"
            revviz = ps.register_curve_network(name, seg_nodes, seg_edges)
            revviz.set_radius(0.02)
            revviz.set_color((1., 1., 0.))

            # Visualize center
            name = "RevoluteCenterPC"
            c_pc = ps.register_point_cloud(name, center.reshape(1, 3))
            c_pc.set_radius(0.05)
            c_pc.set_enabled(True)

        elif joint_model.name == "rigid":
            # Extract parameters
            pos = np.array([joint_model.offset.x, joint_model.offset.y, joint_model.offset.z])

            # Visualize position
            name = "RigidPositionPC"
            r_pc = ps.register_point_cloud(name, pos.reshape(1, 3))
            r_pc.set_radius(0.05)
            r_pc.set_color((0.5, 0.5, 0.5))
            r_pc.set_enabled(True)

    def visualize_ground_truth(self):
        """Visualize ground truth joint information"""
        # Remove existing ground truth visualization
        self.remove_ground_truth_visualization()

        # Loop through all visible datasets
        for dataset_key, dataset in self.datasets.items():
            if not dataset["visible"] or dataset["ground_truth_key"] is None:
                continue

            gt_key = dataset["ground_truth_key"]
            object_type = gt_key["object"]
            part_info = gt_key["part"]

            if object_type not in self.ground_truth_data:
                continue

            # If part info is provided, use it, otherwise check all parts
            if part_info and part_info in self.ground_truth_data[object_type]:
                self.visualize_ground_truth_joint(object_type, part_info, dataset["display_name"])
            else:
                # Just try all parts for this object
                for part_name in self.ground_truth_data[object_type]:
                    self.visualize_ground_truth_joint(object_type, part_name, f"{dataset['display_name']}_{part_name}")

    def visualize_ground_truth_joint(self, object_type, part_name, display_name):
        """Visualize a specific ground truth joint"""
        joint_info = self.ground_truth_data[object_type][part_name]

        # Extract axis and pivot
        if "axis" not in joint_info or "pivot" not in joint_info:
            return

        axis = np.array(joint_info["axis"])

        # Handle different pivot formats (some are single values, some are arrays)
        pivot = joint_info["pivot"]
        if isinstance(pivot, list):
            if len(pivot) == 1:
                # Handle single value pivot
                # Use a default position along the axis
                pivot_point = np.array([0., 0., 0.]) + float(pivot[0]) * axis
            elif len(pivot) == 3:
                # Full 3D pivot point
                pivot_point = np.array(pivot)
            else:
                return
        else:
            # Numeric pivot value
            pivot_point = np.array([0., 0., 0.]) + float(pivot) * axis

        # Normalize axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis = axis / axis_norm

        # Scale axis for visualization
        axis = axis * self.ground_truth_scale

        # Add axis visualization
        seg_nodes = np.array([pivot_point - axis, pivot_point + axis])
        seg_edges = np.array([[0, 1]])

        name = f"GT_{display_name}_Axis"
        axis_viz = ps.register_curve_network(name, seg_nodes, seg_edges)
        axis_viz.set_radius(0.015)
        axis_viz.set_color((0.0, 0.8, 0.2))  # Green for ground truth
        self.gt_curve_networks.append(name)

        # Add pivot point visualization
        name = f"GT_{display_name}_Pivot"
        pivot_viz = ps.register_curve_network(
            name,
            np.array([pivot_point, pivot_point + 0.01 * axis]),
            np.array([[0, 1]])
        )
        pivot_viz.set_radius(0.02)
        pivot_viz.set_color((0.2, 0.8, 0.2))  # Green for ground truth
        self.gt_curve_networks.append(name)

    def remove_ground_truth_visualization(self):
        """Remove all ground truth visualization elements"""
        # Remove tracked curve networks
        for name in self.gt_curve_networks:
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)

        # Remove tracked point clouds
        for name in self.gt_point_clouds:
            if ps.has_point_cloud(name):
                ps.remove_point_cloud(name)

        # Clear the tracking lists
        self.gt_curve_networks = []
        self.gt_point_clouds = []

    def find_neighbors(self, points, num_neighbors):
        """Find neighbors for each point"""
        # Calculate distance matrix
        N = points.shape[0]
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            dist_matrix[i] = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))

        # Find closest points (excluding self)
        neighbors = np.zeros((N, num_neighbors), dtype=int)
        for i in range(N):
            indices = np.argsort(dist_matrix[i])[1:num_neighbors + 1]
            neighbors[i] = indices

        return neighbors

    def compute_rotation_matrix(self, src_points, dst_points):
        """Compute rotation matrix between two point sets using SVD"""
        # Center points
        src_center = np.mean(src_points, axis=0)
        dst_center = np.mean(dst_points, axis=0)

        src_centered = src_points - src_center
        dst_centered = dst_points - dst_center

        # Compute covariance matrix
        H = np.dot(src_centered.T, dst_centered)

        # SVD decomposition
        U, _, Vt = np.linalg.svd(H)

        # Construct rotation matrix
        R = np.dot(Vt.T, U.T)

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        return R, src_center, dst_center

    def rotation_matrix_to_angular_velocity(self, R, dt):
        """Extract angular velocity from rotation matrix"""
        # Ensure R is a valid rotation matrix
        U, _, Vt = np.linalg.svd(R)
        R = np.dot(U, Vt)

        # Compute rotation angle
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # If angle is too small, return zero vector
        if abs(theta) < 1e-6:
            return np.zeros(3)

        # Calculate rotation axis
        sin_theta = np.sin(theta)
        if abs(sin_theta) < 1e-6:
            return np.zeros(3)

        # Extract angular velocity vector from skew-symmetric part
        W = (R - R.T) / (2 * sin_theta)
        omega = np.zeros(3)
        omega[0] = W[2, 1]
        omega[1] = W[0, 2]
        omega[2] = W[1, 0]

        # Angular velocity = axis * angle / time
        omega = omega * theta / (dt + 0.0000000000000000001)

        return omega

    def calculate_angular_velocity(self, d_filter, N):
        """Compute angular velocity using SVD on filtered point cloud data"""
        T = d_filter.shape[0]

        # Initialize arrays
        angular_velocity = np.zeros((T - 1, N, 3))

        # For each pair of consecutive frames
        for t in range(T - 1):
            # Current and next frame points
            current_points = d_filter[t]
            next_points = d_filter[t + 1]

            # Find neighbors for all points
            neighbors = self.find_neighbors(current_points, self.num_neighbors)

            # Compute angular velocity for each point
            for i in range(N):
                # Get current point and its neighbors
                src_neighborhood = current_points[neighbors[i]]
                dst_neighborhood = next_points[neighbors[i]]

                # Compute rotation matrix
                R, _, _ = self.compute_rotation_matrix(src_neighborhood, dst_neighborhood)

                # Extract angular velocity
                omega = self.rotation_matrix_to_angular_velocity(R, self.dt_mean)

                # Store result
                angular_velocity[t, i] = omega

        # Filter angular velocity
        angular_velocity_filtered = np.zeros_like(angular_velocity)

        if T - 1 >= 5:
            for i in range(N):
                for dim in range(3):
                    angular_velocity_filtered[:, i, dim] = savgol_filter(
                        angular_velocity[:, i, dim],
                        window_length=11,  # Ensure window length doesn't exceed data length
                        polyorder=2  # Ensure polynomial order is appropriate
                    )
        else:
            angular_velocity_filtered = angular_velocity.copy()

        return angular_velocity, angular_velocity_filtered

    def switch_dataset(self, new_dataset_key):
        """Switch current active dataset"""
        if new_dataset_key in self.datasets:
            self.current_dataset_key = new_dataset_key
            dataset = self.datasets[new_dataset_key]

            # Update current data and time/point limits
            self.d = dataset["data"]
            self.d_filter = dataset["data_filter"]
            self.dv_filter = dataset["dv_filter"]
            self.angular_velocity_raw = dataset["angular_velocity_raw"]
            self.angular_velocity_filtered = dataset["angular_velocity_filtered"]

            # Update joint information
            self.current_joint_model = dataset["joint_model"]
            self.current_best_joint_type = dataset["best_joint_type"]

            # Update T and N values
            self.T = dataset["T"]
            self.N = dataset["N"]

            # Ensure current t and idx_point are within valid range
            self.t = min(self.t, self.T - 1)
            self.idx_point = min(self.idx_point, self.N - 1)

            # Update plot
            self.plot_image()

            # Update joint visualization
            self.visualize_joint_parameters()

            # Update ground truth if showing
            if self.show_ground_truth:
                self.visualize_ground_truth()

            return True
        return False

    def toggle_visibility(self, dataset_key):
        """Toggle dataset visibility"""
        if dataset_key in self.datasets:
            self.datasets[dataset_key]["visible"] = not self.datasets[dataset_key]["visible"]
            return True
        return False

    def plot_image(self):
        """Plot trajectory of current point"""
        dataset = self.datasets[self.current_dataset_key]

        # Create time axis array (seconds)
        t = np.arange(self.T) * self.dt_mean
        t_angular = np.arange(self.T - 1) * self.dt_mean

        # Create figure with five subplots
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), dpi=100)

        # First subplot: position data
        # Plot original data
        ax1.plot(t, self.d[:, self.idx_point, 0], "--", c="red", label="Original X")
        ax1.plot(t, self.d[:, self.idx_point, 1], "--", c="green", label="Original Y")
        ax1.plot(t, self.d[:, self.idx_point, 2], "--", c="blue", label="Original Z")

        # Plot filtered data
        ax1.plot(t, self.d_filter[:, self.idx_point, 0], "-", c="darkred", label="Filtered X")
        ax1.plot(t, self.d_filter[:, self.idx_point, 1], "-", c="darkgreen", label="Filtered Y")
        ax1.plot(t, self.d_filter[:, self.idx_point, 2], "-", c="darkblue", label="Filtered Z")

        # Set first subplot title and labels
        ax1.set_title(f"Position Trajectory of Point #{self.idx_point} - {dataset['display_name']}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.set_xlim(0, max(t))
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        # Second subplot: velocity data
        ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity")

        # Set second subplot title and labels
        ax2.set_title(f"Linear Velocity of Point #{self.idx_point} - {dataset['display_name']}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlim(0, max(t))
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        # Third subplot: Angular velocity X component
        ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red", label="Raw ωx")
        ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred", label="Filtered ωx")

        ax3.set_title(f"Angular Velocity X Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angular Velocity (rad/s)")
        ax3.set_xlim(0, max(t))
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right')

        # Fourth subplot: Angular velocity Y component
        ax4.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green", label="Raw ωy")
        ax4.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
                 label="Filtered ωy")

        ax4.set_title(f"Angular Velocity Y Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angular Velocity (rad/s)")
        ax4.set_xlim(0, max(t))
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')

        # Fifth subplot: Angular velocity Z component
        ax5.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 2], "--", c="blue", label="Raw ωz")
        ax5.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 2], "-", c="darkblue",
                 label="Filtered ωz")

        ax5.set_title(f"Angular Velocity Z Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Angular Velocity (rad/s)")
        ax5.set_xlim(0, max(t))
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend(loc='upper right')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Create binary buffer in memory
        buf = BytesIO()
        # Save plot as PNG format to buffer
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        # Load image from buffer and convert to RGB format
        image = Image.open(buf).convert('RGB')
        # Convert image to NumPy array
        rgb_array = np.array(image)
        # Add image to Polyscope interface and enable display
        ps.add_color_image_quantity("plot", rgb_array / 255.0, enabled=True)
        # Close figure to release resources
        plt.close(fig)

    def save_current_plot(self):
        """Save current point's trajectory plot to file"""
        dataset = self.datasets[self.current_dataset_key]

        # Create time axis array (seconds)
        t = np.arange(self.T) * self.dt_mean
        t_angular = np.arange(self.T - 1) * self.dt_mean

        # Create figure with five subplots
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), dpi=100)

        # Plot position data
        ax1.plot(t, self.d[:, self.idx_point, 0], "--", c="red", label="Original X")
        ax1.plot(t, self.d[:, self.idx_point, 1], "--", c="green", label="Original Y")
        ax1.plot(t, self.d[:, self.idx_point, 2], "--", c="blue", label="Original Z")
        ax1.plot(t, self.d_filter[:, self.idx_point, 0], "-", c="darkred", label="Filtered X")
        ax1.plot(t, self.d_filter[:, self.idx_point, 1], "-", c="darkgreen", label="Filtered Y")
        ax1.plot(t, self.d_filter[:, self.idx_point, 2], "-", c="darkblue", label="Filtered Z")
        ax1.set_title(f"Position Trajectory of Point #{self.idx_point} - {dataset['display_name']}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.set_xlim(0, max(t))
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        # Plot velocity data
        ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity")
        ax2.set_title(f"Linear Velocity of Point #{self.idx_point} - {dataset['display_name']}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlim(0, max(t))
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        # Plot angular velocity X component
        ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red", label="Raw ωx")
        ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred", label="Filtered ωx")
        ax3.set_title(f"Angular Velocity X Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angular Velocity (rad/s)")
        ax3.set_xlim(0, max(t))
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right')

        # Plot angular velocity Y component
        ax4.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green", label="Raw ωy")
        ax4.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
                 label="Filtered ωy")
        ax4.set_title(f"Angular Velocity Y Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angular Velocity (rad/s)")
        ax4.set_xlim(0, max(t))
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')

        # Plot angular velocity Z component
        ax5.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 2], "--", c="blue", label="Raw ωz")
        ax5.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 2], "-", c="darkblue",
                 label="Filtered ωz")
        ax5.set_title(f"Angular Velocity Z Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Angular Velocity (rad/s)")
        ax5.set_xlim(0, max(t))
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend(loc='upper right')

        plt.tight_layout()

        # Create filename including timestamp and point index
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{dataset['display_name']}_point_{self.idx_point}_t{self.t}_{timestamp}.png"

        # Save to file
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Plot saved to: {filename}")
        return filename

    def callback(self):
        """Polyscope UI callback function"""
        # Display title text
        psim.Text("Point Cloud Joint Analysis")

        # Display current dataset info
        psim.Text(f"Active Dataset: {self.datasets[self.current_dataset_key]['display_name']}")

        # Display joint type information
        psim.Text(f"Detected Joint Type: {self.current_best_joint_type}")

        # Ground truth toggle
        changed_gt, self.show_ground_truth = psim.Checkbox("Show Ground Truth", self.show_ground_truth)
        if changed_gt:
            if self.show_ground_truth:
                self.visualize_ground_truth()
            else:
                self.remove_ground_truth_visualization()

        # Ground truth scale slider
        changed_gt_scale, self.ground_truth_scale = psim.SliderFloat("Ground Truth Scale", self.ground_truth_scale, 0.1,
                                                                     2.0)
        if changed_gt_scale and self.show_ground_truth:
            self.remove_ground_truth_visualization()
            self.visualize_ground_truth()

        psim.Separator()

        # Create time frame slider, returns change status and new value
        self.t_changed, self.t = psim.SliderInt("Time Frame", self.t, 0, self.T - 1)
        # Create point index slider, returns change status and new value
        self.idx_point_changed, self.idx_point = psim.SliderInt("Point Index", self.idx_point, 0, self.N - 1)
        # Number of neighbors slider for SVD
        changed_neighbors, self.num_neighbors = psim.SliderInt("Neighbors for SVD", self.num_neighbors, 5, 30)

        # If time frame changes, update point cloud display
        if self.t_changed:
            for dataset_key, dataset in self.datasets.items():
                if dataset["visible"] and self.t < dataset["T"]:
                    register_point_cloud(
                        dataset["display_name"],
                        dataset["data_filter"][min(self.t, dataset["T"] - 1)],
                        radius=0.01,
                        enabled=True
                    )

        # If point index changes, redraw trajectory plot
        if self.idx_point_changed:
            self.plot_image()

        # Dataset selection section
        if psim.TreeNode("Dataset Selection"):
            # Display dataset list, allow selection and toggle visibility
            for dataset_key, dataset in self.datasets.items():
                if psim.Button(f"Select {dataset['display_name']}"):
                    self.switch_dataset(dataset_key)

                psim.SameLine()
                vis_text = "Visible" if dataset["visible"] else "Hidden"
                if psim.Button(f"{vis_text}###{dataset_key}"):
                    self.toggle_visibility(dataset_key)
                    # Update point cloud display
                    if dataset["visible"]:
                        register_point_cloud(
                            dataset["display_name"],
                            dataset["data_filter"][min(self.t, dataset["T"] - 1)],
                            radius=0.01,
                            enabled=True
                        )
                    else:
                        ps.remove_point_cloud(dataset["display_name"])

            psim.TreePop()

        # Add save image button
        if psim.Button("Save Current Plot"):
            saved_path = self.save_current_plot()
            psim.Text(f"Saved to: {saved_path}")

        # Add recalculate button
        if changed_neighbors or psim.Button("Reanalyze Joint"):
            # Recalculate angular velocity
            for dataset_key, dataset in self.datasets.items():
                angular_velocity_raw, angular_velocity_filtered = self.calculate_angular_velocity(
                    dataset["data_filter"], dataset["N"]
                )
                dataset["angular_velocity_raw"] = angular_velocity_raw
                dataset["angular_velocity_filtered"] = angular_velocity_filtered

                # Perform joint analysis
                joint_model, best_joint_type = self.perform_joint_analysis(dataset["data_filter"])
                dataset["joint_model"] = joint_model
                dataset["best_joint_type"] = best_joint_type

            # Update current dataset's values
            self.angular_velocity_raw = self.datasets[self.current_dataset_key]["angular_velocity_raw"]
            self.angular_velocity_filtered = self.datasets[self.current_dataset_key]["angular_velocity_filtered"]
            self.current_joint_model = self.datasets[self.current_dataset_key]["joint_model"]
            self.current_best_joint_type = self.datasets[self.current_dataset_key]["best_joint_type"]

            # Update plot and visualization
            self.plot_image()
            self.visualize_joint_parameters()

            psim.Text("Joint analysis recalculated")

        # # Display ground truth information for current dataset
        # if psim.TreeNode("Ground Truth Information"):
        #     dataset = self.datasets[self.current_dataset_key]
        #     gt_key = dataset["ground_truth_key"]
        #
        #     if gt_key:
        #         object_type = gt_key["object"]
        #         part_info = gt_key["part"]
        #
        #         psim.Text(f"Object Type: {object_type}")
        #         if part_info:
        #             psim.Text(f"Part: {part_info}")
        #
        #             if object_type in self.ground_truth_data and part_info in self.ground_truth_data[object_type]:
        #                 joint_info = self.ground_truth_data[object_type][part_info]
        #
        #                 if "axis" in joint_info:
        #                     axis = joint_info["axis"]
        #                     psim.Text(f"Ground Truth Axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")
        #
        #                     # Calculate normalized version
        #                     axis_norm = np.linalg.norm(axis)
        #                     if axis_norm > 1e-6:
        #                         norm_axis = [ax / axis_norm for ax in axis]
        #                         psim.Text(f"Normalized: [{norm_axis[0]:.4f}, {norm_axis[1]:.4f}, {norm_axis[2]:.4f}]")
        #
        #                 if "pivot" in joint_info:
        #                     pivot = joint_info["pivot"]
        #                     if isinstance(pivot, list):
        #                         if len(pivot) == 1:
        #                             psim.Text(f"Ground Truth Pivot (parameter): {pivot[0]:.4f}")
        #                         elif len(pivot) == 3:
        #                             psim.Text(f"Ground Truth Pivot: [{pivot[0]:.4f}, {pivot[1]:.4f}, {pivot[2]:.4f}]")
        #                     else:
        #                         psim.Text(f"Ground Truth Pivot (parameter): {pivot:.4f}")
        #
        #         # Compare with current joint estimate
        #         if self.current_joint_model is not None and gt_key:
        #             psim.Text("\nComparison with Estimated Joint:")
        #             psim.Text(f"Estimated Type: {self.current_joint_model.name}")
        #
        #             # Show model-specific comparison
        #             if self.current_joint_model.name == "prismatic" and object_type in self.ground_truth_data:
        #                 if part_info and part_info in self.ground_truth_data[object_type]:
        #                     gt_data = self.ground_truth_data[object_type][part_info]
        #                     if "axis" in gt_data:
        #                         gt_axis = np.array(gt_data["axis"])
        #                         gt_axis_norm = np.linalg.norm(gt_axis)
        #                         if gt_axis_norm > 1e-6:
        #                             gt_axis = gt_axis / gt_axis_norm
        #
        #                         est_axis = self.current_joint_model.prismatic_dir
        #                         est_axis_norm = np.linalg.norm(est_axis)
        #                         if est_axis_norm > 1e-6:
        #                             est_axis = est_axis / est_axis_norm
        #
        #                         # Calculate angle between axes
        #                         dot_product = np.clip(np.abs(np.dot(gt_axis, est_axis)), 0.0, 1.0)
        #                         angle_diff = np.arccos(dot_product)
        #                         angle_diff_deg = np.degrees(angle_diff)
        #
        #                         psim.Text(f"Axis Angle Difference: {angle_diff_deg:.2f}°")
        #
        #             elif self.current_joint_model.name == "revolute" and object_type in self.ground_truth_data:
        #                 if part_info and part_info in self.ground_truth_data[object_type]:
        #                     gt_data = self.ground_truth_data[object_type][part_info]
        #                     if "axis" in gt_data:
        #                         gt_axis = np.array(gt_data["axis"])
        #                         gt_axis_norm = np.linalg.norm(gt_axis)
        #                         if gt_axis_norm > 1e-6:
        #                             gt_axis = gt_axis / gt_axis_norm
        #
        #                         # Get estimated axis from quaternion
        #                         axis_q = self.current_joint_model.rot_axis_q.numpy() if hasattr(
        #                             self.current_joint_model.rot_axis_q,
        #                             'numpy') else self.current_joint_model.rot_axis_q
        #                         R = quaternion_to_matrix(axis_q[0], axis_q[1], axis_q[2], axis_q[3])
        #                         est_axis = R.dot(np.array([0, 0, 1]))
        #                         est_axis_norm = np.linalg.norm(est_axis)
        #                         if est_axis_norm > 1e-6:
        #                             est_axis = est_axis / est_axis_norm
        #
        #                         # Calculate angle between axes
        #                         dot_product = np.clip(np.abs(np.dot(gt_axis, est_axis)), 0.0, 1.0)
        #                         angle_diff = np.arccos(dot_product)
        #                         angle_diff_deg = np.degrees(angle_diff)
        #
        #                         psim.Text(f"Axis Angle Difference: {angle_diff_deg:.2f}°")
        #     else:
        #         psim.Text("No ground truth data found for this dataset")
        #
        #     psim.TreePop()
        # Display ground truth information for current dataset
        if psim.TreeNode("Ground Truth Information"):
            dataset = self.datasets[self.current_dataset_key]
            gt_key = dataset["ground_truth_key"]

            if gt_key:
                object_type = gt_key["object"]
                part_info = gt_key["part"]

                psim.Text(f"Object Type: {object_type}")
                if part_info:
                    psim.Text(f"Part: {part_info}")

                    if object_type in self.ground_truth_data and part_info in self.ground_truth_data[object_type]:
                        joint_info = self.ground_truth_data[object_type][part_info]

                        if "axis" in joint_info:
                            axis = joint_info["axis"]
                            psim.Text(f"Ground Truth Axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")

                            # Calculate normalized version
                            axis_norm = np.linalg.norm(axis)
                            if axis_norm > 1e-6:
                                norm_axis = [ax / axis_norm for ax in axis]
                                psim.Text(f"Normalized: [{norm_axis[0]:.4f}, {norm_axis[1]:.4f}, {norm_axis[2]:.4f}]")

                        if "pivot" in joint_info:
                            pivot = joint_info["pivot"]
                            if isinstance(pivot, list):
                                if len(pivot) == 1:
                                    psim.Text(f"Ground Truth Pivot (parameter): {pivot[0]:.4f}")
                                elif len(pivot) == 3:
                                    psim.Text(f"Ground Truth Pivot: [{pivot[0]:.4f}, {pivot[1]:.4f}, {pivot[2]:.4f}]")
                            else:
                                psim.Text(f"Ground Truth Pivot (parameter): {pivot:.4f}")

                # Compare with current joint estimate
                if self.current_joint_model is not None and gt_key:
                    psim.Text("\nComparison with Estimated Joint:")
                    psim.Text(f"Estimated Type: {self.current_joint_model.name}")

                    # Show model-specific comparison
                    if self.current_joint_model.name == "prismatic" and object_type in self.ground_truth_data:
                        if part_info and part_info in self.ground_truth_data[object_type]:
                            gt_data = self.ground_truth_data[object_type][part_info]
                            if "axis" in gt_data:
                                gt_axis = np.array(gt_data["axis"])
                                gt_axis_norm = np.linalg.norm(gt_axis)
                                if gt_axis_norm > 1e-6:
                                    gt_axis = gt_axis / gt_axis_norm

                                est_axis = self.current_joint_model.prismatic_dir
                                est_axis_norm = np.linalg.norm(est_axis)
                                if est_axis_norm > 1e-6:
                                    est_axis = est_axis / est_axis_norm

                                # Calculate angle between axes
                                dot_product = np.clip(np.abs(np.dot(gt_axis, est_axis)), 0.0, 1.0)
                                angle_diff = np.arccos(dot_product)
                                angle_diff_deg = np.degrees(angle_diff)

                                psim.Text(f"Axis Angle Difference: {angle_diff_deg:.2f}°")

                                # Calculate distance between axes
                                # Get points on axes
                                est_origin = self.current_joint_model.rigid_position

                                # Get ground truth pivot
                                gt_pivot = gt_data.get("pivot", [0, 0, 0])
                                if isinstance(gt_pivot, list):
                                    if len(gt_pivot) == 1:
                                        gt_origin = np.array([0., 0., 0.]) + float(gt_pivot[0]) * gt_axis
                                    elif len(gt_pivot) == 3:
                                        gt_origin = np.array(gt_pivot)
                                    else:
                                        gt_origin = np.array([0., 0., 0.])
                                else:
                                    gt_origin = np.array([0., 0., 0.]) + float(gt_pivot) * gt_axis

                                # Calculate distance between two lines
                                cross_product = np.cross(est_axis, gt_axis)
                                cross_norm = np.linalg.norm(cross_product)

                                if cross_norm < 1e-6:
                                    # Axes are parallel
                                    vec_to_point = est_origin - gt_origin
                                    proj_on_axis = np.dot(vec_to_point, gt_axis) * gt_axis
                                    perpendicular = vec_to_point - proj_on_axis
                                    axis_distance = np.linalg.norm(perpendicular)
                                    psim.Text(f"Axis Distance (parallel): {axis_distance:.4f} m")
                                else:
                                    # Axes are skew
                                    vec_between = gt_origin - est_origin
                                    axis_distance = abs(np.dot(vec_between, cross_product)) / cross_norm
                                    psim.Text(f"Axis Distance (skew): {axis_distance:.4f} m")

                    elif self.current_joint_model.name == "revolute" and object_type in self.ground_truth_data:
                        if part_info and part_info in self.ground_truth_data[object_type]:
                            gt_data = self.ground_truth_data[object_type][part_info]
                            if "axis" in gt_data:
                                gt_axis = np.array(gt_data["axis"])
                                gt_axis_norm = np.linalg.norm(gt_axis)
                                if gt_axis_norm > 1e-6:
                                    gt_axis = gt_axis / gt_axis_norm

                                # Get estimated axis from quaternion
                                axis_q = self.current_joint_model.rot_axis_q.numpy() if hasattr(
                                    self.current_joint_model.rot_axis_q,
                                    'numpy') else self.current_joint_model.rot_axis_q
                                R = quaternion_to_matrix(axis_q[0], axis_q[1], axis_q[2], axis_q[3])
                                est_axis = R.dot(np.array([0, 0, 1]))
                                est_axis_norm = np.linalg.norm(est_axis)
                                if est_axis_norm > 1e-6:
                                    est_axis = est_axis / est_axis_norm

                                # Calculate angle between axes
                                dot_product = np.clip(np.abs(np.dot(gt_axis, est_axis)), 0.0, 1.0)
                                angle_diff = np.arccos(dot_product)
                                angle_diff_deg = np.degrees(angle_diff)

                                psim.Text(f"Axis Angle Difference: {angle_diff_deg:.2f}°")

                                # Calculate distance between axes
                                # Get origins/pivots for both axes
                                est_origin = self.current_joint_model.rot_center.numpy() if hasattr(
                                    self.current_joint_model.rot_center,
                                    'numpy') else self.current_joint_model.rot_center

                                # Get ground truth pivot
                                gt_pivot = gt_data.get("pivot", [0, 0, 0])
                                if isinstance(gt_pivot, list):
                                    if len(gt_pivot) == 1:
                                        # Single value pivot - assume it's along the axis from origin
                                        gt_origin = np.array([0., 0., 0.]) + float(gt_pivot[0]) * gt_axis
                                    elif len(gt_pivot) == 3:
                                        gt_origin = np.array(gt_pivot)
                                    else:
                                        gt_origin = np.array([0., 0., 0.])
                                else:
                                    # Numeric pivot value
                                    gt_origin = np.array([0., 0., 0.]) + float(gt_pivot) * gt_axis

                                # Calculate distance between two lines (axes)
                                # For skew lines, the distance is ||(p2-p1) · (d1×d2)|| / ||d1×d2||
                                # For parallel lines, use point-to-line distance

                                cross_product = np.cross(est_axis, gt_axis)
                                cross_norm = np.linalg.norm(cross_product)

                                if cross_norm < 1e-6:
                                    # Axes are parallel, calculate point-to-line distance
                                    # Distance from est_origin to gt_axis line
                                    vec_to_point = est_origin - gt_origin
                                    proj_on_axis = np.dot(vec_to_point, gt_axis) * gt_axis
                                    perpendicular = vec_to_point - proj_on_axis
                                    axis_distance = np.linalg.norm(perpendicular)
                                    psim.Text(f"Axis Distance (parallel): {axis_distance:.4f} m")
                                    psim.Text(f"Axis Distance (parallel): {axis_distance * 1000:.2f} mm")
                                else:
                                    # Axes are skew, calculate minimum distance between lines
                                    vec_between = gt_origin - est_origin
                                    axis_distance = abs(np.dot(vec_between, cross_product)) / cross_norm
                                    psim.Text(f"Axis Distance (skew): {axis_distance:.4f} m")
                                    psim.Text(f"Axis Distance (skew): {axis_distance * 1000:.2f} mm")

                                # Also calculate the distance between origin points
                                origin_distance = np.linalg.norm(est_origin - gt_origin)
                                psim.Text(f"Origin Distance: {origin_distance:.4f} m ({origin_distance * 1000:.2f} mm)")

                                # Calculate the closest points on each axis (for skew lines)
                                if cross_norm > 1e-6:
                                    # For skew lines, find closest points
                                    # Direction perpendicular to both axes
                                    n = cross_product / cross_norm

                                    # Find parameter t for closest point on gt axis
                                    # and parameter s for closest point on est axis
                                    # Using the formula for closest points between skew lines

                                    # Build system of equations
                                    # est_origin + s * est_axis = closest point on est axis
                                    # gt_origin + t * gt_axis = closest point on gt axis
                                    # The vector between closest points is parallel to n

                                    # Solve for t
                                    d = est_origin - gt_origin
                                    a1 = np.dot(est_axis, est_axis)
                                    b1 = np.dot(est_axis, gt_axis)
                                    c1 = np.dot(est_axis, d)
                                    a2 = np.dot(gt_axis, est_axis)
                                    b2 = np.dot(gt_axis, gt_axis)
                                    c2 = np.dot(gt_axis, d)

                                    denom = a1 * b2 - b1 * a2
                                    if abs(denom) > 1e-6:
                                        s = (b1 * c2 - b2 * c1) / denom
                                        t = (a2 * c1 - a1 * c2) / denom

                                        closest_on_est = est_origin + s * est_axis
                                        closest_on_gt = gt_origin + t * gt_axis

                                        psim.Text(f"\nClosest points between axes:")
                                        psim.Text(
                                            f"On estimated axis: [{closest_on_est[0]:.3f}, {closest_on_est[1]:.3f}, {closest_on_est[2]:.3f}]")
                                        psim.Text(
                                            f"On GT axis: [{closest_on_gt[0]:.3f}, {closest_on_gt[1]:.3f}, {closest_on_gt[2]:.3f}]")

                                        # Verify the distance
                                        closest_dist = np.linalg.norm(closest_on_est - closest_on_gt)
                                        psim.Text(
                                            f"Distance between closest points: {closest_dist:.4f} m ({closest_dist * 1000:.2f} mm)")
            else:
                psim.Text("No ground truth data found for this dataset")

            psim.TreePop()

        if psim.TreeNode("Coordinate System Settings"):
            options = ["y_up", "z_up"]
            for i, option in enumerate(options):
                is_selected = (ps.get_up_dir() == option)
                changed, is_selected = psim.Checkbox(f"{option}", is_selected)
                if changed and is_selected:
                    ps.set_up_dir(option)

            # 添加调整坐标系的滑块
            changed_scale, self.coord_scale = psim.SliderFloat("Coord Frame Scale", self.coord_scale, 0.05, 1.0)
            if changed_scale:
                # 重新绘制原点坐标系
                draw_frame_3d(np.zeros(6), label="origin", scale=self.coord_scale)

            # 地面平面选项
            ground_modes = ["none", "tile", "shadow_only"]
            current_mode = ps.get_ground_plane_mode()
            for mode in ground_modes:
                is_selected = (current_mode == mode)
                changed, is_selected = psim.Checkbox(f"Ground: {mode}", is_selected)
                if changed and is_selected:
                    ps.set_ground_plane_mode(mode)

            psim.TreePop()

        # Display current data and joint information
        if psim.TreeNode("Joint Information"):
            if self.current_joint_model is not None:
                psim.Text(f"Model Type: {self.current_joint_model.name}")
                psim.Text(f"BIC Score: {self.current_joint_model.bic:.6f}")
                psim.Text(f"Log Likelihood: {self.current_joint_model.loglikelihood:.6f}")
                psim.Text(f"Position Error: {self.current_joint_model.avg_pos_err:.6f}")
                psim.Text(f"Orientation Error: {self.current_joint_model.avg_ori_err:.6f}")

                if self.current_joint_model.name == "prismatic":
                    psim.Text(
                        f"Position: [{self.current_joint_model.rigid_position[0]:.3f}, {self.current_joint_model.rigid_position[1]:.3f}, {self.current_joint_model.rigid_position[2]:.3f}]")
                    psim.Text(
                        f"Direction: [{self.current_joint_model.prismatic_dir[0]:.3f}, {self.current_joint_model.prismatic_dir[1]:.3f}, {self.current_joint_model.prismatic_dir[2]:.3f}]")

                elif self.current_joint_model.name == "revolute":
                    center = self.current_joint_model.rot_center.numpy() if hasattr(self.current_joint_model.rot_center,
                                                                                    'numpy') else self.current_joint_model.rot_center
                    axis_q = self.current_joint_model.rot_axis_q.numpy() if hasattr(self.current_joint_model.rot_axis_q,
                                                                                    'numpy') else self.current_joint_model.rot_axis_q
                    psim.Text(f"Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
                    psim.Text(f"Axis Quat: [{axis_q[0]:.3f}, {axis_q[1]:.3f}, {axis_q[2]:.3f}, {axis_q[3]:.3f}]")
                    psim.Text(f"Radius: {self.current_joint_model.rot_radius:.3f}")

                elif self.current_joint_model.name == "rigid":
                    psim.Text(
                        f"Position: [{self.current_joint_model.offset.x:.3f}, {self.current_joint_model.offset.y:.3f}, {self.current_joint_model.offset.z:.3f}]")
                    psim.Text(
                        f"Orientation: [{self.current_joint_model.offset.qx:.3f}, {self.current_joint_model.offset.qy:.3f}, {self.current_joint_model.offset.qz:.3f}, {self.current_joint_model.offset.qw:.3f}]")
            else:
                psim.Text("No joint model available")

            psim.TreePop()


# Program entry point
if __name__ == "__main__":
    # You can specify multiple data file paths here
    file_paths = [
        # open refrigerator 1
        "./parahome_data_thesis/s1_refrigerator_part2_3180_3240.npy",
        "./parahome_data_thesis/s1_refrigerator_base_3180_3240.npy",
        "./parahome_data_thesis/s1_refrigerator_part1_3180_3240.npy"

        # close refrigerator
        # "./demo_data/s1_refrigerator_part2_3360_3420.npy",
        # "./demo_data/s1_refrigerator_base_3360_3420.npy",
        # "./demo_data/s1_refrigerator_part1_3360_3420.npy"

        # open and close drawer (1)
        # "./demo_data/s2_drawer_part1_1770_1950.npy",
        # "./demo_data/s2_drawer_base_1770_1950.npy",
        # "./demo_data/s2_drawer_part2_1770_1950.npy"

        # #open gasstove 1
        # "./demo_data/s1_gasstove_part2_1110_1170.npy",
        # "./demo_data/s1_gasstove_part1_1110_1170.npy",
        # "./demo_data/s1_gasstove_base_1110_1170.npy"

        # close gasstove 1
        # "./demo_data/s1_gasstove_part2_2760_2850.npy",
        # "./demo_data/s1_gasstove_part1_2760_2850.npy",
        # "./demo_data/s1_gasstove_base_2760_2850.npy"

        # open microwave 1
        # "./demo_data/s1_microwave_part1_1380_1470.npy",
        # "./demo_data/s1_microwave_base_1380_1470.npy"

        # close microwave 1
        # "./demo_data/s1_microwave_part1_1740_1830.npy",
        # "./demo_data/s1_microwave_base_1740_1830.npy"

        # open washingmachine  1
        # "./demo_data/s2_washingmachine_part1_1140_1170.npy",
        # "./demo_data/s2_washingmachine_base_1140_1170.npy"

        # close washingmachine
        # "./demo_data/s2_washingmachine_part1_1260_1290.npy",
        # "./demo_data/s2_washingmachine_base_1260_1290.npy"

        # # #chair  1
        # "./demo_data/s3_chair_base_2610_2760.npy"

        # #chair 1
        # "./demo_data/s6_chair_base_90_270.npy"

        # trashbin
        # "./demo_data/s6_trashbin_part1_750_900.npy",
        # "./demo_data/s6_trashbin_base_750_900.npy"

        # #cap
        # "./demo_data/screw.npy"

        # #ball
        # "./demo_data/ball.npy"

        # 1
        # "./demo_data/revolute.npy"
        # "./demo_data/fridge.npy"
    ]

    # Create EnhancedViz instance and execute visualization
    viz = EnhancedViz(file_paths)