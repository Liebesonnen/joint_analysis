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
from scipy.spatial.transform import Rotation
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union


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
#                      Twist & SE(3) Utils from Second Code                   #
###############################################################################

def skew_symmetric(v):
    """Convert 3D vector to skew symmetric matrix"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


class Twist:
    """Represents a spatial twist (linear and angular velocities combined)"""

    def __init__(self, rx=0.0, ry=0.0, rz=0.0, vx=0.0, vy=0.0, vz=0.0):
        self.angular = np.array([rx, ry, rz], dtype=float)
        self.linear = np.array([vx, vy, vz], dtype=float)

    def __add__(self, other):
        """Add two twists"""
        result = Twist()
        result.angular = self.angular + other.angular
        result.linear = self.linear + other.linear
        return result

    def __mul__(self, scalar):
        """Multiply twist by scalar"""
        result = Twist()
        result.angular = self.angular * scalar
        result.linear = self.linear * scalar
        return result

    def norm(self):
        """Compute the norm of the twist"""
        return np.sqrt(np.sum(self.angular ** 2) + np.sum(self.linear ** 2))

    def exp(self, epsilon=1e-12):
        """Exponential map from twist to transformation matrix"""
        v = self.linear
        w = self.angular

        theta = np.linalg.norm(w)
        if theta < epsilon:
            # Pure translation
            transform = np.eye(4)
            transform[:3, 3] = v
            return transform

        # Rotation and translation
        w_hat = skew_symmetric(w / theta)

        # Rodrigues' formula for rotation
        R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * np.dot(w_hat, w_hat)

        # Translation component
        V = np.eye(3) + ((1 - np.cos(theta)) / theta ** 2) * w_hat + \
            ((theta - np.sin(theta)) / theta ** 3) * np.dot(w_hat, w_hat)

        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = np.dot(V, v)

        return transform

    @staticmethod
    def log(transform, epsilon=1e-12):
        """Logarithmic map from transformation matrix to twist"""
        R = transform[:3, :3]
        p = transform[:3, 3]

        # Check if this is a pure translation
        trace = np.trace(R)
        if abs(trace - 3) < epsilon:
            # Pure translation
            return Twist(vx=p[0], vy=p[1], vz=p[2])

        # Extract rotation angle
        theta = np.arccos((trace - 1) / 2)

        # Extract rotation axis
        if abs(theta) < epsilon:
            w = np.zeros(3)
        else:
            w_hat = (R - R.T) / (2 * np.sin(theta))
            w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) * theta

        # Extract translation component
        if abs(theta) < epsilon:
            v = p
        else:
            w_hat = skew_symmetric(w / theta)
            A = np.eye(3) - (theta / 2) * w_hat + (1 - theta / (2 * np.tan(theta / 2))) * np.dot(w_hat, w_hat)
            v = np.dot(np.linalg.inv(A), p)

        return Twist(rx=w[0], ry=w[1], rz=w[2], vx=v[0], vy=v[1], vz=v[2])


###############################################################################
#                      Joint Classification Framework                          #
###############################################################################

class JointType(Enum):
    RIGID = 0
    REVOLUTE = 1
    PRISMATIC = 2
    DISCONNECTED = 6
    UNKNOWN = 7


class JointFilter:
    """Base class for joint filters"""

    def __init__(self):
        # Joint state variables
        self._joint_state = 0.0
        self._joint_velocity = 0.0
        self._joint_position = np.zeros(3)
        self._joint_orientation = np.array([1.0, 0.0, 0.0])

        # Probabilities
        self._measurements_likelihood = 0.0
        self._model_prior_probability = 0.25  # Equal prior probability for all joint types
        self._unnormalized_model_probability = 0.0
        self._normalizing_term = 1.0

        # Measurement history
        self._delta_poses_in_rrbf = []
        self._all_points_history = None

        # Parameters
        self._likelihood_sample_num = 30

    def set_all_points_history(self, all_points_history):
        """Set the point cloud history for analysis"""
        self._all_points_history = all_points_history
        # Calculate deltas for joint estimation
        self._calculate_delta_poses()

    def _calculate_delta_poses(self):
        """Calculate delta poses from point cloud history"""
        if self._all_points_history is None or len(self._all_points_history) < 2:
            return

        # For simplicity, we'll take the average point position as the "pose"
        # and calculate delta poses from consecutive frames
        poses = []
        for frame_points in self._all_points_history:
            center = np.mean(frame_points, axis=0)
            # Create an identity pose with the centroid as translation
            pose = np.eye(4)
            pose[:3, 3] = center
            poses.append(pose)

        # Calculate delta poses (T_i+1 * inv(T_i))
        self._delta_poses_in_rrbf = []
        for i in range(len(poses) - 1):
            T_i = poses[i]
            T_i_plus_1 = poses[i + 1]
            T_i_inv = np.linalg.inv(T_i)
            delta = np.dot(T_i_plus_1, T_i_inv)
            self._delta_poses_in_rrbf.append(Twist.log(delta))

    def estimate_measurement_history_likelihood(self):
        """Estimate the likelihood of the measurement history"""
        pass

    def estimate_unnormalized_model_probability(self):
        """Estimate the unnormalized model probability"""
        self._unnormalized_model_probability = (
                self._model_prior_probability *
                self._measurements_likelihood
        )

    def get_joint_filter_type(self):
        """Get the type of the joint filter"""
        raise NotImplementedError("Subclasses must implement this method")

    def get_joint_filter_type_str(self):
        """Get the type of the joint filter as a string"""
        raise NotImplementedError("Subclasses must implement this method")

    def get_probability_of_joint_filter(self):
        """Get the normalized probability of the joint filter"""
        return self._unnormalized_model_probability / max(self._normalizing_term, 1e-8)

    def set_normalizing_term(self, term):
        """Set the normalizing term for probability calculation"""
        self._normalizing_term = term

    def extract_joint_parameters(self):
        """Extract joint parameters for visualization"""
        return {}

    def set_measurement(self, twist):
        """设置观测值"""
        self._measurement = twist

    def get_probability(self):
        """获取模型概率"""
        return self._measurements_likelihood


class RigidJointFilter(JointFilter):
    """Enhanced filter for rigid joints"""

    def __init__(self):
        super().__init__()
        self._measurements_likelihood = 0
        self._motion_memory_prior = 1
        self._rig_max_translation = 0.005
        self._rig_max_rotation = 0.01
        self._measurement = None

    def estimate_measurement_history_likelihood(self):
        """Estimate the likelihood of the measurement history"""
        if len(self._delta_poses_in_rrbf) == 0:
            self._measurements_likelihood = 1.0
            return

        sigma_translation = 0.05
        sigma_rotation = 0.2
        accumulated_error = 0.0
        frame_counter = 0.0

        trajectory_length = len(self._delta_poses_in_rrbf)
        amount_samples = min(trajectory_length, self._likelihood_sample_num)
        delta_idx_samples = max(1.0, float(trajectory_length) / float(self._likelihood_sample_num))
        current_idx = 0

        p_all_meas_given_model_params = 0.0

        for sample_idx in range(amount_samples):
            current_idx = min(int(round(sample_idx * delta_idx_samples)), trajectory_length - 1)
            rb2_last_delta_relative_twist = self._delta_poses_in_rrbf[current_idx]
            rb2_last_delta_relative_displ = rb2_last_delta_relative_twist.exp()

            rb2_last_delta_relative_translation = rb2_last_delta_relative_displ[:3, 3]
            rb2_last_delta_relative_rotation = Rotation.from_matrix(rb2_last_delta_relative_displ[:3, :3])

            rigid_joint_translation = np.zeros(3)
            rb2_last_delta_relative_displ_rigid_hyp = Twist().exp()

            rb2_last_delta_relative_translation_rigid_hyp = rb2_last_delta_relative_displ_rigid_hyp[:3, 3]
            rb2_last_delta_relative_rotation_rigid_hyp = Rotation.from_matrix(
                rb2_last_delta_relative_displ_rigid_hyp[:3, :3])

            translation_error = np.linalg.norm(
                rb2_last_delta_relative_translation - rb2_last_delta_relative_translation_rigid_hyp
            )

            if translation_error > self._rig_max_translation:
                self._motion_memory_prior = 0.0

            rotation_error = rb2_last_delta_relative_rotation.inv() * rb2_last_delta_relative_rotation_rigid_hyp
            rotation_error_angle = np.linalg.norm(rotation_error.as_rotvec())

            if rotation_error_angle > self._rig_max_rotation:
                self._motion_memory_prior = 0.0

            accumulated_error += translation_error + abs(rotation_error_angle)

            p_one_meas_given_model_params = (
                                                    (1.0 / (sigma_translation * np.sqrt(2.0 * np.pi))) *
                                                    np.exp((-1.0 / 2.0) * pow(translation_error / sigma_translation, 2))
                                            ) * (
                                                    (1.0 / (sigma_rotation * np.sqrt(2.0 * np.pi))) *
                                                    np.exp((-1.0 / 2.0) * pow(rotation_error_angle / sigma_rotation, 2))
                                            )

            p_all_meas_given_model_params += (p_one_meas_given_model_params / float(amount_samples))
            frame_counter += 1.0

        self._measurements_likelihood = p_all_meas_given_model_params

    def estimate_unnormalized_model_probability(self):
        """Estimate the unnormalized model probability, including motion_memory_prior"""
        self._unnormalized_model_probability = (
                self._model_prior_probability *
                self._measurements_likelihood *
                self._motion_memory_prior
        )

    def get_joint_filter_type(self):
        """Get the type of the joint filter"""
        return JointType.RIGID

    def get_joint_filter_type_str(self):
        """Get the type of the joint filter as a string"""
        return "RigidJointFilter"

    def extract_joint_parameters(self):
        """Extract joint parameters for visualization"""
        return {
            "type": JointType.RIGID
        }


class PrismaticJointFilter(JointFilter):
    """Filter for prismatic joints"""

    def __init__(self):
        super().__init__()
        self._joint_params = {
            "axis": np.array([1.0, 0.0, 0.0]),
            "origin": np.array([0.0, 0.0, 0.0]),
            "motion_limit": (0.0, 0.0)
        }
        self._measurement = None
        self._joint_variable = 0.0
        self._joint_velocity = 0.0

    def estimate_measurement_history_likelihood(self):
        """Estimate the likelihood of the measurement history"""
        if len(self._delta_poses_in_rrbf) == 0 or self._all_points_history is None:
            self._measurements_likelihood = 0.1
            return

        T, N, _ = self._all_points_history.shape

        if T < 3:
            self._measurements_likelihood = 0.1
            return

        displacements = []
        for i in range(N):
            start_pos = self._all_points_history[0, i]
            end_pos = self._all_points_history[-1, i]
            displacement = end_pos - start_pos
            displacements.append(displacement)

        displacement_vectors = np.array(displacements)
        avg_displacement = np.mean(displacement_vectors, axis=0)

        displacement_norm = np.linalg.norm(avg_displacement)
        if displacement_norm > 1e-6:
            self._joint_params["axis"] = avg_displacement / displacement_norm

            has_low_rotation = True
            for twist in self._delta_poses_in_rrbf:
                angular_norm = np.linalg.norm(twist.angular)
                linear_norm = np.linalg.norm(twist.linear)
                if angular_norm > 0.1 * linear_norm and angular_norm > 0.01:
                    has_low_rotation = False
                    break

            if has_low_rotation:
                self._joint_params["origin"] = np.mean(self._all_points_history[0], axis=0)
                total_displacement = np.dot(avg_displacement, self._joint_params["axis"])
                self._joint_variable = total_displacement

                if T > 1:
                    frame_time = 1.0
                    self._joint_velocity = total_displacement / ((T - 1) * frame_time)

                if total_displacement >= 0:
                    self._joint_params["motion_limit"] = (0.0, total_displacement)
                else:
                    self._joint_params["motion_limit"] = (total_displacement, 0.0)

                self._measurements_likelihood = min(1.0, abs(total_displacement) / 2.0)
            else:
                self._measurements_likelihood = 0.1
        else:
            self._measurements_likelihood = 0.1

    def get_joint_filter_type(self):
        """Get the type of the joint filter"""
        return JointType.PRISMATIC

    def get_joint_filter_type_str(self):
        """Get the type of the joint filter as a string"""
        return "PrismaticJointFilter"

    def extract_joint_parameters(self):
        """Extract joint parameters for visualization"""
        return {
            "type": JointType.PRISMATIC,
            "axis": self._joint_params["axis"],
            "origin": self._joint_params["origin"],
            "motion_limit": self._joint_params["motion_limit"],
            "joint_velocity": self._joint_velocity
        }


class RevoluteJointFilter(JointFilter):
    """Enhanced filter for revolute joints"""

    def __init__(self):
        super().__init__()
        self._accumulated_rotation = 0.0
        self._joint_states_all = []
        self._prev_joint_state = 0.0
        self._joint_velocity = 0.0
        self._joint_params = {
            "axis": np.array([0.0, 1.0, 0.0]),
            "origin": np.array([0.0, 0.0, 0.0]),
            "motion_limit": (0.0, 0.0)
        }
        self._rev_min_rot_for_ee = 0.1
        self._measurement = None
        self._joint_variable = 0.0

    def estimate_measurement_history_likelihood(self):
        """Estimate the likelihood of the measurement history"""
        if len(self._delta_poses_in_rrbf) == 0 or self._all_points_history is None:
            self._measurements_likelihood = 0.1
            return

        T, N, _ = self._all_points_history.shape

        if T < 3:
            self._measurements_likelihood = 0.1
            return

        try:
            # Calculate mean positions for each point
            points_trajectories = np.zeros((N, T, 3))
            for i in range(N):
                for t in range(T):
                    points_trajectories[i, t] = self._all_points_history[t, i]

            mean_positions = np.mean(points_trajectories, axis=1)
            centroid = np.mean(mean_positions, axis=0)

            # Calculate variances in each direction
            variances = np.zeros((N, 3))
            for i in range(N):
                variances[i] = np.var(points_trajectories[i], axis=0)

            total_variance = np.sum(variances, axis=1)
            high_variance_indices = np.argsort(total_variance)[-int(N / 3):]
            high_var_points = points_trajectories[high_variance_indices]

            # Compute PCA for these points to find the rotation axis
            centered_points = high_var_points - centroid
            all_points_flat = centered_points.reshape(-1, 3)

            cov_matrix = np.cov(all_points_flat.T)
            eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

            axis_idx = np.argmin(eig_vals)
            axis = eig_vecs[:, axis_idx]
            axis = axis / np.linalg.norm(axis)
            self._joint_params["axis"] = axis

            # Estimate rotation center
            A = np.eye(3) - np.outer(axis, axis)
            b = np.zeros(3)

            for i in high_variance_indices:
                point_mean = mean_positions[i]
                b += A @ point_mean

            origin = np.linalg.lstsq(A, b, rcond=None)[0]
            self._joint_params["origin"] = origin

            # Calculate joint angles between consecutive frames
            joint_states = []
            for t in range(1, T):
                prev_points = self._all_points_history[t - 1]
                curr_points = self._all_points_history[t]

                prev_centered = prev_points - origin
                curr_centered = curr_points - origin

                proj_matrix = np.eye(3) - np.outer(axis, axis)
                prev_proj = prev_centered @ proj_matrix.T
                curr_proj = curr_centered @ proj_matrix.T

                valid_indices = np.logical_and(
                    np.linalg.norm(prev_proj, axis=1) > 1e-6,
                    np.linalg.norm(curr_proj, axis=1) > 1e-6
                )

                if np.sum(valid_indices) > 0:
                    dot_products = np.sum(prev_proj[valid_indices] * curr_proj[valid_indices], axis=1)
                    norms_prev = np.linalg.norm(prev_proj[valid_indices], axis=1)
                    norms_curr = np.linalg.norm(curr_proj[valid_indices], axis=1)

                    cosines = np.clip(dot_products / (norms_prev * norms_curr), -1.0, 1.0)
                    angles = np.arccos(cosines)

                    cross_products = np.cross(prev_proj[valid_indices], curr_proj[valid_indices])
                    dot_with_axis = np.sum(cross_products * axis, axis=1)
                    signs = np.sign(dot_with_axis)

                    signed_angles = angles * signs
                    avg_angle = np.mean(signed_angles)
                    joint_states.append(avg_angle)

            # Accumulate the rotation
            total_rotation = 0
            if joint_states:
                total_rotation = np.sum(joint_states)

                if len(self._joint_states_all) > 0:
                    last_state = self._joint_states_all[-1]
                    if abs(total_rotation - last_state) > np.pi:
                        if total_rotation * last_state < 0:
                            pass  # Handle discontinuity if needed

                self._joint_states_all.append(total_rotation)

                if len(self._joint_states_all) >= 2:
                    time_step = 1.0
                    self._joint_velocity = (total_rotation - self._prev_joint_state) / time_step

                self._prev_joint_state = total_rotation
                self._joint_state = total_rotation
                self._joint_variable = total_rotation

                min_rotation = min(joint_states) if joint_states else 0
                max_rotation = max(joint_states) if joint_states else 0
                self._joint_params["motion_limit"] = (min_rotation, max_rotation)

                # Compute measurement likelihood
                errors = []
                for t in range(1, T):
                    prev_points = self._all_points_history[t - 1]
                    curr_points = self._all_points_history[t]

                    angle = joint_states[t - 1]
                    # Rotate points around the estimated axis and origin
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 1e-6:
                        axis_unit = axis / axis_norm
                        points = prev_points - origin
                        c, s = np.cos(angle), np.sin(angle)
                        t = 1 - c
                        R = np.array([
                            [t * axis_unit[0] * axis_unit[0] + c,
                             t * axis_unit[0] * axis_unit[1] - s * axis_unit[2],
                             t * axis_unit[0] * axis_unit[2] + s * axis_unit[1]],
                            [t * axis_unit[0] * axis_unit[1] + s * axis_unit[2],
                             t * axis_unit[1] * axis_unit[1] + c,
                             t * axis_unit[1] * axis_unit[2] - s * axis_unit[0]],
                            [t * axis_unit[0] * axis_unit[2] - s * axis_unit[1],
                             t * axis_unit[1] * axis_unit[2] + s * axis_unit[0],
                             t * axis_unit[2] * axis_unit[2] + c]
                        ])
                        expected_points = points @ R.T + origin
                    else:
                        expected_points = prev_points

                    point_errors = np.linalg.norm(expected_points - curr_points, axis=1)
                    errors.append(np.mean(point_errors))

                mean_error = np.mean(errors) if errors else 1.0

                sigma = 0.05
                self._measurements_likelihood = np.exp(-0.5 * (mean_error / sigma) ** 2)

                if abs(total_rotation) > self._rev_min_rot_for_ee:
                    self._measurements_likelihood = min(1.0, self._measurements_likelihood * 1.5)

            else:
                self._measurements_likelihood = 0.1

        except Exception as e:
            print(f"Error in revolute joint estimation: {e}")
            self._measurements_likelihood = 0.1

    def get_joint_filter_type(self):
        """Get the type of the joint filter"""
        return JointType.REVOLUTE

    def get_joint_filter_type_str(self):
        """Get the type of the joint filter as a string"""
        return "RevoluteJointFilter"

    def extract_joint_parameters(self):
        """Extract joint parameters for visualization"""
        return {
            "type": JointType.REVOLUTE,
            "axis": self._joint_params["axis"],
            "origin": self._joint_params["origin"],
            "motion_limit": self._joint_params["motion_limit"],
            "accumulated_rotation": self._accumulated_rotation,
            "joint_velocity": self._joint_velocity
        }


class DisconnectedJointFilter(JointFilter):
    """Filter for disconnected joints"""

    def __init__(self):
        super().__init__()
        self._unnormalized_model_probability = 0.8
        self._measurement = None

    def estimate_measurement_history_likelihood(self):
        """Estimate the likelihood of the measurement history"""
        self._measurements_likelihood = 0.1

    def get_joint_filter_type(self):
        """Get the type of the joint filter"""
        return JointType.DISCONNECTED

    def get_joint_filter_type_str(self):
        """Get the type of the joint filter as a string"""
        return "DisconnectedJointFilter"

    def extract_joint_parameters(self):
        """Extract joint parameters for visualization"""
        return {
            "type": JointType.DISCONNECTED
        }


class JointCombinedFilter:
    """Combines different joint filters and determines the most probable joint type"""

    def __init__(self):
        self._normalizing_term = 0.25

        # Create the joint filters
        self._joint_filters = {
            JointType.RIGID: RigidJointFilter(),
            JointType.REVOLUTE: RevoluteJointFilter(),
            JointType.PRISMATIC: PrismaticJointFilter(),
            JointType.DISCONNECTED: DisconnectedJointFilter()
        }

    def set_all_points_history(self, all_points_history):
        """Set the point cloud history for all filters"""
        for joint_filter in self._joint_filters.values():
            joint_filter.set_all_points_history(all_points_history)

    def estimate_joint_filter_probabilities(self):
        """Estimate the probabilities for all joint filters"""
        for joint_filter in self._joint_filters.values():
            joint_filter.estimate_measurement_history_likelihood()
            joint_filter.estimate_unnormalized_model_probability()

        self.normalize_joint_filter_probabilities()

    def normalize_joint_filter_probabilities(self):
        """Normalize the joint filter probabilities"""
        sum_unnormalized_probs = sum(
            joint_filter._unnormalized_model_probability
            for joint_filter in self._joint_filters.values()
        )

        self._normalizing_term = max(sum_unnormalized_probs, 1e-5)

        for joint_filter in self._joint_filters.values():
            joint_filter.set_normalizing_term(self._normalizing_term)

    def get_most_probable_joint_filter(self):
        """Get the most probable joint filter"""
        max_probability = -1.0
        most_probable_filter = None

        for joint_filter in self._joint_filters.values():
            probability = joint_filter.get_probability_of_joint_filter()
            if probability > max_probability:
                max_probability = probability
                most_probable_filter = joint_filter

        return most_probable_filter

    def get_joint_filter(self, joint_type):
        """Get a specific joint filter by type"""
        return self._joint_filters.get(joint_type)

    def get_joint_probabilities(self):
        """Get all joint probabilities"""
        return {
            joint_type: joint_filter.get_probability_of_joint_filter()
            for joint_type, joint_filter in self._joint_filters.items()
        }

    def set_measurement(self, twist):
        """设置所有滤波器的观测值"""
        for joint_filter in self._joint_filters.values():
            joint_filter.set_measurement(twist)


###############################################################################
#                Main Joint Analysis Functions                                 #
###############################################################################

def compute_joint_info_all_types(all_points_history):
    """
    Main entry for joint type estimation using advanced filtering:
    Returns (joint_params_dict, best_joint, info_dict)
    """
    if all_points_history.shape[0] < 2:
        ret = {
            "prismatic": {"axis": np.array([0., 0., 0.]), "motion_limit": (0., 0.)},
            "revolute": {"axis": np.array([0., 0., 0.]), "origin": np.array([0., 0., 0.]), "motion_limit": (0., 0.)}
        }
        return ret, "Unknown", None

    # Create and initialize the combined filter
    joint_filter = JointCombinedFilter()
    joint_filter.set_all_points_history(all_points_history)

    # Estimate joint probabilities
    joint_filter.estimate_joint_filter_probabilities()

    # Get the most probable joint type
    most_probable = joint_filter.get_most_probable_joint_filter()
    best_joint = most_probable.get_joint_filter_type_str() if most_probable else "Unknown"

    # Get joint parameters
    joint_params = {}

    # Add parameters for each joint type
    revolute_filter = joint_filter.get_joint_filter(JointType.REVOLUTE)
    if revolute_filter:
        revolute_params = revolute_filter.extract_joint_parameters()
        joint_params["revolute"] = {
            "axis": revolute_params.get("axis", np.array([0., 0., 0.])),
            "origin": revolute_params.get("origin", np.array([0., 0., 0.])),
            "motion_limit": revolute_params.get("motion_limit", (0., 0.))
        }

    prismatic_filter = joint_filter.get_joint_filter(JointType.PRISMATIC)
    if prismatic_filter:
        prismatic_params = prismatic_filter.extract_joint_parameters()
        joint_params["prismatic"] = {
            "axis": prismatic_params.get("axis", np.array([0., 0., 0.])),
            "origin": prismatic_params.get("origin", np.array([0., 0., 0.])),
            "motion_limit": prismatic_params.get("motion_limit", (0., 0.))
        }

    # Get probabilities for info_dict
    probs = joint_filter.get_joint_probabilities()

    # Prepare info_dict with the probabilities
    info_dict = {
        "joint_probs": {
            "prismatic": probs.get(JointType.PRISMATIC, 0.),
            "revolute": probs.get(JointType.REVOLUTE, 0.),
            "rigid": probs.get(JointType.RIGID, 0.),
            "disconnected": probs.get(JointType.DISCONNECTED, 0.)
        },
        "basic_score_avg": {
            "col_mean": 0.0,
            "cop_mean": 0.0,
            "rad_mean": 0.0,
            "zp_mean": 0.0
        }
    }

    # Determine which joint is the best based on probability
    best_joint = "Unknown"
    best_prob = 0.0
    for joint_type, prob in info_dict["joint_probs"].items():
        if prob > best_prob:
            best_prob = prob
            best_joint = joint_type.capitalize()

    return joint_params, best_joint, info_dict


###############################################################################
#                         Enhanced Visualization Class                        #
###############################################################################

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
        self.noise_sigma = 0

        # Load ground truth joint data
        self.ground_truth_json = "./demo_data/joint_info.json"
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

        # Calculate angular velocity for all datasets
        for dataset_key in self.datasets:
            dataset = self.datasets[dataset_key]
            angular_velocity_raw, angular_velocity_filtered = self.calculate_angular_velocity(
                dataset["data_filter"], dataset["N"]
            )
            dataset["angular_velocity_raw"] = angular_velocity_raw
            dataset["angular_velocity_filtered"] = angular_velocity_filtered

            # Perform joint analysis using new advanced filtering methods
            joint_model_dict, best_joint_type = self.perform_joint_analysis(dataset["data_filter"])
            dataset["joint_model_dict"] = joint_model_dict
            dataset["best_joint_type"] = best_joint_type

            # Try to map dataset to ground truth entry
            dataset["ground_truth_key"] = self.map_dataset_to_ground_truth(dataset["display_name"])

        # Current dataset's angular velocity
        self.angular_velocity_raw = self.datasets[self.current_dataset_key]["angular_velocity_raw"]
        self.angular_velocity_filtered = self.datasets[self.current_dataset_key]["angular_velocity_filtered"]

        # Current dataset's joint information
        self.current_joint_model_dict = self.datasets[self.current_dataset_key]["joint_model_dict"]
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
        """使用新的关节分析方法（基于先进滤波器的方法）"""
        # 使用新的关节分析函数
        joint_params_dict, best_joint_type, info_dict = compute_joint_info_all_types(point_history)

        # 打印分析结果
        print("\n" + "=" * 80)
        print(f"Joint Type: {best_joint_type}")
        print("=" * 80)

        if info_dict and "joint_probs" in info_dict:
            joint_probs = info_dict["joint_probs"]
            print(f"Joint Probabilities:")
            print(f"  Prismatic: {joint_probs.get('prismatic', 0.0):.6f}")
            print(f"  Revolute: {joint_probs.get('revolute', 0.0):.6f}")
            print(f"  Rigid: {joint_probs.get('rigid', 0.0):.6f}")
            print(f"  Disconnected: {joint_probs.get('disconnected', 0.0):.6f}")

        # 打印关节参数
        best_joint_lower = best_joint_type.lower()
        if best_joint_lower in joint_params_dict:
            params = joint_params_dict[best_joint_lower]
            print(f"{best_joint_type} Joint Parameters:")

            if "axis" in params:
                axis = params["axis"]
                print(f"  Axis: [{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}]")

            if "origin" in params:
                origin = params["origin"]
                print(f"  Origin: [{origin[0]:.6f}, {origin[1]:.6f}, {origin[2]:.6f}]")

            if "motion_limit" in params:
                motion_limit = params["motion_limit"]
                print(f"  Motion Limit: ({motion_limit[0]:.6f}, {motion_limit[1]:.6f})")

        print("=" * 80 + "\n")

        return joint_params_dict, best_joint_type

    def visualize_joint_parameters(self):
        """Visualize estimated joint parameters in Polyscope"""
        # Remove any existing joint visualizations
        self.remove_joint_visualization()

        joint_model_dict = self.current_joint_model_dict
        best_joint_type = self.current_best_joint_type

        if joint_model_dict is not None and best_joint_type.lower() in joint_model_dict:
            joint_params = joint_model_dict[best_joint_type.lower()]
            self.show_joint_visualization(best_joint_type.lower(), joint_params)

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

    def show_joint_visualization(self, joint_type, joint_params):
        """Show visualization for a specific joint type."""
        if joint_type == "prismatic":
            # Extract parameters
            axis = joint_params.get("axis", np.array([1., 0., 0.]))
            origin = joint_params.get("origin", np.array([0., 0., 0.]))

            # Visualize axis
            seg_nodes = np.array([origin, origin + axis])
            seg_edges = np.array([[0, 1]])
            name = "Prismatic Axis"
            prisviz = ps.register_curve_network(name, seg_nodes, seg_edges)
            prisviz.set_color((0., 1., 1.))
            prisviz.set_radius(0.02)

        elif joint_type == "revolute":
            # Extract parameters
            axis = joint_params.get("axis", np.array([0., 1., 0.]))
            origin = joint_params.get("origin", np.array([0., 0., 0.]))

            # Normalize axis
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-6:
                axis = axis / axis_norm

            # Visualize axis
            seg_nodes = np.array([origin - axis * 0.5, origin + axis * 0.5])
            seg_edges = np.array([[0, 1]])

            name = "Revolute Axis"
            revviz = ps.register_curve_network(name, seg_nodes, seg_edges)
            revviz.set_radius(0.02)
            revviz.set_color((1., 1., 0.))

            # Visualize center
            name = "RevoluteCenterPC"
            c_pc = ps.register_point_cloud(name, origin.reshape(1, 3))
            c_pc.set_radius(0.05)
            c_pc.set_enabled(True)

        elif joint_type == "rigid":
            # For rigid joints, we can show a simple marker
            name = "RigidPositionPC"
            origin = np.array([0., 0., 0.])  # Default position
            r_pc = ps.register_point_cloud(name, origin.reshape(1, 3))
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
            self.current_joint_model_dict = dataset["joint_model_dict"]
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

                # Perform joint analysis using new method
                joint_model_dict, best_joint_type = self.perform_joint_analysis(dataset["data_filter"])
                dataset["joint_model_dict"] = joint_model_dict
                dataset["best_joint_type"] = best_joint_type

            # Update current dataset's values
            self.angular_velocity_raw = self.datasets[self.current_dataset_key]["angular_velocity_raw"]
            self.angular_velocity_filtered = self.datasets[self.current_dataset_key]["angular_velocity_filtered"]
            self.current_joint_model_dict = self.datasets[self.current_dataset_key]["joint_model_dict"]
            self.current_best_joint_type = self.datasets[self.current_dataset_key]["best_joint_type"]

            # Update plot and visualization
            self.plot_image()
            self.visualize_joint_parameters()

            psim.Text("Joint analysis recalculated")

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
                if self.current_joint_model_dict is not None and gt_key:
                    psim.Text("\nComparison with Estimated Joint:")
                    psim.Text(f"Estimated Type: {self.current_best_joint_type}")

                    # Show model-specific comparison
                    best_joint_lower = self.current_best_joint_type.lower()
                    if best_joint_lower in self.current_joint_model_dict and object_type in self.ground_truth_data:
                        if part_info and part_info in self.ground_truth_data[object_type]:
                            gt_data = self.ground_truth_data[object_type][part_info]
                            est_data = self.current_joint_model_dict[best_joint_lower]

                            if "axis" in gt_data and "axis" in est_data:
                                gt_axis = np.array(gt_data["axis"])
                                gt_axis_norm = np.linalg.norm(gt_axis)
                                if gt_axis_norm > 1e-6:
                                    gt_axis = gt_axis / gt_axis_norm

                                est_axis = est_data["axis"]
                                est_axis_norm = np.linalg.norm(est_axis)
                                if est_axis_norm > 1e-6:
                                    est_axis = est_axis / est_axis_norm

                                # Calculate angle between axes
                                dot_product = np.clip(np.abs(np.dot(gt_axis, est_axis)), 0.0, 1.0)
                                angle_diff = np.arccos(dot_product)
                                angle_diff_deg = np.degrees(angle_diff)

                                psim.Text(f"Axis Angle Difference: {angle_diff_deg:.2f}°")
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
            if self.current_joint_model_dict is not None and self.current_best_joint_type.lower() in self.current_joint_model_dict:
                joint_params = self.current_joint_model_dict[self.current_best_joint_type.lower()]

                psim.Text(f"Model Type: {self.current_best_joint_type}")

                if "axis" in joint_params:
                    axis = joint_params["axis"]
                    psim.Text(f"Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")

                if "origin" in joint_params:
                    origin = joint_params["origin"]
                    psim.Text(f"Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")

                if "motion_limit" in joint_params:
                    motion_limit = joint_params["motion_limit"]
                    psim.Text(f"Motion Limit: ({motion_limit[0]:.3f}, {motion_limit[1]:.3f})")

            else:
                psim.Text("No joint model available")

            psim.TreePop()


# Program entry point
if __name__ == "__main__":
    # You can specify multiple data file paths here
    file_paths = [
        # open refrigerator 1
        # "./demo_data/s1_refrigerator_part2_3180_3240.npy",
        # "./demo_data/s1_refrigerator_base_3180_3240.npy",
        # "./demo_data/s1_refrigerator_part1_3180_3240.npy"

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

        # close washingmachine  1 1
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
        #
        # # #ball
        # "./demo_data/ball.npy"

        "./demo_data/prismatic.npy"

    ]

    # Create EnhancedViz instance and execute visualization
    viz = EnhancedViz(file_paths)