import numpy as np
import os
import json
import re
from scipy.signal import savgol_filter
from datetime import datetime
import sys
from collections import defaultdict
import glob
import math
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

# sys.path.append('/common/homes/all/uksqc_chen/projects/control')
# Import joint analysis project modules - commented out since we're implementing it directly



def quaternion_to_matrix(qx, qy, qz, qw):
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
        return "rigid"

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
        return "prismatic"

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
        return "revolute"

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
        return "disconnected"

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


###############################################################################
#                Main Joint Analysis Functions (Martin's Method)               #
###############################################################################

def compute_joint_info_all_types_martin(all_points_history):
    """
    Martin's method for joint type estimation using advanced filtering:
    Returns (joint_params_dict, best_joint, info_dict)
    """
    if all_points_history.shape[0] < 2:
        ret = {
            "prismatic": {"axis": np.array([0., 0., 0.]), "origin": np.array([0., 0., 0.]), "motion_limit": (0., 0.)},
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
        self.ground_truth_json = "./parahome_data_slide/all_scenes_transformed_axis_pivot_data.json"
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
        """Perform joint analysis on point history data using Martin's method"""
        joint_params, best_joint, info_dict = compute_joint_info_all_types_martin(point_history)
        return joint_params, best_joint, info_dict

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
            return {
                "filename": filename,
                "scene_info": scene_info,
                "error": f"Could not load or process file: {e}"
            }

        # Perform joint analysis using Martin's method
        try:
            joint_params, best_joint, info_dict = self.perform_joint_analysis(data_filter)
        except Exception as e:
            return {
                "filename": filename,
                "scene_info": scene_info,
                "error": f"Joint analysis failed: {e}"
            }

        # Initialize result
        result = {
            "filename": filename,
            "scene_info": scene_info,
            "expected_joint_type": expected_joint_type,
            "detected_joint_type": best_joint,
            "classification_correct": best_joint.lower() == expected_joint_type,
            "joint_params": joint_params.get(best_joint.lower(), {}),
            "all_joint_probabilities": info_dict.get("joint_probs", {}) if info_dict else {}
        }

        # Calculate errors if classification is correct
        if result["classification_correct"]:
            ground_truth = self.get_ground_truth_for_object(scene_info)
            if ground_truth:
                result["ground_truth"] = ground_truth
                result["errors"] = self.calculate_joint_errors(best_joint.lower(), joint_params[best_joint.lower()], ground_truth)

        return result

    def evaluate_all_groups(self):
        """Evaluate all groups and return comprehensive results"""
        print("Starting improved joint analysis evaluation using Martin's method...")
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
            "method": "Martin's Advanced Joint Analysis",
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

    def print_summary(self, results):
        """Print a summary of the evaluation results"""
        print("\n" + "=" * 80)
        print("MARTIN'S METHOD EVALUATION SUMMARY")
        print("=" * 80)

        overall_stats = results["overall_statistics"]
        print(f"Method: {results['method']}")
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
    directory_path = "./parahome_data_slide"
    noise_levels = [0.01]  # Different noise levels to test

    for noise_std in noise_levels:
        print(f"\n{'=' * 60}")
        print(f"Running evaluation with Martin's method - noise std = {noise_std}")
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

        # Save results with martin in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        noise_str = f"noise_{noise_std:g}".replace('.', 'p')
        output_file = f"martin_joint_analysis_evaluation_{noise_str}_{timestamp}.json"

        success = evaluator.save_results(results, output_file)

        if success:
            print(f"\n✓ Evaluation completed successfully using Martin's method!")
            print(f"✓ Results saved to: {output_file}")
        else:
            print(f"\n✗ Failed to save results")


if __name__ == "__main__":
    main()