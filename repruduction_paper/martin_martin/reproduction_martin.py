import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import os
import math
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union


# DearPyGUI
import dearpygui.dearpygui as dpg
import threading



################################################################################
#                               Twist & SE(3) Utils                            #
################################################################################


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




def unwrap_twist(twist, previous_twist, out_inverted=None):
   """Handle discontinuities in twist representation"""
   # Similar to unwrapTwist function in C++ code
   current_norm = np.linalg.norm(twist.angular)
   previous_norm = np.linalg.norm(previous_twist.angular)


   inverted = False


   # Check if we need to invert the twist to avoid discontinuities
   if abs(current_norm - previous_norm) > np.pi:
       dot_product = np.dot(twist.angular, previous_twist.angular)
       if dot_product < 0:
           # Invert the twist
           twist.angular = -twist.angular
           twist.linear = -twist.linear
           inverted = True


   if out_inverted is not None:
       out_inverted = inverted


   return twist




################################################################################
#                           Geometry and Motion Utils                          #
################################################################################




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
       [t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
       [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0]],
       [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c]
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
       [0, 1., 0.],
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
       [np.sin(angle_z), np.cos(angle_z), 0],
       [0, 0, 1]
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
       [t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
       [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0]],
       [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c]
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




################################################################################
#                         Joint Classification Framework                       #
################################################################################


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
       """设置观测值 - 从第二段代码整合"""
       self._measurement = twist


   def get_probability(self):
       """获取模型概率 - 从第二段代码整合"""
       return self._measurements_likelihood




class RigidJointFilter(JointFilter):
   """Enhanced filter for rigid joints based on the C++ implementation"""


   def __init__(self):
       super().__init__()
       # Initially the most probable joint type is rigid
       self._measurements_likelihood = 0
       self._motion_memory_prior = 1  # 初始化为1.0表示没有先验限制
       self._rig_max_translation = 0.005
       self._rig_max_rotation = 0.01


       # Additional fields for prediction
       self._predicted_delta_pose_in_rrbf = None
       self._srb_predicted_pose_in_rrbf = None
       self._measurement = None


   def initialize(self):
       """Initialize the filter, resetting likelihood to 1.0"""
       super().initialize()
       # Initially the most probable joint type is rigid
       self._measurements_likelihood = 1.0


   def set_max_translation_rigid(self, max_trans):
       """Set the maximum allowed translation for a rigid joint"""
       self._rig_max_translation = max_trans


   def set_max_rotation_rigid(self, max_rot):
       """Set the maximum allowed rotation for a rigid joint"""
       self._rig_max_rotation = max_rot


   def predict_measurement(self):
       """Predict the next measurement based on rigid joint hypothesis"""
       # For a rigid joint, the predicted delta is identity (zero motion)
       self._predicted_delta_pose_in_rrbf = Twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


       # Compute the predicted pose in reference rigid body frame
       predicted_delta = self._predicted_delta_pose_in_rrbf.exp()


       # Check if we have initial pose information
       if hasattr(self, '_srb_initial_pose_in_rrbf'):
           T_rrbf_srbf_t0 = self._srb_initial_pose_in_rrbf.exp()
           T_rrbf_srbf_t_next = np.dot(predicted_delta, T_rrbf_srbf_t0)


           # Convert back to twist representation
           self._srb_predicted_pose_in_rrbf = Twist.log(T_rrbf_srbf_t_next)


   def estimate_measurement_history_likelihood(self):
       """Estimate the likelihood of the measurement history"""
       if len(self._delta_poses_in_rrbf) == 0:
           self._measurements_likelihood = 1.0
           return


       p_one_meas_given_model_params = 0.0
       p_all_meas_given_model_params = 0.0


       sigma_translation = 0.05
       sigma_rotation = 0.2


       accumulated_error = 0.0
       frame_counter = 0.0


       # Estimate the number of samples to use
       trajectory_length = len(self._delta_poses_in_rrbf)
       amount_samples = min(trajectory_length, self._likelihood_sample_num)


       # Uniform sampling across the trajectory (similar to C++ implementation)
       delta_idx_samples = max(1.0, float(trajectory_length) / float(self._likelihood_sample_num))
       current_idx = 0


       # Evaluate the likelihood for each sampled delta pose
       for sample_idx in range(amount_samples):
           current_idx = min(int(round(sample_idx * delta_idx_samples)), trajectory_length - 1)


           # Get the delta pose from the history
           rb2_last_delta_relative_twist = self._delta_poses_in_rrbf[current_idx]
           rb2_last_delta_relative_displ = rb2_last_delta_relative_twist.exp()


           # Extract translation and rotation
           rb2_last_delta_relative_translation = rb2_last_delta_relative_displ[:3, 3]
           rb2_last_delta_relative_rotation = Rotation.from_matrix(rb2_last_delta_relative_displ[:3, :3])


           # For a rigid joint, the expected delta is identity
           rigid_joint_translation = np.zeros(3)
           rb2_last_delta_relative_displ_rigid_hyp = Twist().exp()


           rb2_last_delta_relative_translation_rigid_hyp = rb2_last_delta_relative_displ_rigid_hyp[:3, 3]
           rb2_last_delta_relative_rotation_rigid_hyp = Rotation.from_matrix(
               rb2_last_delta_relative_displ_rigid_hyp[:3, :3])


           # Compute translation error
           translation_error = np.linalg.norm(
               rb2_last_delta_relative_translation - rb2_last_delta_relative_translation_rigid_hyp
           )


           # Check if translation error exceeds maximum
           if translation_error > self._rig_max_translation:
               self._motion_memory_prior = 0.0


           # Compute rotation error using as_rotvec() for the rotation error angle
           rotation_error = rb2_last_delta_relative_rotation.inv() * rb2_last_delta_relative_rotation_rigid_hyp
           rotation_error_angle = np.linalg.norm(rotation_error.as_rotvec())


           # Check if rotation error exceeds maximum
           if rotation_error_angle > self._rig_max_rotation:
               self._motion_memory_prior = 0.0


           accumulated_error += translation_error + abs(rotation_error_angle)


           # Compute likelihood for this measurement (Gaussian probability)
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


   def get_predicted_srb_delta_pose_with_cov_in_sensor_frame(self):
       """Get predicted delta pose with covariance in sensor frame"""
       # For a rigid joint, the delta in the pose of the SRB is the delta in the pose of the RRB
       if hasattr(self, '_rrb_current_vel_in_sf') and hasattr(self, '_loop_period_ns'):
           time_factor = self._loop_period_ns / 1e9
           delta_rrb_in_sf = self._rrb_current_vel_in_sf * time_factor
           delta_srb_in_sf = delta_rrb_in_sf


           # Create a twist with covariance structure similar to the C++ version
           result = {
               "twist": {
                   "linear": {
                       "x": delta_srb_in_sf.linear[0],
                       "y": delta_srb_in_sf.linear[1],
                       "z": delta_srb_in_sf.linear[2]
                   },
                   "angular": {
                       "x": delta_srb_in_sf.angular[0],
                       "y": delta_srb_in_sf.angular[1],
                       "z": delta_srb_in_sf.angular[2]
                   }
               },
               "covariance": np.zeros(36)
           }


           # Add covariance if available
           if hasattr(self, '_rrb_vel_cov_in_sf'):
               # Flatten the 6x6 covariance matrix into a 36-element array
               cov_flat = self._rrb_vel_cov_in_sf.flatten() * time_factor
               result["covariance"] = cov_flat


           return result


       return None


   def predict_reference_body_motion(self):
       """预测参考刚体的下一运动 - 从第二段代码整合"""
       # 对于刚性关节，参考刚体通常固定
       return Twist()


   def predict_secondary_body_motion(self):
       """预测次要刚体的下一运动 - 从第二段代码整合"""
       # 对于刚性关节，次要刚体与参考刚体一起运动
       return Twist()


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
       # 从第二段代码整合
       self._measurement = None
       self._joint_variable = 0.0  # q_p (位移)
       self._joint_velocity = 0.0


   def estimate_measurement_history_likelihood(self):
       """Estimate the likelihood of the measurement history"""
       if len(self._delta_poses_in_rrbf) == 0 or self._all_points_history is None:
           self._measurements_likelihood = 0.1
           return


       # Extract the point cloud data
       T, N, _ = self._all_points_history.shape


       if T < 3:
           self._measurements_likelihood = 0.1
           return


       # Estimate the axis of translation
       # We'll take the main direction of all point movements


       # Calculate the average displacement vector for all points
       displacements = []
       for i in range(N):
           start_pos = self._all_points_history[0, i]
           end_pos = self._all_points_history[-1, i]
           displacement = end_pos - start_pos
           displacements.append(displacement)


       displacement_vectors = np.array(displacements)
       avg_displacement = np.mean(displacement_vectors, axis=0)


       # Normalize to get the axis
       displacement_norm = np.linalg.norm(avg_displacement)
       if displacement_norm > 1e-6:
           self._joint_params["axis"] = avg_displacement / displacement_norm


           # Check if this is primarily a translation (low rotation component)
           # by comparing angular and linear velocity
           has_low_rotation = True
           for twist in self._delta_poses_in_rrbf:
               angular_norm = np.linalg.norm(twist.angular)
               linear_norm = np.linalg.norm(twist.linear)
               if angular_norm > 0.1 * linear_norm and angular_norm > 0.01:
                   has_low_rotation = False
                   break


           if has_low_rotation:
               # This is likely a prismatic joint
               # Set the origin to the first frame centroid
               self._joint_params["origin"] = np.mean(self._all_points_history[0], axis=0)


               # Set motion limits based on total displacement along the axis
               total_displacement = np.dot(avg_displacement, self._joint_params["axis"])
               self._joint_variable = total_displacement  # 从第二段代码整合


               # 估计关节速度 - 从第二段代码整合
               if T > 1:
                   frame_time = 1.0  # 假设单位时间步长
                   self._joint_velocity = total_displacement / ((T - 1) * frame_time)


               if total_displacement >= 0:
                   self._joint_params["motion_limit"] = (0.0, total_displacement)
               else:
                   self._joint_params["motion_limit"] = (total_displacement, 0.0)


               # Higher displacement means higher likelihood of being prismatic
               self._measurements_likelihood = min(1.0, abs(total_displacement) / 2.0)
           else:
               # Significant rotation component, less likely to be prismatic
               self._measurements_likelihood = 0.1
       else:
           # Very small displacement, likely not prismatic
           self._measurements_likelihood = 0.1


   def predict_reference_body_motion(self):
       """预测参考刚体的下一运动 - 从第二段代码整合"""
       # 对于棱柱关节，参考刚体通常固定
       return Twist()


   def predict_secondary_body_motion(self):
       """预测次要刚体的下一运动 - 从第二段代码整合"""
       # 次要刚体沿轴方向移动
       prediction = Twist()
       prediction.linear = self._joint_params["axis"] * self._joint_velocity
       return prediction


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
   """Enhanced filter for revolute joints with better tracking and uncertainty modeling"""


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
       self._uncertainty_joint_orientation = np.eye(3) * 0.1
       self._uncertainty_joint_position = np.eye(3) * 0.1
       self._uncertainty_joint_state = 0.1
       self._rev_min_rot_for_ee = 0.1
       self._rev_max_joint_distance_for_ee = 0.5


       # For handling rotation unwrapping
       self._from_inverted_to_non_inverted = False
       self._from_non_inverted_to_inverted = False
       self._inverted_delta_srb_pose_in_rrbf = False


       # 从第二段代码整合
       self._measurement = None
       self._joint_variable = 0.0  # q_r (角度)


   def estimate_measurement_history_likelihood(self):
       """Estimate the likelihood of the measurement history with improved error modeling"""
       # At the beginning of the method
       if len(self._delta_poses_in_rrbf) == 0 or self._all_points_history is None:
           self._measurements_likelihood = 0.1
           return


       # Extract the point cloud data
       T, N, _ = self._all_points_history.shape


       if T < 3:
           self._measurements_likelihood = 0.1
           return


       # Estimate the axis of rotation using a more robust method
       # Similar to the C++ implementation, we'll analyze the motion patterns


       # First, identify the axis using PCA on the point movements
       points_trajectories = np.zeros((N, T, 3))
       for i in range(N):
           for t in range(T):
               points_trajectories[i, t] = self._all_points_history[t, i]


       # Calculate mean positions for each point
       mean_positions = np.mean(points_trajectories, axis=1)


       # Calculate the mean position of all points (centroid)
       centroid = np.mean(mean_positions, axis=0)


       # Calculate variances in each direction
       variances = np.zeros((N, 3))
       for i in range(N):
           variances[i] = np.var(points_trajectories[i], axis=0)


       # Points with highest variance are likely to be moving the most
       total_variance = np.sum(variances, axis=1)
       high_variance_indices = np.argsort(total_variance)[-int(N / 3):]
       high_var_points = points_trajectories[high_variance_indices]


       # Compute PCA for these points to find the rotation axis
       centered_points = high_var_points - centroid
       all_points_flat = centered_points.reshape(-1, 3)


       try:
           cov_matrix = np.cov(all_points_flat.T)
           eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)


           # The eigenvector with the smallest eigenvalue should be close to the rotation axis
           axis_idx = np.argmin(eig_vals)
           axis = eig_vecs[:, axis_idx]


           # Normalize the axis
           axis = axis / np.linalg.norm(axis)
           self._joint_params["axis"] = axis


           # More precise estimation of the rotation center
           # Using a geometric approach similar to the C++ implementation


           # For each point, compute its distance to the estimated axis
           # The rotation center is a point on the axis that minimizes the sum of squares of distances
           # This is essentially finding the closest point on the axis to all trajectories


           # First convert the problem to a least squares formulation
           A = np.eye(3) - np.outer(axis, axis)
           b = np.zeros(3)


           for i in high_variance_indices:
               point_mean = mean_positions[i]
               b += A @ point_mean


           # Solve for the point on the axis closest to all trajectories
           origin = np.linalg.lstsq(A, b, rcond=None)[0]
           self._joint_params["origin"] = origin


           # Now that we have a better estimate of the axis and origin, compute the joint angles
           joint_states = []


           # Calculate angles between consecutive frames
           for t in range(1, T):
               prev_points = self._all_points_history[t - 1]
               curr_points = self._all_points_history[t]


               # Get the points with significant motion
               prev_centered = prev_points - origin
               curr_centered = curr_points - origin


               # Project points onto a plane perpendicular to the rotation axis
               proj_matrix = np.eye(3) - np.outer(axis, axis)
               prev_proj = prev_centered @ proj_matrix.T
               curr_proj = curr_centered @ proj_matrix.T


               # Find the rotation angle using vector dot products
               valid_indices = np.logical_and(
                   np.linalg.norm(prev_proj, axis=1) > 1e-6,
                   np.linalg.norm(curr_proj, axis=1) > 1e-6
               )


               if np.sum(valid_indices) > 0:
                   # Calculate dot products and normalize
                   dot_products = np.sum(prev_proj[valid_indices] * curr_proj[valid_indices], axis=1)
                   norms_prev = np.linalg.norm(prev_proj[valid_indices], axis=1)
                   norms_curr = np.linalg.norm(curr_proj[valid_indices], axis=1)


                   # Compute cosines, clamping to valid range for arccos
                   cosines = np.clip(dot_products / (norms_prev * norms_curr), -1.0, 1.0)


                   # Get angles and use cross product to determine sign
                   angles = np.arccos(cosines)


                   # Determine rotation direction using cross product
                   cross_products = np.cross(prev_proj[valid_indices], curr_proj[valid_indices])
                   dot_with_axis = np.sum(cross_products * axis, axis=1)
                   signs = np.sign(dot_with_axis)


                   # Apply signs to angles and compute average
                   signed_angles = angles * signs
                   avg_angle = np.mean(signed_angles)


                   joint_states.append(avg_angle)


           # Accumulate the rotation, handling discontinuities
           total_rotation = 0
           if joint_states:
               total_rotation = np.sum(joint_states)


               # Update the accumulated rotation
               if len(self._joint_states_all) > 0:
                   last_state = self._joint_states_all[-1]


                   # Check for discontinuities (e.g., around ±π)
                   if abs(total_rotation - last_state) > np.pi:
                       # Handle the discontinuity - similar to unwrapTwist in C++ code
                       if total_rotation * last_state < 0:  # Different signs
                           if total_rotation < 0:
                               self._from_inverted_to_non_inverted = True
                           else:
                               self._from_non_inverted_to_inverted = True


               self._joint_states_all.append(total_rotation)


               # Update joint velocity
               if len(self._joint_states_all) >= 2:
                   time_step = 1.0  # Assuming unit time between frames
                   self._joint_velocity = (total_rotation - self._prev_joint_state) / time_step


               self._prev_joint_state = total_rotation
               self._joint_state = total_rotation
               # 从第二段代码整合
               self._joint_variable = total_rotation


               # Set motion limits based on observed rotation
               min_rotation = min(joint_states) if joint_states else 0
               max_rotation = max(joint_states) if joint_states else 0
               self._joint_params["motion_limit"] = (min_rotation, max_rotation)


               # Compute measurement likelihood based on the consistency of the axis
               # Higher likelihood if points consistently rotate around the estimated axis


               # We'll compute an error metric based on how well the points adhere to the revolute motion model
               errors = []
               for t in range(1, T):
                   prev_points = self._all_points_history[t - 1]
                   curr_points = self._all_points_history[t]


                   # Calculate the expected positions based on the revolute model
                   angle = joint_states[t - 1]
                   expected_points = rotate_points(prev_points, angle, axis, origin)


                   # Calculate errors between expected and actual positions
                   point_errors = np.linalg.norm(expected_points - curr_points, axis=1)
                   errors.append(np.mean(point_errors))


               mean_error = np.mean(errors) if errors else 1.0


               # Convert error to likelihood (higher error = lower likelihood)
               # Using a Gaussian-like function to convert error to probability
               sigma = 0.05  # Adjust based on expected error scale
               self._measurements_likelihood = np.exp(-0.5 * (mean_error / sigma) ** 2)


               # Boost likelihood if we've observed significant rotation
               if abs(total_rotation) > self._rev_min_rot_for_ee:
                   self._measurements_likelihood = min(1.0, self._measurements_likelihood * 1.5)


           else:
               self._measurements_likelihood = 0.1


       except Exception as e:
           print(f"Error in revolute joint estimation: {e}")
           self._measurements_likelihood = 0.1


   def predict_reference_body_motion(self):
       """预测参考刚体的下一运动 - 从第二段代码整合"""
       # 对于旋转关节，参考刚体通常固定
       return Twist()


   def predict_secondary_body_motion(self):
       """预测次要刚体的下一运动 - 从第二段代码整合"""
       # 次要刚体绕轴旋转
       prediction = Twist()
       axis_normalized = self._joint_params["axis"] / np.linalg.norm(self._joint_params["axis"])
       prediction.angular = axis_normalized * self._joint_velocity


       # 计算由旋转引起的线性运动
       prediction.linear = np.cross(prediction.angular, -self._joint_params["origin"])


       return prediction


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
           "joint_velocity": self._joint_velocity,
           "uncertainty_position": self._uncertainty_joint_position,
           "uncertainty_orientation": self._uncertainty_joint_orientation
       }




class DisconnectedJointFilter(JointFilter):
   """Filter for disconnected joints"""


   def __init__(self):
       super().__init__()
       self._unnormalized_model_probability = 0.8
       self._measurement = None  # 从第二段代码整合


   def estimate_measurement_history_likelihood(self):
       """Estimate the likelihood of the measurement history"""
       # For disconnected joints, we use a fixed likelihood
       self._measurements_likelihood = 0.1


   def predict_reference_body_motion(self):
       """预测参考刚体的下一运动 - 从第二段代码整合"""
       # 对于断开连接，不做预测
       return None


   def predict_secondary_body_motion(self):
       """预测次要刚体的下一运动 - 从第二段代码整合"""
       # 对于断开连接，不做预测
       return None


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
       # Compute the sum of unnormalized probabilities
       sum_unnormalized_probs = sum(
           joint_filter._unnormalized_model_probability
           for joint_filter in self._joint_filters.values()
       )


       # Set the normalizing term
       self._normalizing_term = max(sum_unnormalized_probs, 1e-5)


       # Update all filters with the new normalizing term
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
       """设置所有滤波器的观测值 - 从第二段代码整合"""
       for joint_filter in self._joint_filters.values():
           joint_filter.set_measurement(twist)




################################################################################
#                    Multi-Level Recursive Estimation Framework                 #
################################################################################


class FeatureMotionEstimator:
   """
   第一层递归状态估计：特征运动估计
   根据点云中的3D特征跟踪估计特征运动


   修改: 使用所有点作为特征点，不进行抽样
   """


   def __init__(self):
       # 移除 num_features 参数，使用所有点作为特征
       self.features_3d = None  # 状态 x^fm_t ∈ R^3N
       self.rigid_body_velocities = {}  # 从更高层接收的刚体速度
       self.feature_to_rigid_body = {}  # 特征到刚体的映射
       self.feature_covariances = None  # 特征位置的协方差


   def initialize_features(self, initial_3d_points):
       """初始化特征点 - 使用所有点作为特征"""
       self.features_3d = initial_3d_points.copy()
       self.feature_covariances = np.ones((len(initial_3d_points), 3, 3)) * 0.01


   def predict_feature_motion(self, dt=0.1):
       """
       预测步骤：根据上一帧位置和从第二层得到的刚体速度预测特征的下一位置
       """
       if self.features_3d is None:
           return


       predicted_features = self.features_3d.copy()


       # 使用每个特征所属刚体的速度来预测特征运动
       for i, feat_pos in enumerate(self.features_3d):
           if i in self.feature_to_rigid_body:
               rb_id = self.feature_to_rigid_body[i]
               if rb_id in self.rigid_body_velocities:
                   twist = self.rigid_body_velocities[rb_id]


                   # 构造变换矩阵
                   transform = twist.exp()


                   # 应用变换
                   homogeneous_pos = np.append(feat_pos, 1.0)
                   transformed_pos = transform @ homogeneous_pos
                   predicted_features[i] = transformed_pos[:3]


       return predicted_features


   def update_feature_motion(self, new_3d_features):
       """
       测量更新步骤：更新3D特征位置
       """
       if self.features_3d is None or new_3d_features is None:
           return self.features_3d


       # 更新状态
       self.features_3d = new_3d_features.copy()
       return self.features_3d


   def assign_feature_to_rigid_body(self, feature_idx, rigid_body_id):
       """将特征分配给刚体"""
       self.feature_to_rigid_body[feature_idx] = rigid_body_id


   def receive_rigid_body_velocities(self, rigid_body_velocities):
       """从第二层接收刚体速度信息"""
       self.rigid_body_velocities = rigid_body_velocities




class RigidBodyMotionEstimator:
   """
   第二层递归状态估计：刚体运动估计
   基于特征运动估计刚体的姿态和速度
   """


   def __init__(self):
       self.rigid_bodies = {}  # 刚体ID到特征索引的映射
       self.rigid_body_poses = {}  # 刚体ID到姿态的映射 (p in SE(3))
       self.rigid_body_velocities = {}  # 刚体ID到速度的映射 (v as twist)
       self.kinematic_model_predictions = {}  # 从第三层接收的预测


   def initialize_rigid_body(self, rb_id, feature_indices, initial_features_3d):
       """初始化一个新的刚体"""
       self.rigid_bodies[rb_id] = feature_indices


       # 初始化刚体姿态为单位矩阵
       self.rigid_body_poses[rb_id] = np.eye(4)


       # 计算刚体中心作为参考
       if len(feature_indices) > 0:
           features = [initial_features_3d[i] for i in feature_indices]
           centroid = np.mean(features, axis=0)
           self.rigid_body_poses[rb_id][:3, 3] = centroid


       # 初始化刚体速度为零
       self.rigid_body_velocities[rb_id] = Twist()


   def detect_rigid_bodies(self, features_3d, prev_features_3d, min_features=15):
       """
       使用RANSAC检测刚体
       如果一些特征的运动无法被现有刚体解释，则创建新的刚体
       """
       # 这里是简化实现，实际应使用RANSAC
       all_assigned = set()
       for rb_id, feature_indices in self.rigid_bodies.items():
           all_assigned.update(feature_indices)


       unassigned = [i for i in range(len(features_3d)) if i not in all_assigned]


       if len(unassigned) >= min_features:
           # 创建新刚体
           new_rb_id = len(self.rigid_bodies)
           self.initialize_rigid_body(new_rb_id, unassigned, features_3d)
           return new_rb_id


       return None


   def predict_rigid_body_motion(self, dt=0.1):
       """
       预测步骤：基于三种预测模型预测刚体运动
       """
       predicted_poses = {}


       for rb_id, pose in self.rigid_body_poses.items():
           # 标准运动模型
           if rb_id in self.rigid_body_velocities:
               twist = self.rigid_body_velocities[rb_id]
               delta_transform = twist.exp()
               std_prediction = delta_transform @ pose
               predicted_poses[rb_id] = std_prediction


           # 零速度模型
           zero_vel_prediction = pose.copy()


           # 基于运动学模型的预测
           kinematic_prediction = None
           if rb_id in self.kinematic_model_predictions:
               kinematic_prediction = self.kinematic_model_predictions[rb_id]


           # 简化为使用标准预测
           if rb_id not in predicted_poses:
               predicted_poses[rb_id] = zero_vel_prediction


       return predicted_poses


   def update_rigid_body_motion(self, features_3d, feature_to_rigid_body):
       """
       测量更新步骤：基于特征3D位置更新刚体运动
       """
       for rb_id, feature_indices in self.rigid_bodies.items():
           # 获取属于这个刚体的特征
           rb_features = []
           for i, feat in enumerate(features_3d):
               if i in feature_indices:
                   rb_features.append(feat)


           if not rb_features:
               continue


           # 获取当前刚体姿态
           current_pose = self.rigid_body_poses[rb_id]


           # 通过特征位置估计更新的刚体姿态
           # 这里简化为取特征的平均位置更新质心
           new_centroid = np.mean(rb_features, axis=0)
           updated_pose = current_pose.copy()
           updated_pose[:3, 3] = new_centroid


           # 计算位姿变化量
           delta_pose = np.linalg.inv(current_pose) @ updated_pose


           # 更新速度
           delta_twist = Twist.log(delta_pose)


           # 更新状态
           self.rigid_body_poses[rb_id] = updated_pose
           self.rigid_body_velocities[rb_id] = delta_twist


       return self.rigid_body_poses, self.rigid_body_velocities


   def receive_kinematic_predictions(self, predictions):
       """从第三层接收运动学模型预测"""
       self.kinematic_model_predictions = predictions




class KinematicModelEstimator:
   """
   第三层递归状态估计：运动学模型估计
   基于刚体运动估计关节类型和参数
   """


   def __init__(self):
       self.joint_filters = {}  # 每对刚体间的关节滤波器
       self.joint_types = {}  # 每对刚体间的关节类型
       self.joint_parameters = {}  # 每对刚体间的关节参数


   def initialize_joint_filters(self, rb_pairs):
       """为每对刚体初始化所有类型的关节滤波器"""
       for rb1, rb2 in rb_pairs:
           pair_key = (min(rb1, rb2), max(rb1, rb2))
           if pair_key not in self.joint_filters:
               self.joint_filters[pair_key] = JointCombinedFilter()
               # 初始假设为断开连接
               self.joint_types[pair_key] = 'disconnected'


   def estimate_joint_parameters(self, rb_poses, rb_velocities):
       """
       估计每对刚体间的关节类型和参数
       实现论文中的多类型关节模型估计
       """
       joint_predictions = {}


       for pair_key, joint_filter in self.joint_filters.items():
           rb1, rb2 = pair_key


           # 计算刚体间的相对运动 (twist)
           if rb1 in rb_poses and rb2 in rb_poses:
               pose1 = rb_poses[rb1]
               pose2 = rb_poses[rb2]


               # 相对位姿 T_1^-1 * T_2
               relative_pose = np.linalg.inv(pose1) @ pose2


               # 相对运动 (twist representation)
               relative_twist = Twist.log(relative_pose)


               # 设置观测值
               joint_filter.set_measurement(relative_twist)


               # 估计关节概率
               joint_filter.estimate_joint_filter_probabilities()


               # 获取最可能的关节类型
               most_probable = joint_filter.get_most_probable_joint_filter()
               if most_probable:
                   best_type = most_probable.get_joint_filter_type_str().lower().replace('jointfilter', '')
                   self.joint_types[pair_key] = best_type


                   # 提取关节参数
                   self.joint_parameters[pair_key] = most_probable.extract_joint_parameters()


                   # 生成预测
                   if best_type in ['revolute', 'prismatic']:
                       rb1_prediction = most_probable.predict_reference_body_motion()
                       rb2_prediction = most_probable.predict_secondary_body_motion()


                       if rb1_prediction:
                           joint_predictions[rb1] = rb1_prediction
                       if rb2_prediction:
                           joint_predictions[rb2] = rb2_prediction


       return self.joint_types, self.joint_parameters, joint_predictions




class MultiLevelArticulatedPerception:
   """
   多层次递归估计框架
   整合三个层级的估计器，实现双向信息流


   修改: 使用所有点作为特征
   """


   def __init__(self):
       # 初始化三个层级的估计器 - 移除 num_features 参数
       self.feature_estimator = FeatureMotionEstimator()
       self.rigid_body_estimator = RigidBodyMotionEstimator()
       self.kinematic_estimator = KinematicModelEstimator()


       # 跟踪已知的刚体ID
       self.next_body_id = 0
       self.rigid_body_pairs = []


       # 点云历史
       self.all_points_history = []


   def initialize(self, initial_point_cloud):
       """初始化系统"""
       # 保存点云历史
       self.all_points_history = [initial_point_cloud.copy()]


       # 初始化特征
       self.feature_estimator.initialize_features(initial_point_cloud)


       # 创建初始刚体
       rb_id = self.next_body_id
       self.next_body_id += 1


       feature_indices = list(range(len(initial_point_cloud)))
       self.rigid_body_estimator.initialize_rigid_body(rb_id, feature_indices, initial_point_cloud)


       # 将所有特征分配给刚体
       for i in feature_indices:
           self.feature_estimator.assign_feature_to_rigid_body(i, rb_id)


   def process_frame(self, new_point_cloud, dt=0.1):
       """
       处理新的点云帧
       实现完整的多层递归状态估计
       """
       # 更新点云历史
       self.all_points_history.append(new_point_cloud.copy())


       # 1. Level 1: 特征运动估计
       # 预测特征运动
       predicted_features = self.feature_estimator.predict_feature_motion(dt)


       # 更新特征
       updated_features = self.feature_estimator.update_feature_motion(new_point_cloud)


       # 2. Level 2: 刚体运动估计
       # 检测新的刚体
       new_rb_id = self.rigid_body_estimator.detect_rigid_bodies(updated_features, predicted_features)
       if new_rb_id is not None:
           # 如果检测到新刚体，更新刚体对
           for rb_id in self.rigid_body_estimator.rigid_bodies.keys():
               if rb_id != new_rb_id:
                   self.rigid_body_pairs.append((rb_id, new_rb_id))


           # 初始化关节滤波器
           self.kinematic_estimator.initialize_joint_filters(self.rigid_body_pairs)


       # 预测刚体运动
       predicted_rb_poses = self.rigid_body_estimator.predict_rigid_body_motion(dt)


       # 使用特征运动更新刚体运动
       feature_to_rigid_body = self.feature_estimator.feature_to_rigid_body
       rb_poses, rb_velocities = self.rigid_body_estimator.update_rigid_body_motion(updated_features,
                                                                                    feature_to_rigid_body)


       # 3. Level 3: 运动学模型估计
       joint_types, joint_parameters, joint_predictions = self.kinematic_estimator.estimate_joint_parameters(rb_poses,
                                                                                                             rb_velocities)


       # 4. 双向信息流：向下反馈
       # 将刚体速度传递给特征运动估计器
       self.feature_estimator.receive_rigid_body_velocities(rb_velocities)


       # 将关节预测传递给刚体运动估计器
       self.rigid_body_estimator.receive_kinematic_predictions(joint_predictions)


       return joint_types, joint_parameters


   def get_joint_estimation_results(self):
       """获取关节估计结果"""
       results = {
           "joint_types": self.kinematic_estimator.joint_types,
           "joint_parameters": self.kinematic_estimator.joint_parameters
       }
       return results




def compute_joint_info_all_types(all_points_history):
   """
   Main entry for joint type estimation:
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




def compute_joint_info_all_types_improved(all_points_history):
   """
   改进的关节类型估计主函数
   使用多层递归估计框架
   修改: 使用所有点作为特征
   """
   if all_points_history.shape[0] < 2:
       ret = {
           "prismatic": {"axis": np.array([0., 0., 0.]), "motion_limit": (0., 0.)},
           "revolute": {"axis": np.array([0., 0., 0.]), "origin": np.array([0., 0., 0.]), "motion_limit": (0., 0.)}
       }
       return ret, "Unknown", None


   # 创建并初始化多层估计系统 - 移除 num_features 参数
   perception_system = MultiLevelArticulatedPerception()
   perception_system.initialize(all_points_history[0])


   # 逐帧处理点云序列
   for t in range(1, all_points_history.shape[0]):
       dt = 1.0  # 假设单位时间步长
       joint_types, joint_parameters = perception_system.process_frame(all_points_history[t], dt)


   # 获取最终结果
   results = perception_system.get_joint_estimation_results()


   # 找出最可能的关节类型
   joint_probs = {
       "prismatic": 0.0,
       "revolute": 0.0,
       "rigid": 0.0,
       "disconnected": 0.0
   }
   best_type = "Unknown"
   best_prob = 0.0


   # 只关注第一对刚体的关节（简化）
   if len(perception_system.rigid_body_pairs) > 0:
       first_pair = perception_system.rigid_body_pairs[0]
       best_type_pair = results["joint_types"].get(first_pair, "disconnected")


       # 从字符串转换为标题格式
       best_type = best_type_pair.capitalize()


       # 设置最高概率
       joint_probs[best_type_pair] = 0.8
       best_prob = 0.8


   # 提取关节参数
   joint_params = {}
   if best_type.lower() in ["prismatic", "revolute"]:
       if len(perception_system.rigid_body_pairs) > 0:
           first_pair = perception_system.rigid_body_pairs[0]
           if first_pair in results["joint_parameters"]:
               params = results["joint_parameters"][first_pair]
               joint_type = best_type.lower()


               if joint_type == "prismatic":
                   joint_params["prismatic"] = {
                       "axis": params.get("axis", np.array([0., 0., 0.])),
                       "origin": params.get("origin", np.array([0., 0., 0.])),
                       "motion_limit": params.get("motion_limit", (0., 0.))
                   }
               elif joint_type == "revolute":
                   joint_params["revolute"] = {
                       "axis": params.get("axis", np.array([0., 0., 0.])),
                       "origin": params.get("origin", np.array([0., 0., 0.])),
                       "motion_limit": params.get("motion_limit", (0., 0.))
                   }


   # 如果没有找到有效关节类型，添加默认值
   if "prismatic" not in joint_params:
       joint_params["prismatic"] = {
           "axis": np.array([0., 0., 0.]),
           "origin": np.array([0., 0., 0.]),
           "motion_limit": (0., 0.)
       }
   if "revolute" not in joint_params:
       joint_params["revolute"] = {
           "axis": np.array([0., 0., 0.]),
           "origin": np.array([0., 0., 0.]),
           "motion_limit": (0., 0.)
       }


   # 准备信息字典
   info_dict = {
       "joint_probs": joint_probs,
       "basic_score_avg": {
           "col_mean": 0.0,
           "cop_mean": 0.0,
           "rad_mean": 0.0,
           "zp_mean": 0.0
       }
   }


   return joint_params, best_type, info_dict




################################################################################
#                          Polyscope + DPG Integration                         #
################################################################################
ps.init()


# 1) Load Real Drawer Data
file_path_drawer = "/home/rui/projects/kitchen_drawer/exp2_pro_kvil/pro_kvil/data/demo1/obj/xyz_filtered.pt"
try:
   real_data_dict_drawer = torch.load(file_path_drawer, weights_only=False)
   drawer_points_tensor = real_data_dict_drawer.data["drawer"]
   real_drawer_data_np = np.array(drawer_points_tensor)  # shape (T, N, 3)
   ps_real_drawer = ps.register_point_cloud("Real Drawer Data", real_drawer_data_np[0], enabled=False)
except Exception as e:
   print(f"Could not load drawer data: {e}")
   real_drawer_data_np = np.zeros((1, 100, 3))
   ps_real_drawer = ps.register_point_cloud("Real Drawer Data", real_drawer_data_np[0], enabled=False)


# 2) Load Real Dishwasher Data
file_path1 = "/home/rui/projects/dishwasher_door/pro_kvil/data/demo_01/obj/xyz_filtered.pt"
try:
   real_data_dict_dishwasher = torch.load(file_path1, weights_only=False)
   dishwasher_points_tensor = real_data_dict_dishwasher.data["dishwasher"]  # key="dishwasher"
   real_dishwasher_data_np = np.array(dishwasher_points_tensor)  # shape (T, N, 3)
   ps_dishwasher = ps.register_point_cloud("Real Dishwasher Data", real_dishwasher_data_np[0], enabled=False)
except Exception as e:
   print(f"Could not load dishwasher data: {e}")
   real_dishwasher_data_np = np.zeros((1, 100, 3))
   ps_dishwasher = ps.register_point_cloud("Real Dishwasher Data", real_dishwasher_data_np[0], enabled=False)


# 3) Load Real Fridge Data
file_path2 = "/home/rui/projects/fridge_door/pro_kvil/data/demo_01/obj/xyz_filtered.pt"
try:
   real_data_dict_fridge = torch.load(file_path2, weights_only=False)
   fridge_points_tensor = real_data_dict_fridge.data["fridge"]  # key="fridge"
   real_fridge_data_np = np.array(fridge_points_tensor)  # shape (T, N, 3)
   ps_fridge = ps.register_point_cloud("Real Fridge Data", real_fridge_data_np[0], enabled=False)
except Exception as e:
   print(f"Could not load fridge data: {e}")
   real_fridge_data_np = np.zeros((1, 100, 3))
   ps_fridge = ps.register_point_cloud("Real Fridge Data", real_fridge_data_np[0], enabled=False)


################################################################################
# Now define all other modes (Prismatic, Revolute, etc.)                       #
################################################################################


modes = [
   "Prismatic Door", "Prismatic Door 2", "Prismatic Door 3",
   "Revolute Door", "Revolute Door 2", "Revolute Door 3",
   "Planar Mouse", "Planar Mouse 2", "Planar Mouse 3",
   "Ball Joint", "Ball Joint 2", "Ball Joint 3",
   "Screw Joint", "Screw Joint 2", "Screw Joint 3",
   "Real Drawer Data",
   "Real Dishwasher Data",
   "Real Fridge Data"
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
       "revolute": [],
       "rigid": [],
       "disconnected": []
   } for m in modes
}
frame_count_per_mode = {m: 0 for m in modes}


# ===================== Synthetic Data Setup ===========================
output_dir = "exported_pointclouds"
os.makedirs(output_dir, exist_ok=True)


noise_sigma = 0.000


# Synthetic prismatic door data
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


# Synthetic revolute data
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


# Synthetic planar mouse data
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


# Synthetic ball joint data
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


# Synthetic screw data
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


################################################################################
#                            Utility + Polyscope                                #
################################################################################




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






def show_joint_visual(joint_type, joint_params,
                     name_axis="Joint Axis",
                     name_origin="Joint Origin",
                     length_scale=1.0,
                     radius_axis=0.02,
                     radius_origin=0.03):
   """
   Visualize the joint parameters in Polyscope.


   Args:
       joint_type: "revolute", "prismatic", etc.
       joint_params: Dictionary containing joint parameters
       name_axis: Name for the axis visualization
       name_origin: Name for the origin visualization
       length_scale: Scale factor for axis length visualization
       radius_axis: Radius for the axis curve
       radius_origin: Radius for the origin point
   """
   # First remove any existing visualizations
   remove_joint_visual()
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   eps = 1e-6


   def torch_normalize(vec_t):
       norm_ = torch.norm(vec_t)
       return vec_t if norm_ < eps else vec_t / norm_


   # First remove any existing visualizations
   for n in [name_axis, name_origin, "Prismatic Axis"]:
       if ps.has_curve_network(n):
           ps.remove_curve_network(n)


   if joint_type == "prismatic":
       axis_np = joint_params.get("axis", np.array([1., 0., 0.]))
       origin_np = joint_params.get("origin", np.array([0., 0., 0.]))


       # Normalize axis and scale it
       axis_norm = np.linalg.norm(axis_np)
       if axis_norm > 1e-6:  # Avoid division by zero
           axis_dir = axis_np / axis_norm
       else:
           axis_dir = np.array([1., 0., 0.])  # Default if axis is zero


       # Create a segment for visualization
       seg_nodes = np.vstack([
           origin_np,
           origin_np + axis_dir * length_scale
       ])
       seg_edges = np.array([[0, 1]])


       net = ps.register_curve_network("Prismatic Axis", seg_nodes, seg_edges)
       net.set_radius(radius_axis)
       net.set_color((0., 1., 1.))  # Cyan color for prismatic


   elif joint_type == "revolute":
       axis_np = joint_params.get("axis", np.array([0., 1., 0.]))
       origin_np = joint_params.get("origin", np.array([0., 0., 0.]))


       # Normalize axis and scale it
       axis_norm = np.linalg.norm(axis_np)
       if axis_norm > 1e-6:  # Avoid division by zero
           axis_dir = axis_np / axis_norm
       else:
           axis_dir = np.array([0., 1., 0.])  # Default if axis is zero


       # Create line segment along the axis passing through the origin
       seg_nodes = np.vstack([
           origin_np - axis_dir * length_scale,
           origin_np + axis_dir * length_scale
       ])
       seg_edges = np.array([[0, 1]])


       # 1) Draw the axis
       axis_net = ps.register_curve_network(name_axis, seg_nodes, seg_edges)
       axis_net.set_radius(radius_axis)
       axis_net.set_color((1., 1., 0.))  # Yellow color for revolute axis


       # 2) Draw a point at the origin
       seg_nodes2 = np.vstack([
           origin_np,
           origin_np + axis_dir * 1e-4  # Tiny segment to represent a point
       ])
       seg_edges2 = np.array([[0, 1]])
       ori_net = ps.register_curve_network(name_origin, seg_nodes2, seg_edges2)
       ori_net.set_radius(radius_origin)
       ori_net.set_color((1., 0., 0.))  # Red color for revolute origin




   elif joint_type == "planar":
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

position_error_profile = {m: [] for m in modes}
angular_error_profile = {m: [] for m in modes}
def compute_error_for_mode(mode, param_dict):
    """
    计算位置误差和角度误差。
    返回 (position_error, angular_error) 元组。
    位置误差: 估计轴与groundtruth轴之间的最近距离
    角度误差: 估计轴与groundtruth轴之间的角度差值
    """
    if mode in ("Real Drawer Data", "Real Dishwasher Data", "Real Fridge Data"):
        # 真实数据集，返回0.0
        return (0.0, 0.0)

    # 处理合成数据集
    # Prismatic Doors
    if mode == "Prismatic Door":
        prismatic_door_gt_axis = np.array([1., 0., 0.])
        info = param_dict["prismatic"]
        est_axis = info["axis"]
        est_origin = info["origin"]

        # 角度误差
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, prismatic_door_gt_axis)
        ang_err = abs(1.0 - abs(dotv))

        # 位置误差 - 轴之间的最近距离
        gt_origin = np.array([0., 0., 0.])  # Ground truth origin
        d12 = gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, prismatic_door_gt_axis)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:  # 轴平行
            pos_err = np.linalg.norm(np.cross(d12, prismatic_door_gt_axis))
        else:
            n_ = cross_ / cross_norm
            pos_err = abs(np.dot(d12, n_))

        return (pos_err, ang_err)

    elif mode == "Prismatic Door 2":
        prismatic_door_gt_axis_2 = np.array([0., 1., 0.])
        info = param_dict["prismatic"]
        est_axis = info["axis"]
        est_origin = info["origin"]

        # 角度误差
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, prismatic_door_gt_axis_2)
        ang_err = abs(1.0 - abs(dotv))

        # 位置误差
        gt_origin = np.array([1.5, 0., 0.])  # Ground truth origin
        d12 = gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, prismatic_door_gt_axis_2)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            pos_err = np.linalg.norm(np.cross(d12, prismatic_door_gt_axis_2))
        else:
            n_ = cross_ / cross_norm
            pos_err = abs(np.dot(d12, n_))

        return (pos_err, ang_err)

    elif mode == "Prismatic Door 3":
        prismatic_door_gt_axis_3 = np.array([0., 0., 1.])
        info = param_dict["prismatic"]
        est_axis = info["axis"]
        est_origin = info["origin"]

        # 角度误差
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)
        dotv = np.dot(est_axis_norm, prismatic_door_gt_axis_3)
        ang_err = abs(1.0 - abs(dotv))

        # 位置误差
        gt_origin = np.array([-1., 1., 0.])  # Ground truth origin
        d12 = gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, prismatic_door_gt_axis_3)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            pos_err = np.linalg.norm(np.cross(d12, prismatic_door_gt_axis_3))
        else:
            n_ = cross_ / cross_norm
            pos_err = abs(np.dot(d12, n_))

        return (pos_err, ang_err)

    # Revolute Doors
    elif mode == "Revolute Door":
        info = param_dict["revolute"]
        est_axis = info["axis"]
        est_origin = info["origin"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)

        # 角度误差
        dotv = np.dot(est_axis_norm, revolute_door_gt_axis_norm)
        ang_err = abs(1.0 - abs(dotv))

        # 位置误差 - 两条轴之间的最短距离
        d12 = revolute_door_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, revolute_door_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:  # 轴平行
            pos_err = np.linalg.norm(np.cross(d12, revolute_door_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            pos_err = abs(np.dot(d12, n_))
        pos_err = pos_err / revolute_door_max_dist  # 归一化

        return (pos_err, ang_err)

    elif mode == "Revolute Door 2":
        info = param_dict["revolute"]
        est_axis = info["axis"]
        est_origin = info["origin"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)

        # 角度误差
        dotv = np.dot(est_axis_norm, revolute_door_2_gt_axis_norm)
        ang_err = abs(1.0 - abs(dotv))

        # 位置误差
        d12 = revolute_door_2_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, revolute_door_2_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            pos_err = np.linalg.norm(np.cross(d12, revolute_door_2_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            pos_err = abs(np.dot(d12, n_))
        pos_err = pos_err / revolute_door_2_max_dist

        return (pos_err, ang_err)

    elif mode == "Revolute Door 3":
        info = param_dict["revolute"]
        est_axis = info["axis"]
        est_origin = info["origin"]
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)

        # 角度误差
        dotv = np.dot(est_axis_norm, revolute_door_3_gt_axis_norm)
        ang_err = abs(1.0 - abs(dotv))

        # 位置误差
        d12 = revolute_door_3_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, revolute_door_3_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            pos_err = np.linalg.norm(np.cross(d12, revolute_door_3_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            pos_err = abs(np.dot(d12, n_))
        pos_err = pos_err / revolute_door_3_max_dist

        return (pos_err, ang_err)

    # Planar Mice
    elif mode == "Planar Mouse":
        info = param_dict["revolute"] if "revolute" in param_dict else param_dict["prismatic"]
        est_normal = info["axis"]
        est_normal_norm = est_normal / (np.linalg.norm(est_normal) + 1e-9)

        # 对于平面关节，角度误差是法向量之间的角度
        dotv = np.dot(est_normal_norm, planar_mouse_gt_normal)
        ang_err = abs(1.0 - abs(dotv))

        # 位置误差简化为从平面到原点的距离
        pos_err = abs(np.dot(est_normal_norm, np.array([0., 0., 0.])))

        return (pos_err, ang_err)

    elif mode == "Planar Mouse 2":
        info = param_dict["revolute"] if "revolute" in param_dict else param_dict["prismatic"]
        est_normal = info["axis"]
        est_normal_norm = est_normal / (np.linalg.norm(est_normal) + 1e-9)
        dotv = np.dot(est_normal_norm, planar_mouse_2_gt_normal)
        ang_err = abs(1.0 - abs(dotv))
        pos_err = abs(np.dot(est_normal_norm, np.array([1., 0., 1.])))
        return (pos_err, ang_err)

    elif mode == "Planar Mouse 3":
        info = param_dict["revolute"] if "revolute" in param_dict else param_dict["prismatic"]
        est_normal = info["axis"]
        est_normal_norm = est_normal / (np.linalg.norm(est_normal) + 1e-9)
        dotv = np.dot(est_normal_norm, planar_mouse_3_gt_normal)
        ang_err = abs(1.0 - abs(dotv))
        pos_err = abs(np.dot(est_normal_norm, np.array([-1., 0., 1.])))
        return (pos_err, ang_err)

    # Ball Joints
    elif mode == "Ball Joint":
        info = param_dict["revolute"]
        est_origin = info["origin"]

        # 球关节没有轴方向，所以角度误差为0
        ang_err = 0.0

        # 位置误差是中心点的距离
        pos_err = np.linalg.norm(est_origin - ball_joint_gt_center) / ball_joint_max_dist

        return (pos_err, ang_err)

    elif mode == "Ball Joint 2":
        info = param_dict["revolute"]
        est_origin = info["origin"]
        ang_err = 0.0
        pos_err = np.linalg.norm(est_origin - ball_joint_2_gt_center) / ball_joint_2_max_dist
        return (pos_err, ang_err)

    elif mode == "Ball Joint 3":
        info = param_dict["revolute"]
        est_origin = info["origin"]
        ang_err = 0.0
        pos_err = np.linalg.norm(est_origin - ball_joint_3_gt_center) / ball_joint_3_max_dist
        return (pos_err, ang_err)

    # Screw Joints
    elif mode == "Screw Joint":
        info = param_dict["revolute"] if "revolute" in param_dict else param_dict["prismatic"]
        est_axis = info["axis"]
        est_origin = info.get("origin", np.array([0., 0., 0.]))
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)

        # 角度误差
        dotv = np.dot(est_axis_norm, screw_gt_axis_norm)
        ang_err = abs(1.0 - abs(dotv))

        # 位置误差
        d12 = screw_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, screw_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            pos_err = np.linalg.norm(np.cross(d12, screw_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            pos_err = abs(np.dot(d12, n_))
        pos_err = pos_err / screw_joint_max_dist

        return (pos_err, ang_err)

    elif mode == "Screw Joint 2":
        info = param_dict["revolute"] if "revolute" in param_dict else param_dict["prismatic"]
        est_axis = info["axis"]
        est_origin = info.get("origin", np.array([0., 0., 0.]))
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)

        # 角度误差
        dotv = np.dot(est_axis_norm, screw_2_gt_axis_norm)
        ang_err = abs(1.0 - abs(dotv))

        # 位置误差
        d12 = screw_2_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, screw_2_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            pos_err = np.linalg.norm(np.cross(d12, screw_2_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            pos_err = abs(np.dot(d12, n_))
        pos_err = pos_err / screw_joint_2_max_dist

        return (pos_err, ang_err)

    elif mode == "Screw Joint 3":
        info = param_dict["revolute"] if "revolute" in param_dict else param_dict["prismatic"]
        est_axis = info["axis"]
        est_origin = info.get("origin", np.array([0., 0., 0.]))
        est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)

        # 角度误差
        dotv = np.dot(est_axis_norm, screw_3_gt_axis_norm)
        ang_err = abs(1.0 - abs(dotv))

        # 位置误差
        d12 = screw_3_gt_origin - est_origin
        cross_ = np.cross(est_axis_norm, screw_3_gt_axis_norm)
        cross_norm = np.linalg.norm(cross_)
        if cross_norm < 1e-9:
            pos_err = np.linalg.norm(np.cross(d12, screw_3_gt_axis_norm))
        else:
            n_ = cross_ / cross_norm
            pos_err = abs(np.dot(d12, n_))
        pos_err = pos_err / screw_joint_3_max_dist

        return (pos_err, ang_err)

    return (0.0, 0.0)


# Count total frames for each mode
TOTAL_FRAMES_PER_MODE = {}
for m in modes:
   if m == "Real Drawer Data":
       TOTAL_FRAMES_PER_MODE[m] = real_drawer_data_np.shape[0]
   elif m == "Real Dishwasher Data":
       TOTAL_FRAMES_PER_MODE[m] = real_dishwasher_data_np.shape[0]
   elif m == "Real Fridge Data":
       TOTAL_FRAMES_PER_MODE[m] = real_fridge_data_np.shape[0]
   else:
       # By default, let's use 120 frames for synthetic modes
       TOTAL_FRAMES_PER_MODE[m] = 50




def restore_original_shape(mode):
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
   global ps_real_drawer, ps_dishwasher, ps_fridge


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
   ps_dishwasher.set_enabled(mode == "Real Dishwasher Data")
   ps_fridge.set_enabled(mode == "Real Fridge Data")


   if mode == "Real Drawer Data":
       if real_drawer_data_np.shape[0] > 0:
           ps_real_drawer.update_point_positions(real_drawer_data_np[0])
   elif mode == "Real Dishwasher Data":
       if real_dishwasher_data_np.shape[0] > 0:
           ps_dishwasher.update_point_positions(real_dishwasher_data_np[0])
   elif mode == "Real Fridge Data":
       if real_fridge_data_np.shape[0] > 0:
           ps_fridge.update_point_positions(real_fridge_data_np[0])


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


   clear_data_for_mode(mode)




running = False
current_mode = "Prismatic Door"
previous_mode = None
current_best_joint = "Unknown"
current_scores_info = None


screw_pitch = 0.5
screw_pitch_2 = 0.8
screw_pitch_3 = 0.6
def update_motion_and_store(mode):
   """
   Advance to the next frame of motion for the current mode, store the frame, and update Polyscope.
   """
   global frame_count_per_mode
   global prismatic_door_points, prismatic_door_points_2, prismatic_door_points_3
   global revolute_door_points, revolute_door_points_2, revolute_door_points_3
   global ps_prismatic_door, ps_prismatic_door_2, ps_prismatic_door_3
   global ps_revolute_door, ps_revolute_door_2, ps_revolute_door_3
   global ps_real_drawer, ps_dishwasher, ps_fridge
   global real_drawer_data_np, real_dishwasher_data_np, real_fridge_data_np
   global noise_sigma
   global mouse_points, mouse_points_3, mouse_points_2, joint_points, joint_points_2, joint_points_3, cap_points, cap_points_2, cap_points_3
   fidx = frame_count_per_mode[mode]
   limit = TOTAL_FRAMES_PER_MODE[mode]
   if fidx >= limit:
       return


   prev_points = None
   current_points = None


   # Synthetic modes:
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
       current_points = prismatic_door_points




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
       current_points = prismatic_door_points_2




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
       current_points = prismatic_door_points_3




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
       current_points = revolute_door_points




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
       current_points = revolute_door_points_2




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
       current_points = revolute_door_points_3


   # Real data modes:
   elif mode == "Real Drawer Data":
       if fidx < real_drawer_data_np.shape[0]:
           prev_positions = real_drawer_data_np[fidx - 1] if fidx > 0 else None
           current_positions = real_drawer_data_np[fidx]
           ps_real_drawer.update_point_positions(current_positions)
           store_point_cloud(current_positions, mode)
           highlight_max_points(ps_real_drawer, current_positions, prev_positions)
           current_points = current_positions






   elif mode == "Planar Mouse":
       prev_points = mouse_points.copy()
       # Demo motion for Planar Mouse
       if fidx < 20:
           alpha = fidx / 19.0
           tx = -1.0 + alpha * (0.0 - (-1.0))
           tz = 1.0 + alpha * (0.0 - 1.0)
           ry = 0.0
           mouse_points = original_mouse_points.copy()
           mouse_points += np.array([tx, 0., tz])
           mouse_points = rotate_points_y(mouse_points, ry, [0., 0., 0.])
       elif fidx < 30:
           alpha = (fidx - 20) / 9.0
           ry = math.radians(40.0) * alpha
           mouse_points = original_mouse_points.copy()
           mouse_points = rotate_points_y(mouse_points, ry, [0., 0., 0.])
       else:
           alpha = (fidx - 30) / 9.0
           tx = alpha * 1.0
           tz = alpha * (-1.0)
           ry = math.radians(40.0)
           mouse_points = original_mouse_points.copy()
           mouse_points += np.array([tx, 0., tz])
           mouse_points = rotate_points_y(mouse_points, ry, [0., 0., 0.])
       if noise_sigma > 1e-6:
           mouse_points += np.random.normal(0, noise_sigma, mouse_points.shape)
       ps_mouse.update_point_positions(mouse_points)
       store_point_cloud(mouse_points, mode)
       highlight_max_points(ps_mouse, mouse_points, prev_points)
       current_points = mouse_points


   elif mode == "Planar Mouse 2":
       prev_points = mouse_points_2.copy()
       # Another custom motion
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
       current_points = mouse_points_2


   elif mode == "Planar Mouse 3":
       prev_points = mouse_points_3.copy()
       # Another custom motion
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
       current_points = mouse_points_3


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
       current_points = joint_points


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
       current_points = joint_points_2


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
       current_points = joint_points_3


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
       current_points = cap_points


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
       current_points = cap_points_2


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
       current_points = cap_points_3


   elif mode == "Real Dishwasher Data":
       if fidx < real_dishwasher_data_np.shape[0]:
           prev_positions = real_dishwasher_data_np[fidx - 1] if fidx > 0 else None
           current_positions = real_dishwasher_data_np[fidx]
           ps_dishwasher.update_point_positions(current_positions)
           store_point_cloud(current_positions, mode)
           highlight_max_points(ps_dishwasher, current_positions, prev_positions)
           current_points = current_positions




   elif mode == "Real Fridge Data":
       if fidx < real_fridge_data_np.shape[0]:
           prev_positions = real_fridge_data_np[fidx - 1] if fidx > 0 else None
           current_positions = real_fridge_data_np[fidx]
           ps_fridge.update_point_positions(current_positions)
           store_point_cloud(current_positions, mode)
           highlight_max_points(ps_fridge, current_positions, prev_positions)
           current_points = current_positions


   frame_count_per_mode[mode] += 1


def callback():
    global current_mode, previous_mode
    global current_best_joint, current_scores_info
    global noise_sigma
    global running
    # Global variables to set at the top of your file

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

    if psim.TreeNodeEx("Noise Settings", flags=psim.ImGuiTreeNodeFlags_DefaultOpen):
        psim.Columns(2, "mycolumns", False)
        psim.SetColumnWidth(0, 230)
        changed_noise, new_noise_sigma = psim.InputFloat("Noise Sigma", noise_sigma, 0.001)
        if changed_noise:
            noise_sigma = max(0.0, new_noise_sigma)
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
        param_dict, best_type, scores_info = compute_joint_info_all_types(all_points_history)
        current_best_joint = best_type
        current_scores_info = scores_info

        if scores_info is not None:
            joint_probs = scores_info["joint_probs"]

            # Add probabilities to plot data
            for jt_name in joint_prob_profile[current_mode]:
                joint_prob_profile[current_mode][jt_name].append(joint_probs[jt_name])

            # Display basic scores
            psim.TextUnformatted("=== Joint Probability ===")
            psim.TextUnformatted(f"Prismatic = {joint_probs['prismatic']:.4f}")
            psim.TextUnformatted(f"Revolute  = {joint_probs['revolute']:.4f}")
            psim.TextUnformatted(f"Rigid     = {joint_probs['rigid']:.4f}")
            psim.TextUnformatted(f"Disconnected = {joint_probs['disconnected']:.4f}")
            psim.TextUnformatted(f"Best Joint Type: {best_type}")

            # 计算分离的位置和角度误差
            pos_err, ang_err = compute_error_for_mode(current_mode, param_dict)
            position_error_profile[current_mode].append(pos_err)
            angular_error_profile[current_mode].append(ang_err)

            # 保持原有的error_profile用于向后兼容
            error_profile[current_mode].append((pos_err + ang_err) / 2.0)  # 使用均值作为总体误差

            # 显示两种误差
            psim.TextUnformatted(f"Position Error => {pos_err:.4f}")
            psim.TextUnformatted(f"Angular Error => {ang_err:.4f} rad")

        if best_type.lower() in param_dict:
            jinfo = param_dict[best_type.lower()]
            show_joint_visual(best_type.lower(), jinfo)

            # Display joint parameters based on joint type
            if best_type.lower() == "prismatic":
                a_ = jinfo["axis"]
                lim = jinfo["motion_limit"]
                psim.TextUnformatted(f"  Axis=({a_[0]:.2f}, {a_[1]:.2f}, {a_[2]:.2f})")
                psim.TextUnformatted(f"  MotionLimit=({lim[0]:.2f}, {lim[1]:.2f})")
            elif best_type.lower() == "revolute":
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
        # 更新位置误差图表
        pos_err_data = position_error_profile[mode]
        tag_line = f"{mode}_pos_err_series"
        if len(pos_err_data) > 0 and dpg.does_item_exist(tag_line):
            dpg.set_value(tag_line, [list(range(len(pos_err_data))), pos_err_data])

        # 更新角度误差图表
        ang_err_data = angular_error_profile[mode]
        tag_line = f"{mode}_ang_err_series"
        if len(ang_err_data) > 0 and dpg.does_item_exist(tag_line):
            dpg.set_value(tag_line, [list(range(len(ang_err_data))), ang_err_data])

        # 更新原有误差图
        err_data = error_profile[mode]
        tag_line = f"{mode}_err_series"
        if len(err_data) > 0 and dpg.does_item_exist(tag_line):
            dpg.set_value(tag_line, [list(range(len(err_data))), err_data])

        # 更新joint probability plots
        for jt_name in ["prismatic", "revolute", "rigid", "disconnected"]:
            p_ = joint_prob_profile[mode][jt_name]
            tag_line = f"{mode}_prob_{jt_name}_series"
            if len(p_) > 0 and dpg.does_item_exist(tag_line):
                dpg.set_value(tag_line, [list(range(len(p_))), p_])

def stop_callback(sender, app_data):
   global stop_refresh
   stop_refresh = not stop_refresh
   if stop_refresh:
       dpg.set_item_label(sender, "Resume Refresh")
   else:
       dpg.set_item_label(sender, "Pause Refresh")


def callback():
    global current_mode, previous_mode
    global current_best_joint, current_scores_info
    global noise_sigma
    global running
    # Global variables to set at the top of your file

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

    if psim.TreeNodeEx("Noise Settings", flags=psim.ImGuiTreeNodeFlags_DefaultOpen):
        psim.Columns(2, "mycolumns", False)
        psim.SetColumnWidth(0, 230)
        changed_noise, new_noise_sigma = psim.InputFloat("Noise Sigma", noise_sigma, 0.001)
        if changed_noise:
            noise_sigma = max(0.0, new_noise_sigma)
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
        param_dict, best_type, scores_info = compute_joint_info_all_types(all_points_history)
        current_best_joint = best_type
        current_scores_info = scores_info

        if scores_info is not None:
            joint_probs = scores_info["joint_probs"]

            # Add probabilities to plot data
            for jt_name in joint_prob_profile[current_mode]:
                joint_prob_profile[current_mode][jt_name].append(joint_probs[jt_name])

            # Display basic scores
            psim.TextUnformatted("=== Joint Probability ===")
            psim.TextUnformatted(f"Prismatic = {joint_probs['prismatic']:.4f}")
            psim.TextUnformatted(f"Revolute  = {joint_probs['revolute']:.4f}")
            psim.TextUnformatted(f"Rigid     = {joint_probs['rigid']:.4f}")
            psim.TextUnformatted(f"Disconnected = {joint_probs['disconnected']:.4f}")
            psim.TextUnformatted(f"Best Joint Type: {best_type}")

            # 计算分离的位置和角度误差
            pos_err, ang_err = compute_error_for_mode(current_mode, param_dict)
            position_error_profile[current_mode].append(pos_err)
            angular_error_profile[current_mode].append(ang_err)

            # 保持原有的error_profile用于向后兼容
            error_profile[current_mode].append((pos_err + ang_err) / 2.0)  # 使用均值作为总体误差

            # 显示两种误差
            psim.TextUnformatted(f"Position Error => {pos_err:.4f}")
            psim.TextUnformatted(f"Angular Error => {ang_err:.4f} rad")

        if best_type.lower() in param_dict:
            jinfo = param_dict[best_type.lower()]
            show_joint_visual(best_type.lower(), jinfo)

            # Display joint parameters based on joint type
            if best_type.lower() == "prismatic":
                a_ = jinfo["axis"]
                lim = jinfo["motion_limit"]
                psim.TextUnformatted(f"  Axis=({a_[0]:.2f}, {a_[1]:.2f}, {a_[2]:.2f})")
                psim.TextUnformatted(f"  MotionLimit=({lim[0]:.2f}, {lim[1]:.2f})")
            elif best_type.lower() == "revolute":
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


   # For real data modes, do nothing (or no GT lines).
   if not enable or mode in ("Real Drawer Data", "Real Dishwasher Data", "Real Fridge Data"):
       return


   # Show synthetic ground-truth lines/axes based on the mode
   if mode == "Prismatic Door":
       axis_np = np.array([1.0, 0., 0.])
       seg_nodes = np.array([[0, 0, 0], axis_np])
       seg_edges = np.array([[0, 1]])
       net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
       net.set_radius(0.02)
       net.set_color((1., 0., 0.))


   elif mode == "Prismatic Door 2":
       axis_np = np.array([0., 1., 0.])
       seg_nodes = np.array([[0, 0, 0], axis_np])
       seg_edges = np.array([[0, 1]])
       net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
       net.set_radius(0.02)
       net.set_color((0., 1., 0.))
   elif mode == "Prismatic Door 3":
       axis_np = np.array([0., 0., 1.])
       seg_nodes = np.array([[0, 0, 0], axis_np])
       seg_edges = np.array([[0, 1]])
       net = ps.register_curve_network(gt_name_axis, seg_nodes, seg_edges)
       net.set_radius(0.02)
       net.set_color((0., 0., 1.))


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


   elif mode == "Ball Joint":
       c_ = np.array([0.0, 0.0, 0.0])
       pc = ps.register_point_cloud(gt_name_center, c_.reshape(1, 3))
       pc.set_radius(0.05)
       pc.set_color((1., 0., 1.))


   elif mode == "Ball Joint 2":
       c_ = np.array([1., 0., 0.])
       pc = ps.register_point_cloud(gt_name_center, c_.reshape(1, 3))
       pc.set_radius(0.05)
       pc.set_color((1., 0., 1.))


   elif mode == "Ball Joint 3":
       c_ = np.array([1., 1., 0.])
       pc = ps.register_point_cloud(gt_name_center, c_.reshape(1, 3))
       pc.set_radius(0.05)
       pc.set_color((1., 0., 1.))


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

def clear_plots_callback():
    global velocity_profile, angular_velocity_profile
    global col_score_profile, cop_score_profile, rad_score_profile, zp_score_profile
    global error_profile, joint_prob_profile
    global position_error_profile, angular_error_profile  # 新增

    for m in modes:
        velocity_profile[m].clear()
        angular_velocity_profile[m].clear()
        col_score_profile[m].clear()
        cop_score_profile[m].clear()
        rad_score_profile[m].clear()
        zp_score_profile[m].clear()
        error_profile[m].clear()
        position_error_profile[m].clear()  # 新增
        angular_error_profile[m].clear()   # 新增
        for jt_name in joint_prob_profile[m]:
            joint_prob_profile[m][jt_name].clear()

    refresh_dearpygui_plots()

def dearpygui_thread_main():
    dpg.create_context()

    def on_mode_change(sender, app_data):
        selected_modes = []
        for mode in modes:
            if dpg.get_value(f"checkbox_{mode}"):
                selected_modes.append(mode)

        for mode in modes:
            for suffix in ["pos_err_series", "ang_err_series",  # 新增
                           "err_series", "prob_prismatic_series", "prob_revolute_series",
                           "prob_rigid_series", "prob_disconnected_series"]:
                tag = f"{mode}_{suffix}"
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, show=(mode in selected_modes))

    with dpg.window(label="Joint Estimation Plots", width=1200, height=800):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Pause Refresh", callback=stop_callback)
            dpg.add_button(label="Clear Plots", callback=clear_plots_callback)

            with dpg.collapsing_header(label="Select Modes to Display", default_open=False, tag="mode_selector_group"):
                for mode in modes:
                    dpg.add_checkbox(label=mode, default_value=(mode == modes[0]),
                                     callback=on_mode_change, tag=f"checkbox_{mode}")

        with dpg.tab_bar():
            # 原有的错误标签页 - 保留用于兼容
            with dpg.tab(label="Joint parameter error"):
                with dpg.plot(label="Combined Joint parameter error", height=300, width=800):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time frames", tag="x_axis_err")
                    y_axis_err = dpg.add_plot_axis(dpg.mvYAxis, label="Joint parameter error", tag="y_axis_err")
                    for mode in modes:
                        tag_line = f"{mode}_err_series"
                        dpg.add_line_series([], [], label=mode, parent=y_axis_err, tag=tag_line,
                                            show=(mode == modes[0]))

            # 新增错误类型标签页
            with dpg.tab(label="Separated Joint Errors"):
                with dpg.group(horizontal=True):
                    # 位置误差图表
                    with dpg.plot(label="Position Error", height=300, width=400):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Time frames", tag="x_axis_pos_err")
                        y_axis_pos_err = dpg.add_plot_axis(dpg.mvYAxis, label="Position Error", tag="y_axis_pos_err")
                        for mode in modes:
                            tag_line = f"{mode}_pos_err_series"
                            dpg.add_line_series([], [], label=mode, parent=y_axis_pos_err, tag=tag_line,
                                                show=(mode == modes[0]))

                    # 角度误差图表
                    with dpg.plot(label="Angular Error", height=300, width=400):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="Time frames", tag="x_axis_ang_err")
                        y_axis_ang_err = dpg.add_plot_axis(dpg.mvYAxis, label="Angular Error (rad)",
                                                           tag="y_axis_ang_err")
                        for mode in modes:
                            tag_line = f"{mode}_ang_err_series"
                            dpg.add_line_series([], [], label=mode, parent=y_axis_ang_err, tag=tag_line,
                                                show=(mode == modes[0]))

            with dpg.tab(label="Joint Probabilities"):
                with dpg.group(horizontal=True):
                    with dpg.group():
                        for label, tag_y, suffix in [("Prismatic", "y_axis_prob_pm", "prob_prismatic_series"),
                                                     ("Revolute", "y_axis_prob_rv", "prob_revolute_series")]:
                            with dpg.plot(label=f"{label} Probability in different time frames", height=250, width=400):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="Time frames")
                                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=f"{label} Probability", tag=tag_y)
                                dpg.set_axis_limits(tag_y, -0.1, 1.1)
                                for mode in modes:
                                    tag_line = f"{mode}_{suffix}"
                                    dpg.add_line_series([], [], label=mode, parent=y_axis, tag=tag_line,
                                                        show=(mode == modes[0]))

                    with dpg.group():
                        for label, tag_y, suffix in [("Rigid", "y_axis_prob_rg", "prob_rigid_series"),
                                                     ("Disconnected", "y_axis_prob_dc", "prob_disconnected_series")]:
                            with dpg.plot(label=f"{label} Probability in different time frames", height=250, width=400):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="Time frames")
                                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=f"{label} Probability", tag=tag_y)
                                dpg.set_axis_limits(tag_y, -0.1, 1.1)
                                for mode in modes:
                                    tag_line = f"{mode}_{suffix}"
                                    dpg.add_line_series([], [], label=mode, parent=y_axis, tag=tag_line,
                                                        show=(mode == modes[0]))

    dpg.create_viewport(title='Joint Type Classification Plots', width=1250, height=900)
    dpg.setup_dearpygui()

    # Initialize state based on checkbox values
    on_mode_change(None, None)

    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        refresh_dearpygui_plots()

    dpg.destroy_context()

def main():
   th = threading.Thread(target=dearpygui_thread_main, daemon=True)
   th.start()
   ps.set_ground_plane_mode("none")
   ps.set_user_callback(callback)
   ps.show()




if __name__ == "__main__":
   main()

