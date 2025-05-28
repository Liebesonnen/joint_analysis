# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# from scipy.signal import savgol_filter
# from io import BytesIO
# import os
# from datetime import datetime
# from robot_utils.viz.polyscope import PolyscopeUtils, ps, psim, register_point_cloud, draw_frame_3d
#
#
# class Viz:
#     def __init__(self):
#         # Create Polyscope utils instance
#         self.pu = PolyscopeUtils()
#         # Initialize current time frame index and change flag
#         self.t, self.t_changed = 0, False
#         # Initialize current point index and change flag
#         self.idx_point, self.idx_point_changed = 0, False
#         # Load revolute joint point cloud data
#         self.d = np.load("./demo_data/ball.npy")
#         # Add random noise
#         # self.d += np.random.randn(*self.d.shape) * 0.01
#         # Print data shape for debugging
#         ic(self.d.shape)
#         # Get data dimensions: number of time frames, points per frame, and coordinate dimensions
#         self.T, self.N, _ = self.d.shape
#
#         # Create folder for saving images
#         self.output_dir = "visualization_output"
#         os.makedirs(self.output_dir, exist_ok=True)
#
#         # Average time step
#         self.dt_mean = 0.1
#         # Number of neighbors for SVD computation
#         self.num_neighbors = 50
#
#         # Apply Savitzky-Golay filter to smooth data
#         self.d_filter = savgol_filter(
#             x=self.d, window_length=21, polyorder=2, deriv=0, axis=0, delta=self.dt_mean
#         )
#         # Use Savitzky-Golay filter to calculate velocity (first derivative)
#         self.dv_filter = savgol_filter(
#             x=self.d, window_length=21, polyorder=2, deriv=1, axis=0, delta=self.dt_mean
#         )
#
#         # Compute angular velocity
#         self.angular_velocity_raw, self.angular_velocity_filtered = self.calculate_angular_velocity()
#
#         # Draw origin coordinate frame
#         draw_frame_3d(np.zeros(6), label="origin", scale=0.1)
#         # Register initial point cloud data and display
#         register_point_cloud(
#             "door", self.d[self.t], radius=0.01, enabled=True
#         )
#         # Reset view bounds to include point cloud
#         self.pu.reset_bbox_from_pcl_list([self.d[self.t]])
#         # Set user interaction callback function
#         ps.set_user_callback(self.callback)
#         # Show Polyscope interface
#         ps.show()
#
#     def find_neighbors(self, points, num_neighbors):
#         """Find neighbors for each point"""
#         # Calculate distance matrix
#         N = points.shape[0]
#         dist_matrix = np.zeros((N, N))
#         for i in range(N):
#             dist_matrix[i] = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))
#
#         # Find closest points (excluding self)
#         neighbors = np.zeros((N, num_neighbors), dtype=int)
#         for i in range(N):
#             indices = np.argsort(dist_matrix[i])[1:num_neighbors + 1]
#             neighbors[i] = indices
#
#         return neighbors
#
#     def compute_rotation_matrix(self, src_points, dst_points):
#         """Compute rotation matrix between two point sets using SVD"""
#         # Center points
#         src_center = np.mean(src_points, axis=0)
#         dst_center = np.mean(dst_points, axis=0)
#
#         src_centered = src_points - src_center
#         dst_centered = dst_points - dst_center
#
#         # Compute covariance matrix
#         H = np.dot(src_centered.T, dst_centered)
#
#         # SVD decomposition
#         U, _, Vt = np.linalg.svd(H)
#
#         # Construct rotation matrix
#         R = np.dot(Vt.T, U.T)
#
#         # Handle reflection case
#         if np.linalg.det(R) < 0:
#             Vt[-1, :] *= -1
#             R = np.dot(Vt.T, U.T)
#
#         return R, src_center, dst_center
#
#     def calculate_angular_velocity(self):
#         """
#         计算角速度：
#         1. 计算每个点相对于第一帧的旋转矩阵
#         2. 提取旋转角度
#         3. 使用SG滤波平滑角度序列
#         4. 计算角度的一阶导数作为角速度
#         """
#         T, N, _ = self.d_filter.shape
#
#         # 初始化数组 - 保持T-1维度
#         angular_velocity_raw = np.zeros((T - 1, N, 3))
#         angular_velocity_filtered = np.zeros((T - 1, N, 3))
#
#         # 存储每个点的旋转角度和旋转轴
#         rotation_angles = np.zeros((T, N))  # 存储角度
#         rotation_axes = np.zeros((N, 3, T))  # 存储旋转轴，对每一帧
#
#         # 对每个点处理
#         for i in range(N):
#             # 找到第一帧的邻居点
#             first_frame_points = self.d_filter[0]
#             neighbors = self.find_neighbors(first_frame_points, self.num_neighbors)
#             first_neighborhood = first_frame_points[neighbors[i]]
#
#             # 计算每一帧相对于第一帧的旋转矩阵
#             for t in range(T):
#                 current_points = self.d_filter[t]
#                 current_neighborhood = current_points[neighbors[i]]
#
#                 # 计算相对于第一帧的旋转矩阵
#                 R, _, _ = self.compute_rotation_matrix(first_neighborhood, current_neighborhood)
#
#                 # 从旋转矩阵中提取角度和轴
#                 try:
#                     # 计算旋转角度(使用矩阵的迹)
#                     cos_theta = (np.trace(R) - 1) / 2
#                     cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差
#                     theta = np.arccos(cos_theta)
#
#                     # 存储角度
#                     rotation_angles[t, i] = theta
#
#                     # 提取旋转轴
#                     if abs(theta) > 1e-6:
#                         sin_theta = np.sin(theta)
#                         if abs(sin_theta) > 1e-6:
#                             # 从反对称矩阵提取旋转轴
#                             axis = np.zeros(3)
#                             axis[0] = (R[2, 1] - R[1, 2]) / (2 * sin_theta)
#                             axis[1] = (R[0, 2] - R[2, 0]) / (2 * sin_theta)
#                             axis[2] = (R[1, 0] - R[0, 1]) / (2 * sin_theta)
#
#                             # 归一化轴向量
#                             axis = axis / np.linalg.norm(axis)
#                             rotation_axes[i, :, t] = axis
#                 except:
#                     # 处理可能的计算异常
#                     rotation_angles[t, i] = 0
#
#             # 确保轴方向一致性
#             for t in range(1, T):
#                 if np.any(rotation_axes[i, :, t]) and np.any(rotation_axes[i, :, t - 1]):
#                     if np.dot(rotation_axes[i, :, t], rotation_axes[i, :, t - 1]) < 0:
#                         rotation_axes[i, :, t] = -rotation_axes[i, :, t]
#
#             # 使用Savitzky-Golay滤波平滑角度
#             if T >= 5:
#                 # 确保窗口长度是奇数
#                 window_length = 21
#
#                 # 应用SG滤波器平滑角度
#                 try:
#                     filtered_angles = savgol_filter(
#                         rotation_angles[:, i],
#                         window_length=window_length,
#                         polyorder=2
#                     )
#
#                     # 计算角度的导数作为角速度大小
#                     angle_velocities = savgol_filter(
#                         filtered_angles,
#                         window_length=window_length,
#                         polyorder=2,
#                         deriv=1,
#                         delta=self.dt_mean
#                     )
#
#                     # 计算角速度向量 = 角速度大小 * 旋转轴方向
#                     for t in range(T - 1):
#                         # 计算当前时刻的角速度
#                         if np.any(rotation_axes[i, :, t + 1]):  # 使用下一帧的旋转轴
#                             angular_velocity_filtered[t, i] = angle_velocities[t] * rotation_axes[i, :, t + 1]
#
#                             # 原始角速度 - 直接使用相邻帧的角度差
#                             raw_vel_magnitude = (rotation_angles[t + 1, i] - rotation_angles[t, i]) / self.dt_mean
#                             angular_velocity_raw[t, i] = raw_vel_magnitude * rotation_axes[i, :, t + 1]
#
#                     # 处理第一帧角速度 - 使用第一帧的实际计算值
#                     if np.any(rotation_axes[i, :, 1]):  # 确保第二帧有有效的旋转轴
#                         first_vel_magnitude = (rotation_angles[1, i] - rotation_angles[0, i]) / self.dt_mean
#                         angular_velocity_raw[0, i] = first_vel_magnitude * rotation_axes[i, :, 1]
#                         angular_velocity_filtered[0, i] = angle_velocities[0] * rotation_axes[i, :, 1]
#
#                 except Exception as e:
#                     print(f"SG filtering failed for point {i}: {e}")
#                     # 使用简单差分计算
#                     for t in range(T - 1):
#                         if np.any(rotation_axes[i, :, t + 1]):
#                             raw_vel_magnitude = (rotation_angles[t + 1, i] - rotation_angles[t, i]) / self.dt_mean
#                             angular_velocity_raw[t, i] = raw_vel_magnitude * rotation_axes[i, :, t + 1]
#                             angular_velocity_filtered[t, i] = angular_velocity_raw[t, i]
#             else:
#                 # 序列太短，使用简单差分
#                 for t in range(T - 1):
#                     if np.any(rotation_axes[i, :, t + 1]):
#                         raw_vel_magnitude = (rotation_angles[t + 1, i] - rotation_angles[t, i]) / self.dt_mean
#                         angular_velocity_raw[t, i] = raw_vel_magnitude * rotation_axes[i, :, t + 1]
#                         angular_velocity_filtered[t, i] = angular_velocity_raw[t, i]
#
#         return angular_velocity_raw, angular_velocity_filtered
#
#
#     def plot_image(self):
#         # Create time axis array (seconds)
#         t = np.arange(self.T) * self.dt_mean
#         t_angular = np.arange(self.T - 1) * self.dt_mean
#
#         # Create a figure with three subplots
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), dpi=100)
#
#         # First subplot: position data
#         # Plot original data
#         ax1.plot(t, self.d[:, self.idx_point, 0], "--", c="red", label="Original X (m)")
#         ax1.plot(t, self.d[:, self.idx_point, 1], "--", c="green", label="Original Y (m)")
#         ax1.plot(t, self.d[:, self.idx_point, 2], "--", c="blue", label="Original Z (m)")
#
#         # Plot filtered data
#         ax1.plot(t, self.d_filter[:, self.idx_point, 0], "-", c="darkred", label="Filtered X (m)")
#         ax1.plot(t, self.d_filter[:, self.idx_point, 1], "-", c="darkgreen", label="Filtered Y (m)")
#         ax1.plot(t, self.d_filter[:, self.idx_point, 2], "-", c="darkblue", label="Filtered Z (m)")
#
#         # Set first subplot title and labels
#         ax1.set_title(f"Position Trajectory of Point #{self.idx_point}")
#         ax1.set_xlabel("Time (seconds)")
#         ax1.set_ylabel("Position (meters)")
#         ax1.set_xlim(0, 20)
#         ax1.set_ylim(-1, 10)
#         ax1.grid(True, linestyle='--', alpha=0.7)
#         ax1.legend(loc='upper right')
#
#         # Second subplot: velocity data
#         ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity (m/s)")
#         ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity (m/s)")
#         ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity (m/s)")
#
#         # Calculate velocity magnitude
#         velocity_magnitude = np.sqrt(
#             self.dv_filter[:, self.idx_point, 0] ** 2 +
#             self.dv_filter[:, self.idx_point, 1] ** 2 +
#             self.dv_filter[:, self.idx_point, 2] ** 2
#         )
#         ax2.plot(t, velocity_magnitude, "-", c="black", label="Velocity Magnitude (m/s)")
#
#         # Set second subplot title and labels
#         ax2.set_title(f"Linear Velocity of Point #{self.idx_point}")
#         ax2.set_xlabel("Time (seconds)")
#         ax2.set_ylabel("Velocity (meters/second)")
#         ax2.set_xlim(0, 20)
#         ax2.set_ylim(-1, 1)
#         ax2.grid(True, linestyle='--', alpha=0.7)
#         ax2.legend(loc='upper right')
#
#         # Third subplot: angular velocity data
#         # Plot raw angular velocity (dashed lines)
#         ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red",
#                  label="Raw ωx (rad/s)")
#         ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green",
#                  label="Raw ωy (rad/s)")
#         ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 2], "--", c="blue",
#                  label="Raw ωz (rad/s)")
#
#         # Plot filtered angular velocity (solid lines)
#         ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred",
#                  label="Filtered ωx (rad/s)")
#         ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
#                  label="Filtered ωy (rad/s)")
#         ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 2], "-", c="darkblue",
#                  label="Filtered ωz (rad/s)")
#
#         # Calculate angular velocity magnitude
#         angular_magnitude = np.sqrt(
#             self.angular_velocity_filtered[:, self.idx_point, 0] ** 2 +
#             self.angular_velocity_filtered[:, self.idx_point, 1] ** 2 +
#             self.angular_velocity_filtered[:, self.idx_point, 2] ** 2
#         )
#         ax3.plot(t_angular, angular_magnitude, "-", c="black", label="Angular Velocity Magnitude (rad/s)")
#
#         # Set third subplot title and labels
#         ax3.set_title(f"Angular Velocity of Point #{self.idx_point}")
#         ax3.set_xlabel("Time (seconds)")
#         ax3.set_ylabel("Angular Velocity (radians/second)")
#         ax3.set_xlim(0, 20)
#         # Fixed y-axis range for angular velocity
#         ax3.set_ylim(-0.25, 0.25)
#         ax3.grid(True, linestyle='--', alpha=0.7)
#         ax3.legend(loc='upper right')
#
#         # Adjust spacing between subplots
#         plt.tight_layout()
#
#         # Create binary buffer in memory
#         buf = BytesIO()
#         # Save plot as PNG format to buffer
#         plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
#         # Load image from buffer and convert to RGB format
#         image = Image.open(buf).convert('RGB')
#         # Convert image to NumPy array
#         rgb_array = np.array(image)
#         # Add image to Polyscope interface and enable display
#         ps.add_color_image_quantity("plot", rgb_array / 255.0, enabled=True)
#         # Close figure to release resources
#         plt.close(fig)
#
#     def save_current_plot(self):
#         """Save current point's trajectory plot to file"""
#         # Create time axis array (seconds)
#         t = np.arange(self.T) * self.dt_mean
#         t_angular = np.arange(self.T - 1) * self.dt_mean
#
#         # Create a figure with three subplots
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), dpi=100)
#
#         # First subplot: position data
#         ax1.plot(t, self.d[:, self.idx_point, 0], "--", c="red", label="Original X (m)")
#         ax1.plot(t, self.d[:, self.idx_point, 1], "--", c="green", label="Original Y (m)")
#         ax1.plot(t, self.d[:, self.idx_point, 2], "--", c="blue", label="Original Z (m)")
#
#         ax1.plot(t, self.d_filter[:, self.idx_point, 0], "-", c="darkred", label="Filtered X (m)")
#         ax1.plot(t, self.d_filter[:, self.idx_point, 1], "-", c="darkgreen", label="Filtered Y (m)")
#         ax1.plot(t, self.d_filter[:, self.idx_point, 2], "-", c="darkblue", label="Filtered Z (m)")
#
#         ax1.set_title(f"Position Trajectory of Point #{self.idx_point}")
#         ax1.set_xlabel("Time (seconds)")
#         ax1.set_ylabel("Position (meters)")
#         ax1.set_xlim(0, 20)
#         ax1.set_ylim(-1, 10)
#         ax1.grid(True, linestyle='--', alpha=0.7)
#         ax1.legend(loc='upper right')
#
#         # Second subplot: velocity data
#         ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity (m/s)")
#         ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity (m/s)")
#         ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity (m/s)")
#
#         velocity_magnitude = np.sqrt(
#             self.dv_filter[:, self.idx_point, 0] ** 2 +
#             self.dv_filter[:, self.idx_point, 1] ** 2 +
#             self.dv_filter[:, self.idx_point, 2] ** 2
#         )
#         ax2.plot(t, velocity_magnitude, "-", c="black", label="Velocity Magnitude (m/s)")
#
#         ax2.set_title(f"Linear Velocity of Point #{self.idx_point}")
#         ax2.set_xlabel("Time (seconds)")
#         ax2.set_ylabel("Velocity (meters/second)")
#         ax2.set_xlim(0, 20)
#         ax2.set_ylim(-1, 1)
#         ax2.grid(True, linestyle='--', alpha=0.7)
#         ax2.legend(loc='upper right')
#
#         # Third subplot: angular velocity data
#         ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red",
#                  label="Raw ωx (rad/s)")
#         ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green",
#                  label="Raw ωy (rad/s)")
#         ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 2], "--", c="blue",
#                  label="Raw ωz (rad/s)")
#
#         ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred",
#                  label="Filtered ωx (rad/s)")
#         ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
#                  label="Filtered ωy (rad/s)")
#         ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 2], "-", c="darkblue",
#                  label="Filtered ωz (rad/s)")
#
#         angular_magnitude = np.sqrt(
#             self.angular_velocity_filtered[:, self.idx_point, 0] ** 2 +
#             self.angular_velocity_filtered[:, self.idx_point, 1] ** 2 +
#             self.angular_velocity_filtered[:, self.idx_point, 2] ** 2
#         )
#         ax3.plot(t_angular, angular_magnitude, "-", c="black", label="Angular Velocity Magnitude (rad/s)")
#
#         ax3.set_title(f"Angular Velocity of Point #{self.idx_point}")
#         ax3.set_xlabel("Time (seconds)")
#         ax3.set_ylabel("Angular Velocity (radians/second)")
#         ax3.set_xlim(0, 20)
#         ax3.set_ylim(-0.25, 0.25)
#         ax3.grid(True, linestyle='--', alpha=0.7)
#         ax3.legend(loc='upper right')
#
#         plt.tight_layout()
#
#         # Create filename including timestamp and point index
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{self.output_dir}/point_{self.idx_point}_t{self.t}_{timestamp}.png"
#
#         # Save to file
#         plt.savefig(filename, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#
#         print(f"Plot saved to: {filename}")
#         return filename
#
#     def callback(self):
#         # Display title text
#         psim.Text("Point Cloud Motion Analysis")
#         # Create time frame slider, returns change status and new value
#         self.t_changed, self.t = psim.SliderInt("Time Frame", self.t, 0, self.T - 1)
#         # Create point index slider, returns change status and new value
#         self.idx_point_changed, self.idx_point = psim.SliderInt("Point Index", self.idx_point, 0, self.N - 1)
#         # Number of neighbors slider
#         changed_neighbors, self.num_neighbors = psim.SliderInt("Neighbors for SVD", self.num_neighbors, 5, 30)
#
#         # If time frame changes, update point cloud display
#         if self.t_changed:
#             register_point_cloud(
#                 "door", self.d_filter[self.t], radius=0.01, enabled=True
#             )
#         # If point index changes, redraw trajectory plot
#         if self.idx_point_changed:
#             self.plot_image()
#
#         # Add save image button
#         if psim.Button("Save Current Plot"):
#             saved_path = self.save_current_plot()
#             psim.Text(f"Saved to: {saved_path}")
#
#         # Add recalculate button
#         if changed_neighbors or psim.Button("Recalculate Angular Velocity"):
#             self.angular_velocity_raw, self.angular_velocity_filtered = self.calculate_angular_velocity()
#             self.plot_image()
#             psim.Text("Angular velocity recalculated")
#
#         # Display current data information
#         if psim.TreeNode("Data Information"):
#             psim.Text(f"Data shape: {self.d.shape}")
#             psim.Text(f"Number of time frames: {self.T}")
#             psim.Text(f"Number of points: {self.N}")
#             psim.Text(f"Time step: {self.dt_mean} seconds")
#
#             # Display current point's position, velocity and angular velocity
#             if self.t < self.T:
#                 pos = self.d_filter[self.t, self.idx_point]
#                 psim.Text(f"Position (m): X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
#
#                 if self.t < self.T - 1:
#                     # Linear velocity
#                     vel = self.dv_filter[self.t, self.idx_point]
#                     vel_mag = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
#                     psim.Text(f"Linear velocity (m/s): X={vel[0]:.3f}, Y={vel[1]:.3f}, Z={vel[2]:.3f}")
#                     psim.Text(f"Linear velocity magnitude: {vel_mag:.3f} m/s")
#
#                     # Angular velocity
#                     ang_vel = self.angular_velocity_filtered[
#                         min(self.t, len(self.angular_velocity_filtered) - 1), self.idx_point]
#                     ang_vel_mag = np.sqrt(ang_vel[0] ** 2 + ang_vel[1] ** 2 + ang_vel[2] ** 2)
#                     psim.Text(
#                         f"Angular velocity (rad/s): ωx={ang_vel[0]:.3f}, ωy={ang_vel[1]:.3f}, ωz={ang_vel[2]:.3f}")
#                     psim.Text(f"Angular velocity magnitude: {ang_vel_mag:.3f} rad/s ({np.degrees(ang_vel_mag):.2f}°/s)")
#
#             psim.TreePop()
#
#
# # Program entry point
# if __name__ == "__main__":
#     # Create Viz class instance and execute visualization
#     viz = Viz()


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from io import BytesIO
import os
from datetime import datetime
from robot_utils.viz.polyscope import PolyscopeUtils, ps, psim, register_point_cloud, draw_frame_3d


class Viz:
    def __init__(self):
        # Create Polyscope utils instance
        self.pu = PolyscopeUtils()
        # Initialize current time frame index and change flag
        self.t, self.t_changed = 0, False
        # Initialize current point index and change flag
        self.idx_point, self.idx_point_changed = 0, False
        # Load revolute joint point cloud data
        self.d = np.load("./demo_data/s1_refrigerator_part2_3180_3240.npy")
        # Add random noise
        # self.d += np.random.randn(*self.d.shape) * 0.01
        # Print data shape for debugging
        ic(self.d.shape)
        # Get data dimensions: number of time frames, points per frame, and coordinate dimensions
        self.T, self.N, _ = self.d.shape

        # Create folder for saving images
        self.output_dir = "visualization_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Average time step
        self.dt_mean = 0.1
        # Number of neighbors for SVD computation
        self.num_neighbors = 50

        # Apply Savitzky-Golay filter to smooth data
        self.d_filter = savgol_filter(
            x=self.d, window_length=21, polyorder=2, deriv=0, axis=0, delta=self.dt_mean
        )
        # Use Savitzky-Golay filter to calculate velocity (first derivative)
        self.dv_filter = savgol_filter(
            x=self.d, window_length=21, polyorder=2, deriv=1, axis=0, delta=self.dt_mean
        )

        # Compute angular velocity
        self.angular_velocity_raw, self.angular_velocity_filtered = self.calculate_angular_velocity()

        # Draw origin coordinate frame
        draw_frame_3d(np.zeros(6), label="origin", scale=0.1)
        # Register initial point cloud data and display
        register_point_cloud(
            "door", self.d[self.t], radius=0.01, enabled=True
        )
        # Reset view bounds to include point cloud
        self.pu.reset_bbox_from_pcl_list([self.d[self.t]])
        # Set user interaction callback function
        ps.set_user_callback(self.callback)
        # Show Polyscope interface
        ps.show()

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
        omega = omega * theta / dt

        return omega

    def calculate_angular_velocity(self):
        """Compute angular velocity using SVD on filtered point cloud data"""
        T, N, _ = self.d_filter.shape

        # Initialize arrays
        angular_velocity = np.zeros((T - 1, N, 3))

        # For each pair of consecutive frames
        for t in range(T - 1):
            # Current and next frame points
            current_points = self.d_filter[t]
            next_points = self.d_filter[t + 1]

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
                        window_length=21,
                        polyorder=2
                    )
        else:
            angular_velocity_filtered = angular_velocity.copy()

        return angular_velocity, angular_velocity_filtered

    def plot_image(self):
        # Create time axis array (seconds)
        t = np.arange(self.T) * self.dt_mean
        t_angular = np.arange(self.T - 1) * self.dt_mean

        # Create a figure with three subplots
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
        ax1.set_title(f"Position Trajectory of Point #{self.idx_point}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.set_xlim(0, 7)
        ax1.set_ylim(-1, 10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        # Second subplot: velocity data
        ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity")

        # Set second subplot title and labels
        ax2.set_title(f"Linear Velocity of Point #{self.idx_point}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlim(0, 7)
        ax2.set_ylim(-1, 1)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        # Third subplot: Angular velocity X component
        ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red", label="Raw ωx")
        ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred", label="Filtered ωx")

        ax3.set_title(f"Angular Velocity X Component of Point #{self.idx_point}")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angular Velocity (rad/s)")
        ax3.set_xlim(0, 7)
        ax3.set_ylim(-1, 1)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right')

        # Fourth subplot: Angular velocity Y component
        ax4.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green", label="Raw ωy")
        ax4.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
                 label="Filtered ωy")

        ax4.set_title(f"Angular Velocity Y Component of Point #{self.idx_point}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angular Velocity (rad/s)")
        ax4.set_xlim(0, 7)
        ax4.set_ylim(-1, 1)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')

        # Fifth subplot: Angular velocity Z component
        ax5.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 2], "--", c="blue", label="Raw ωz")
        ax5.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 2], "-", c="darkblue",
                 label="Filtered ωz")
        # Add groundtruth line

        ax5.set_title(f"Angular Velocity Z Component of Point #{self.idx_point}")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Angular Velocity (rad/s)")
        ax5.set_xlim(0, 7)
        ax5.set_ylim(-1, 1)
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
        # Create time axis array (seconds)
        t = np.arange(self.T) * self.dt_mean
        t_angular = np.arange(self.T - 1) * self.dt_mean

        # Create a figure with three subplots
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), dpi=100)

        ax1.plot(t, self.d[:, self.idx_point, 0], "--", c="red", label="Original X")
        ax1.plot(t, self.d[:, self.idx_point, 1], "--", c="green", label="Original Y")
        ax1.plot(t, self.d[:, self.idx_point, 2], "--", c="blue", label="Original Z")

        # Plot filtered data
        ax1.plot(t, self.d_filter[:, self.idx_point, 0], "-", c="darkred", label="Filtered X")
        ax1.plot(t, self.d_filter[:, self.idx_point, 1], "-", c="darkgreen", label="Filtered Y")
        ax1.plot(t, self.d_filter[:, self.idx_point, 2], "-", c="darkblue", label="Filtered Z")

        # Set first subplot title and labels
        ax1.set_title(f"Position Trajectory of Point #{self.idx_point}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.set_xlim(0, 7)
        ax1.set_ylim(-1, 10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        # Second subplot: velocity data
        ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity")

        # Set second subplot title and labels
        ax2.set_title(f"Linear Velocity of Point #{self.idx_point}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlim(0, 7)
        ax2.set_ylim(-1, 1)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        # Third subplot: Angular velocity X component
        ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red", label="Raw ωx")
        ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred", label="Filtered ωx")

        ax3.set_title(f"Angular Velocity X Component of Point #{self.idx_point}")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angular Velocity (rad/s)")
        ax3.set_xlim(0, 7)
        ax3.set_ylim(-1, 1)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right')

        # Fourth subplot: Angular velocity Y component
        ax4.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green", label="Raw ωy")
        ax4.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
                 label="Filtered ωy")

        ax4.set_title(f"Angular Velocity Y Component of Point #{self.idx_point}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angular Velocity (rad/s)")
        ax4.set_xlim(0, 7)
        ax4.set_ylim(-1, 1)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')

        # Fifth subplot: Angular velocity Z component
        ax5.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 2], "--", c="blue", label="Raw ωz")
        ax5.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 2], "-", c="darkblue",
                 label="Filtered ωz")
        # Add groundtruth line

        ax5.set_title(f"Angular Velocity Z Component of Point #{self.idx_point}")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Angular Velocity (rad/s)")
        ax5.set_xlim(0, 7)
        ax5.set_ylim(-1, 1)
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend(loc='upper right')

        plt.tight_layout()

        # Create filename including timestamp and point index
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/point_{self.idx_point}_t{self.t}_{timestamp}.png"

        # Save to file
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Plot saved to: {filename}")
        return filename

    def callback(self):
        # Display title text
        psim.Text("Point Cloud Motion Analysis")
        # Create time frame slider, returns change status and new value
        self.t_changed, self.t = psim.SliderInt("Time Frame", self.t, 0, self.T - 1)
        # Create point index slider, returns change status and new value
        self.idx_point_changed, self.idx_point = psim.SliderInt("Point Index", self.idx_point, 0, self.N - 1)
        # Number of neighbors slider
        changed_neighbors, self.num_neighbors = psim.SliderInt("Neighbors for SVD", self.num_neighbors, 5, 30)

        # If time frame changes, update point cloud display
        if self.t_changed:
            register_point_cloud(
                "door", self.d_filter[self.t], radius=0.01, enabled=True
            )
        # If point index changes, redraw trajectory plot
        if self.idx_point_changed:
            self.plot_image()

        # Add save image button
        if psim.Button("Save Current Plot"):
            saved_path = self.save_current_plot()
            psim.Text(f"Saved to: {saved_path}")

        # Add recalculate button
        if changed_neighbors or psim.Button("Recalculate Angular Velocity"):
            self.angular_velocity_raw, self.angular_velocity_filtered = self.calculate_angular_velocity()
            self.plot_image()
            psim.Text("Angular velocity recalculated")

        # Display current data information
        if psim.TreeNode("Data Information"):
            psim.Text(f"Data shape: {self.d.shape}")
            psim.Text(f"Number of time frames: {self.T}")
            psim.Text(f"Number of points: {self.N}")
            psim.Text(f"Time step: {self.dt_mean} seconds")

            # Display current point's position, velocity and angular velocity
            if self.t < self.T:
                pos = self.d_filter[self.t, self.idx_point]
                psim.Text(f"Position (m): X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")

                if self.t < self.T - 1:
                    # Linear velocity
                    vel = self.dv_filter[self.t, self.idx_point]
                    vel_mag = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
                    psim.Text(f"Linear velocity (m/s): X={vel[0]:.3f}, Y={vel[1]:.3f}, Z={vel[2]:.3f}")
                    psim.Text(f"Linear velocity magnitude: {vel_mag:.3f} m/s")

                    # Angular velocity
                    ang_vel = self.angular_velocity_filtered[
                        min(self.t, len(self.angular_velocity_filtered) - 1), self.idx_point]
                    ang_vel_mag = np.sqrt(ang_vel[0] ** 2 + ang_vel[1] ** 2 + ang_vel[2] ** 2)
                    psim.Text(
                        f"Angular velocity (rad/s): ωx={ang_vel[0]:.3f}, ωy={ang_vel[1]:.3f}, ωz={ang_vel[2]:.3f}")
                    psim.Text(f"Angular velocity magnitude: {ang_vel_mag:.3f} rad/s ({np.degrees(ang_vel_mag):.2f}°/s)")

            psim.TreePop()


# Program entry point
if __name__ == "__main__":
    # Create Viz class instance and execute visualization
    viz = Viz()