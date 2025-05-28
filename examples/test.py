import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from io import BytesIO
import os
from datetime import datetime
from robot_utils.viz.polyscope import PolyscopeUtils, ps, psim, register_point_cloud, draw_frame_3d

class Viz:
    def __init__(self, file_paths=None):
        # 初始化默认文件路径列表，如果没有提供则使用默认值
        if file_paths is None:
            self.file_paths = ["./demo_data/s1_gasstove_part2_1110_1170.npy"]
        else:
            self.file_paths = file_paths

        # 创建Polyscope工具实例
        self.pu = PolyscopeUtils()
        # 初始化当前时间帧索引和改变标志
        self.t, self.t_changed = 0, False
        # 初始化当前点索引和改变标志
        self.idx_point, self.idx_point_changed = 0, False
        # 当前选择的数据集索引
        self.current_dataset = 0
        self.noise_sigma = 0
        # 储存多个数据集的字典
        self.datasets = {}
        # 加载所有文件
        for i, file_path in enumerate(self.file_paths):
            # 给每个数据集一个名称标识
            dataset_name = f"dataset_{i}"
            # 加载数据
            data = np.load(file_path)
            # 为可能的相对路径创建基于文件名的数据集名称
            display_name = os.path.basename(file_path).split('.')[0]
            if self.noise_sigma > 0:
                data += np.random.randn(*data.shape) * self.noise_sigma

            # 存储数据集信息
            self.datasets[dataset_name] = {
                "path": file_path,
                "display_name": display_name,
                "data": data,
                "T": data.shape[0],  # 时间帧数
                "N": data.shape[1],  # 每帧点数
                "visible": True  # 是否显示
            }

            # 应用Savitzky-Golay滤波器平滑数据
            self.datasets[dataset_name]["data_filter"] = savgol_filter(
                x=data, window_length=21, polyorder=2, deriv=0, axis=0, delta=0.1
            )

            # 使用Savitzky-Golay滤波器计算速度(一阶导数)
            self.datasets[dataset_name]["dv_filter"] = savgol_filter(
                x=data, window_length=21, polyorder=2, deriv=1, axis=0, delta=0.1
            )

        # 使用第一个数据集作为当前数据集
        dataset_keys = list(self.datasets.keys())
        if dataset_keys:
            self.current_dataset_key = dataset_keys[0]
            current_data = self.datasets[self.current_dataset_key]
            # 获取当前数据集的维度
            self.T = current_data["T"]
            self.N = current_data["N"]
            # 当前活跃数据
            self.d = current_data["data"]
            self.d_filter = current_data["data_filter"]
            self.dv_filter = current_data["dv_filter"]
        else:
            print("No datasets loaded.")
            return

        # 平均时间步长
        self.dt_mean = 0.1
        # SVD计算的邻居数
        self.num_neighbors = 50

        # 创建保存图像的文件夹
        self.output_dir = "visualization_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # 计算角速度
        for dataset_key in self.datasets:
            dataset = self.datasets[dataset_key]
            angular_velocity_raw, angular_velocity_filtered = self.calculate_angular_velocity(
                dataset["data_filter"], dataset["N"]
            )
            dataset["angular_velocity_raw"] = angular_velocity_raw
            dataset["angular_velocity_filtered"] = angular_velocity_filtered

        # 当前数据集的角速度
        self.angular_velocity_raw = self.datasets[self.current_dataset_key]["angular_velocity_raw"]
        self.angular_velocity_filtered = self.datasets[self.current_dataset_key]["angular_velocity_filtered"]

        # 绘制原点坐标系
        draw_frame_3d(np.zeros(6), label="origin", scale=0.1)

        # 注册所有数据集的初始点云并显示
        for dataset_key, dataset in self.datasets.items():
            if dataset["visible"]:
                register_point_cloud(
                    dataset["display_name"],
                    dataset["data_filter"][self.t],
                    radius=0.01,
                    enabled=True
                )

        # 重置视图边界以包含所有点云
        visible_point_clouds = [dataset["data_filter"][self.t] for dataset in self.datasets.values()
                                if dataset["visible"]]
        if visible_point_clouds:
            self.pu.reset_bbox_from_pcl_list(visible_point_clouds)

        # 设置用户交互回调函数
        ps.set_user_callback(self.callback)
        # 显示Polyscope界面
        ps.show()

    def find_neighbors(self, points, num_neighbors):
        """寻找每个点的邻居"""
        # 计算距离矩阵
        N = points.shape[0]
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            dist_matrix[i] = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))

        # 寻找最近的点(排除自身)
        neighbors = np.zeros((N, num_neighbors), dtype=int)
        for i in range(N):
            indices = np.argsort(dist_matrix[i])[1:num_neighbors + 1]
            neighbors[i] = indices

        return neighbors

    def compute_rotation_matrix(self, src_points, dst_points):
        """使用SVD计算两组点之间的旋转矩阵"""
        # 点的中心
        src_center = np.mean(src_points, axis=0)
        dst_center = np.mean(dst_points, axis=0)

        src_centered = src_points - src_center
        dst_centered = dst_points - dst_center

        # 计算协方差矩阵
        H = np.dot(src_centered.T, dst_centered)

        # SVD分解
        U, _, Vt = np.linalg.svd(H)

        # 构造旋转矩阵
        R = np.dot(Vt.T, U.T)

        # 处理反射情况
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        return R, src_center, dst_center

    def rotation_matrix_to_angular_velocity(self, R, dt):
        """从旋转矩阵提取角速度"""
        # 确保R是有效的旋转矩阵
        U, _, Vt = np.linalg.svd(R)
        R = np.dot(U, Vt)

        # 计算旋转角度
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # 如果角度太小，返回零向量
        if abs(theta) < 1e-6:
            return np.zeros(3)

        # 计算旋转轴
        sin_theta = np.sin(theta)
        if abs(sin_theta) < 1e-6:
            return np.zeros(3)

        # 从反对称部分提取角速度向量
        W = (R - R.T) / (2 * sin_theta)
        omega = np.zeros(3)
        omega[0] = W[2, 1]
        omega[1] = W[0, 2]
        omega[2] = W[1, 0]

        # 角速度 = 轴 * 角度 / 时间
        omega = omega * theta / dt

        return omega

    def calculate_angular_velocity(self, d_filter, N):
        """使用SVD在滤波点云数据上计算角速度"""
        T = d_filter.shape[0]

        # 初始化数组
        angular_velocity = np.zeros((T - 1, N, 3))

        # 对于每对连续帧
        for t in range(T - 1):
            # 当前和下一帧点
            current_points = d_filter[t]
            next_points = d_filter[t + 1]

            # 为所有点寻找邻居
            neighbors = self.find_neighbors(current_points, self.num_neighbors)

            # 计算每个点的角速度
            for i in range(N):
                # 获取当前点及其邻居
                src_neighborhood = current_points[neighbors[i]]
                dst_neighborhood = next_points[neighbors[i]]

                # 计算旋转矩阵
                R, _, _ = self.compute_rotation_matrix(src_neighborhood, dst_neighborhood)

                # 提取角速度
                omega = self.rotation_matrix_to_angular_velocity(R, self.dt_mean)

                # 存储结果
                angular_velocity[t, i] = omega

        # 滤波角速度
        angular_velocity_filtered = np.zeros_like(angular_velocity)

        if T - 1 >= 5:
            for i in range(N):
                for dim in range(3):
                    angular_velocity_filtered[:, i, dim] = savgol_filter(
                        angular_velocity[:, i, dim],
                        window_length=min(21, T - 1),  # 确保窗口长度不超过数据长度
                        polyorder=min(2, (T - 1) // 2)  # 确保多项式阶数适当
                    )
        else:
            angular_velocity_filtered = angular_velocity.copy()

        return angular_velocity, angular_velocity_filtered

    def switch_dataset(self, new_dataset_key):
        """切换当前活跃的数据集"""
        if new_dataset_key in self.datasets:
            self.current_dataset_key = new_dataset_key
            dataset = self.datasets[new_dataset_key]

            # 更新当前数据和时间/点的限制
            self.d = dataset["data"]
            self.d_filter = dataset["data_filter"]
            self.dv_filter = dataset["dv_filter"]
            self.angular_velocity_raw = dataset["angular_velocity_raw"]
            self.angular_velocity_filtered = dataset["angular_velocity_filtered"]

            # 更新T和N的值
            self.T = dataset["T"]
            self.N = dataset["N"]

            # 确保当前的t和idx_point在有效范围内
            self.t = min(self.t, self.T - 1)
            self.idx_point = min(self.idx_point, self.N - 1)

            # 更新图表
            self.plot_image()

            return True
        return False

    def toggle_visibility(self, dataset_key):
        """切换数据集的可见性"""
        if dataset_key in self.datasets:
            self.datasets[dataset_key]["visible"] = not self.datasets[dataset_key]["visible"]
            return True
        return False

    def plot_image(self):
        """绘制当前点的轨迹图"""
        dataset = self.datasets[self.current_dataset_key]

        # 创建时间轴数组（秒）
        t = np.arange(self.T) * self.dt_mean
        t_angular = np.arange(self.T - 1) * self.dt_mean

        # 创建一个有五个子图的图形
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), dpi=100)

        # 第一个子图：位置数据
        # 绘制原始数据
        ax1.plot(t, self.d[:, self.idx_point, 0], "--", c="red", label="Original X")
        ax1.plot(t, self.d[:, self.idx_point, 1], "--", c="green", label="Original Y")
        ax1.plot(t, self.d[:, self.idx_point, 2], "--", c="blue", label="Original Z")

        # 绘制滤波数据
        ax1.plot(t, self.d_filter[:, self.idx_point, 0], "-", c="darkred", label="Filtered X")
        ax1.plot(t, self.d_filter[:, self.idx_point, 1], "-", c="darkgreen", label="Filtered Y")
        ax1.plot(t, self.d_filter[:, self.idx_point, 2], "-", c="darkblue", label="Filtered Z")

        # 设置第一个子图标题和标签
        ax1.set_title(f"Position Trajectory of Point #{self.idx_point} - {dataset['display_name']}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.set_xlim(0, 20)
        ax1.set_ylim(-1, 10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        # 第二个子图：速度数据
        ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity")

        # 设置第二个子图标题和标签
        ax2.set_title(f"Linear Velocity of Point #{self.idx_point} - {dataset['display_name']}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlim(0, 20)
        ax2.set_ylim(-1, 1)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        # 第三个子图：角速度X分量
        ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red", label="Raw ωx")
        ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred", label="Filtered ωx")

        ax3.set_title(f"Angular Velocity X Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angular Velocity (rad/s)")
        ax3.set_xlim(0, 20)
        ax3.set_ylim(-0.25, 0.25)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right')

        # 第四个子图：角速度Y分量
        ax4.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green", label="Raw ωy")
        ax4.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
                 label="Filtered ωy")

        ax4.set_title(f"Angular Velocity Y Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angular Velocity (rad/s)")
        ax4.set_xlim(0, 20)
        ax4.set_ylim(-1, 1)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')

        # 第五个子图：角速度Z分量
        ax5.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 2], "--", c="blue", label="Raw ωz")
        ax5.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 2], "-", c="darkblue",
                 label="Filtered ωz")

        ax5.set_title(f"Angular Velocity Z Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Angular Velocity (rad/s)")
        ax5.set_xlim(0, 20)
        ax5.set_ylim(-0.25, 0.25)
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend(loc='upper right')

        # 调整子图之间的间距
        plt.tight_layout()

        # 创建内存中的二进制缓冲区
        buf = BytesIO()
        # 将图保存为PNG格式到缓冲区
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        # 从缓冲区加载图像并转换为RGB格式
        image = Image.open(buf).convert('RGB')
        # 将图像转换为NumPy数组
        rgb_array = np.array(image)
        # 将图像添加到Polyscope界面并启用显示
        ps.add_color_image_quantity("plot", rgb_array / 255.0, enabled=True)
        # 关闭图形以释放资源
        plt.close(fig)

    def save_current_plot(self):
        """将当前点的轨迹图保存到文件"""
        dataset = self.datasets[self.current_dataset_key]

        # 创建时间轴数组（秒）
        t = np.arange(self.T) * self.dt_mean
        t_angular = np.arange(self.T - 1) * self.dt_mean

        # 创建带有五个子图的图形
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), dpi=100)

        # 绘制位置数据
        ax1.plot(t, self.d[:, self.idx_point, 0], "--", c="red", label="Original X")
        ax1.plot(t, self.d[:, self.idx_point, 1], "--", c="green", label="Original Y")
        ax1.plot(t, self.d[:, self.idx_point, 2], "--", c="blue", label="Original Z")
        ax1.plot(t, self.d_filter[:, self.idx_point, 0], "-", c="darkred", label="Filtered X")
        ax1.plot(t, self.d_filter[:, self.idx_point, 1], "-", c="darkgreen", label="Filtered Y")
        ax1.plot(t, self.d_filter[:, self.idx_point, 2], "-", c="darkblue", label="Filtered Z")
        ax1.set_title(f"Position Trajectory of Point #{self.idx_point} - {dataset['display_name']}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.set_xlim(0, 20)
        ax1.set_ylim(-1, 10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        # 绘制速度数据
        ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity")
        ax2.set_title(f"Linear Velocity of Point #{self.idx_point} - {dataset['display_name']}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlim(0, 20)
        ax2.set_ylim(-1, 1)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        # 绘制角速度X分量
        ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red", label="Raw ωx")
        ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred", label="Filtered ωx")
        ax3.set_title(f"Angular Velocity X Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angular Velocity (rad/s)")
        ax3.set_xlim(0, 20)
        ax3.set_ylim(-0.25, 0.25)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right')

        # 绘制角速度Y分量
        ax4.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green", label="Raw ωy")
        ax4.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
                 label="Filtered ωy")
        ax4.set_title(f"Angular Velocity Y Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angular Velocity (rad/s)")
        ax4.set_xlim(0, 20)
        ax4.set_ylim(-1, 1)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')

        # 绘制角速度Z分量
        ax5.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 2], "--", c="blue", label="Raw ωz")
        ax5.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 2], "-", c="darkblue",
                 label="Filtered ωz")
        ax5.set_title(f"Angular Velocity Z Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Angular Velocity (rad/s)")
        ax5.set_xlim(0, 20)
        ax5.set_ylim(-0.25, 0.25)
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend(loc='upper right')

        plt.tight_layout()

        # 创建包含时间戳和点索引的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{dataset['display_name']}_point_{self.idx_point}_t{self.t}_{timestamp}.png"

        # 保存到文件
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Plot saved to: {filename}")
        return filename

    def callback(self):
        # 显示标题文本
        psim.Text("Point Cloud Motion Analysis")

        # 显示当前数据集信息
        psim.Text(f"Active Dataset: {self.datasets[self.current_dataset_key]['display_name']}")

        # 创建时间帧滑块，返回更改状态和新值
        self.t_changed, self.t = psim.SliderInt("Time Frame", self.t, 0, self.T - 1)
        # 创建点索引滑块，返回更改状态和新值
        self.idx_point_changed, self.idx_point = psim.SliderInt("Point Index", self.idx_point, 0, self.N - 1)
        # SVD的邻居数滑块
        changed_neighbors, self.num_neighbors = psim.SliderInt("Neighbors for SVD", self.num_neighbors, 5, 30)

        # 如果时间帧发生变化，更新所有可见数据集的点云显示
        if self.t_changed:
            for dataset_key, dataset in self.datasets.items():
                if dataset["visible"] and self.t < dataset["T"]:
                    register_point_cloud(
                        dataset["display_name"],
                        dataset["data_filter"][min(self.t, dataset["T"] - 1)],
                        radius=0.01,
                        enabled=True
                    )

        # 如果点索引发生变化，重绘轨迹图
        if self.idx_point_changed:
            self.plot_image()

        # 添加数据集选择下拉菜单
        if psim.TreeNode("Dataset Selection"):
            # 显示数据集列表，允许选择和切换可见性
            for dataset_key, dataset in self.datasets.items():
                if psim.Button(f"Select {dataset['display_name']}"):
                    self.switch_dataset(dataset_key)

                psim.SameLine()
                vis_text = "Visible" if dataset["visible"] else "Hidden"
                if psim.Button(f"{vis_text}###{dataset_key}"):
                    self.toggle_visibility(dataset_key)
                    # 更新点云显示
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

        # 添加保存图像按钮
        if psim.Button("Save Current Plot"):
            saved_path = self.save_current_plot()
            psim.Text(f"Saved to: {saved_path}")

        # 添加重新计算按钮
        if changed_neighbors or psim.Button("Recalculate Angular Velocity"):
            for dataset_key, dataset in self.datasets.items():
                angular_velocity_raw, angular_velocity_filtered = self.calculate_angular_velocity(
                    dataset["data_filter"], dataset["N"]
                )
                dataset["angular_velocity_raw"] = angular_velocity_raw
                dataset["angular_velocity_filtered"] = angular_velocity_filtered

            # 更新当前数据集的角速度
            self.angular_velocity_raw = self.datasets[self.current_dataset_key]["angular_velocity_raw"]
            self.angular_velocity_filtered = self.datasets[self.current_dataset_key]["angular_velocity_filtered"]

            self.plot_image()
            psim.Text("Angular velocity recalculated")

        # 显示当前数据信息
        if psim.TreeNode("Data Information"):
            dataset = self.datasets[self.current_dataset_key]
            psim.Text(f"Current dataset: {dataset['display_name']}")
            psim.Text(f"Data shape: {dataset['data'].shape}")
            psim.Text(f"Number of time frames: {dataset['T']}")
            psim.Text(f"Number of points: {dataset['N']}")
            psim.Text(f"Time step: {self.dt_mean} seconds")

            # 显示当前点的位置、速度和角速度
            if self.t < self.T:
                pos = self.d_filter[self.t, self.idx_point]
                psim.Text(f"Position (m): X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")

                if self.t < self.T - 1:
                    # 线速度
                    vel = self.dv_filter[self.t, self.idx_point]
                    vel_mag = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
                    psim.Text(f"Linear velocity (m/s): X={vel[0]:.3f}, Y={vel[1]:.3f}, Z={vel[2]:.3f}")
                    psim.Text(f"Linear velocity magnitude: {vel_mag:.3f} m/s")

                    # 角速度
                    ang_vel_idx = min(self.t, len(self.angular_velocity_filtered) - 1)
                    if ang_vel_idx >= 0 and self.idx_point < self.angular_velocity_filtered.shape[1]:
                        ang_vel = self.angular_velocity_filtered[ang_vel_idx, self.idx_point]
                        ang_vel_mag = np.sqrt(ang_vel[0] ** 2 + ang_vel[1] ** 2 + ang_vel[2] ** 2)
                        psim.Text(
                            f"Angular velocity (rad/s): ωx={ang_vel[0]:.3f}, ωy={ang_vel[1]:.3f}, ωz={ang_vel[2]:.3f}")
                        psim.Text(
                            f"Angular velocity magnitude: {ang_vel_mag:.3f} rad/s ({np.degrees(ang_vel_mag):.2f}°/s)")

            psim.TreePop()


# 程序入口点
if __name__ == "__main__":
    # 你可以在这里指定多个数据文件路径
    file_paths = [
        # "./demo_data/s1_refrigerator_part2_3180_3240.npy",
        # "./demo_data/s1_refrigerator_base_3180_3240.npy",
        # "./demo_data/s1_refrigerator_part1_3180_3240.npy"

        "./demo_data/s2_drawer_base_1770_1950.npy",
        "./demo_data/s2_drawer_part1_1770_1950.npy",
        "./demo_data/s2_drawer_part2_1770_1950.npy"
    ]

    # 创建Viz类实例并执行可视化，传入多个文件路径
    viz = Viz(file_paths)