"""
Plot saving utilities for joint analysis.
"""

import os
import time
import matplotlib.pyplot as plt
import numpy as np


class PlotSaver:
    """
    Utility class to save plot data from JointAnalysisGUI to image files.
    """

    def __init__(self, output_dir="plot_images"):
        """
        Initialize the PlotSaver.

        Args:
            output_dir (str): Directory to save plot images
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 记录当前时间作为保存的前缀，避免覆盖
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def _get_filename_suffix(self, app_instance=None):
        """
        Generate a suffix for filenames based on current parameters.

        Args:
            app_instance: JointAnalysisApp instance to get parameters from

        Returns:
            str: Suffix string for filename
        """
        suffix = []

        # 如果没有传入app实例，就返回时间戳
        if app_instance is None:
            return f"_{self.timestamp}"

        # 添加noise参数
        if hasattr(app_instance, 'noise_sigma'):
            suffix.append(f"noise{app_instance.noise_sigma:.4f}")

        # 是否使用SG滤波
        if hasattr(app_instance, 'use_savgol_filter') and app_instance.use_savgol_filter:
            sg_win = getattr(app_instance, 'savgol_window_length', 0)
            sg_poly = getattr(app_instance, 'savgol_polyorder', 0)
            suffix.append(f"sg{sg_win}_{sg_poly}")

        # 是否使用EKF
        if hasattr(app_instance, 'use_ekf_3d') and app_instance.use_ekf_3d:
            suffix.append("ekf")

        # 是否使用多帧拟合
        if hasattr(app_instance, 'use_multi_frame_fit') and app_instance.use_multi_frame_fit:
            mf_radius = getattr(app_instance, 'multi_frame_radius', 0)
            suffix.append(f"mf{mf_radius}")

        # 邻居数量
        if hasattr(app_instance, 'neighbor_k'):
            suffix.append(f"k{app_instance.neighbor_k}")

        # 如果没有任何参数，使用时间戳
        if not suffix:
            return f"_{self.timestamp}"

        # 组合所有参数
        return f"_{self.timestamp}_{'_'.join(suffix)}"

    def save_velocity_plots(self, mode, velocity_data, angular_velocity_data, app_instance=None):
        """
        Save velocity and angular velocity plots.

        Args:
            mode (str): Joint mode name
            velocity_data (list): Linear velocity data
            angular_velocity_data (list): Angular velocity data
            app_instance: JointAnalysisApp instance to get parameters from
        """
        # 创建模式特定的目录
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # 获取文件名后缀
        suffix = self._get_filename_suffix(app_instance)

        # 绘制线速度
        plt.figure(figsize=(10, 6))
        x_data = list(range(len(velocity_data)))
        plt.plot(x_data, velocity_data, 'b-', linewidth=2)
        plt.title(f'Linear Velocity over Time - {mode}')
        plt.xlabel('Time Frames')
        plt.ylabel('Linear Velocity Magnitude(m/s)')
        plt.grid(True)
        plt.savefig(os.path.join(mode_dir, f'linear_velocity{suffix}.png'), dpi=50)
        plt.close()

        # 绘制角速度
        plt.figure(figsize=(10, 6))
        x_data = list(range(len(angular_velocity_data)))
        plt.plot(x_data, angular_velocity_data, 'r-', linewidth=2)
        plt.title(f'Angular Velocity over Time - {mode}')
        plt.xlabel('Time Frames')
        plt.ylabel('Angular Velocity Magnitude(rad/s)')
        plt.grid(True)
        plt.savefig(os.path.join(mode_dir, f'angular_velocity{suffix}.png'), dpi=50)
        plt.close()

    def save_basic_scores_plots(self, mode, col_data, cop_data, rad_data, zp_data, app_instance=None):
        """
        Save basic scores plots.

        Args:
            mode (str): Joint mode name
            col_data (list): Collinearity scores
            cop_data (list): Coplanarity scores
            rad_data (list): Radius consistency scores
            zp_data (list): Zero pitch scores
            app_instance: JointAnalysisApp instance to get parameters from
        """
        # 创建模式特定的目录
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # 获取文件名后缀
        suffix = self._get_filename_suffix(app_instance)

        # 绘制基本分数
        plt.figure(figsize=(12, 8))
        x_data = list(range(len(col_data)))

        plt.subplot(2, 2, 1)
        plt.plot(x_data, col_data, 'g-', linewidth=2)
        plt.title('Collinearity Score')
        plt.xlabel('Time Frames')
        plt.ylabel('Score')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(x_data, cop_data, 'b-', linewidth=2)
        plt.title('Coplanarity Score')
        plt.xlabel('Time Frames')
        plt.ylabel('Score')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(x_data, rad_data, 'm-', linewidth=2)
        plt.title('Radius Consistency Score')
        plt.xlabel('Time Frames')
        plt.ylabel('Score')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(x_data, zp_data, 'c-', linewidth=2)
        plt.title('Zero Pitch Score')
        plt.xlabel('Time Frames')
        plt.ylabel('Score')
        plt.grid(True)

        plt.tight_layout()
        plt.suptitle(f'Basic Scores over Time - {mode}', y=1.02, fontsize=16)
        plt.savefig(os.path.join(mode_dir, f'basic_scores{suffix}.png'), dpi=50, bbox_inches='tight')
        plt.close()

    def save_joint_probability_plots(self, mode, joint_probs, app_instance=None):
        """
        Save joint probability plots.

        Args:
            mode (str): Joint mode name
            joint_probs (dict): Dictionary mapping joint types to probability data
            app_instance: JointAnalysisApp instance to get parameters from
        """
        # 创建模式特定的目录
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # 获取文件名后缀
        suffix = self._get_filename_suffix(app_instance)

        # 绘制关节概率
        plt.figure(figsize=(12, 8))

        # 获取公共x轴
        max_len = 0
        for jt_name, probs in joint_probs.items():
            max_len = max(max_len, len(probs))

        x_data = list(range(max_len))

        # 关节类型到颜色的映射
        colors = {
            'prismatic': 'b',
            'planar': 'g',
            'revolute': 'r',
            'screw': 'm',
            'ball': 'c'
        }

        for jt_name, probs in joint_probs.items():
            if len(probs) > 0:
                # 如果需要，用最后的值填充
                padded_probs = probs + [probs[-1]] * (max_len - len(probs)) if len(probs) < max_len else probs
                plt.plot(x_data, padded_probs, f'{colors.get(jt_name, "k")}-', linewidth=2, label=jt_name.capitalize())

        plt.title(f'Joint Type Probabilities over Time - {mode}')
        plt.xlabel('Time Frames')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(mode_dir, f'joint_probabilities{suffix}.png'), dpi=50)
        plt.close()

    def save_error_plots(self, mode, position_error_data, angular_error_data, app_instance=None):
        """
        Save error plots.

        Args:
            mode (str): Joint mode name
            position_error_data (list): Position error data
            angular_error_data (list): Angular error data
            app_instance: JointAnalysisApp instance to get parameters from
        """
        # 创建模式特定的目录
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # 获取文件名后缀
        suffix = self._get_filename_suffix(app_instance)

        # 绘制误差
        plt.figure(figsize=(12, 5))

        x_data = list(range(len(position_error_data)))

        plt.subplot(1, 2, 1)
        plt.plot(x_data, position_error_data, 'b-', linewidth=2)
        plt.title('Position Error')
        plt.xlabel('Time Frames')
        plt.ylabel('Error(m)')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(x_data, angular_error_data, 'r-', linewidth=2)
        plt.title('Angular Error')
        plt.xlabel('Time Frames')
        plt.ylabel('Error(rad)')
        plt.grid(True)

        plt.tight_layout()
        plt.suptitle(f'Error Metrics over Time - {mode}', y=1.05, fontsize=16)
        plt.savefig(os.path.join(mode_dir, f'error_metrics{suffix}.png'), dpi=50, bbox_inches='tight')
        plt.close()

    def save_all_plots(self, gui_instance, app_instance=None):
        """
        Save all plots for all modes in the GUI.

        Args:
            gui_instance: Instance of JointAnalysisGUI
            app_instance: JointAnalysisApp instance to get parameters from
        """
        for mode in gui_instance.modes:
            # 只有在有数据的情况下保存图表
            if mode in gui_instance.velocity_profile and len(gui_instance.velocity_profile[mode]) > 0:
                self.save_velocity_plots(
                    mode,
                    gui_instance.velocity_profile[mode],
                    gui_instance.angular_velocity_profile[mode],
                    app_instance
                )

                self.save_basic_scores_plots(
                    mode,
                    gui_instance.col_score_profile[mode],
                    gui_instance.cop_score_profile[mode],
                    gui_instance.rad_score_profile[mode],
                    gui_instance.zp_score_profile[mode],
                    app_instance
                )

                self.save_joint_probability_plots(
                    mode,
                    gui_instance.joint_prob_profile[mode],
                    app_instance
                )

                self.save_error_plots(
                    mode,
                    gui_instance.position_error_profile[mode],
                    gui_instance.angular_error_profile[mode],
                    app_instance
                )