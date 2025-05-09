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
        Save velocity and angular velocity plots with scientific formatting.

        Args:
            mode (str): Joint mode name
            velocity_data (list): Linear velocity data (m/s)
            angular_velocity_data (list): Angular velocity data (rad/s)
            app_instance: JointAnalysisApp instance to get parameters from
        """
        # Create mode-specific directory
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # Get file name suffix
        suffix = self._get_filename_suffix(app_instance)

        # Calculate mean velocities for display
        mean_vel = np.mean(velocity_data) if len(velocity_data) > 0 else 0.0
        mean_ang_vel = np.mean(angular_velocity_data) if len(angular_velocity_data) > 0 else 0.0

        # Plot linear velocity with scientific formatting
        plt.figure(figsize=(10, 6))
        x_data = list(range(len(velocity_data)))
        plt.plot(x_data, velocity_data, 'b-', linewidth=1.5)
        plt.plot(x_data, np.ones_like(x_data) * mean_vel, 'r--', linewidth=1)
        plt.title(f'Linear Velocity: {mode}')
        plt.xlabel('Time Frame')
        plt.ylabel('Linear Velocity (m/s)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.text(0.05, 0.95, f"Mean: {mean_vel:.3f} m/s",
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        plt.savefig(os.path.join(mode_dir, f'linear_velocity{suffix}.png'), dpi=300)
        plt.close()

        # Plot angular velocity with scientific formatting
        plt.figure(figsize=(10, 6))
        x_data = list(range(len(angular_velocity_data)))
        plt.plot(x_data, angular_velocity_data, 'r-', linewidth=1.5)
        plt.plot(x_data, np.ones_like(x_data) * mean_ang_vel, 'r--', linewidth=1)
        plt.title(f'Angular Velocity: {mode}')
        plt.xlabel('Time Frame')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.grid(True, linestyle='--', alpha=0.7)
        # Also show degrees for clarity
        mean_ang_vel_deg = np.degrees(mean_ang_vel)
        plt.text(0.05, 0.95, f"Mean: {mean_ang_vel:.3f} rad/s ({mean_ang_vel_deg:.2f}°/s)",
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        plt.savefig(os.path.join(mode_dir, f'angular_velocity{suffix}.png'), dpi=300)
        plt.close()

    def save_basic_scores_plots(self, mode, col_data, cop_data, rad_data, zp_data, app_instance=None):
        """
        Save basic scores plots with scientific paper formatting.

        Args:
            mode (str): Joint mode name
            col_data (list): Collinearity scores
            cop_data (list): Coplanarity scores
            rad_data (list): Radius consistency scores
            zp_data (list): Zero pitch scores
            app_instance: JointAnalysisApp instance to get parameters from
        """
        # Create mode-specific directory
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # Get file name suffix
        suffix = self._get_filename_suffix(app_instance)

        # Calculate mean scores for display
        mean_col = np.mean(col_data) if len(col_data) > 0 else 0.0
        mean_cop = np.mean(cop_data) if len(cop_data) > 0 else 0.0
        mean_rad = np.mean(rad_data) if len(rad_data) > 0 else 0.0
        mean_zp = np.mean(zp_data) if len(zp_data) > 0 else 0.0

        # Plot basic scores with scientific formatting
        plt.figure(figsize=(12, 10))
        x_data = list(range(len(col_data)))

        # Define plot parameters for consistency
        plot_params = [
            {'title': 'Collinearity Score', 'data': col_data, 'mean': mean_col, 'color': 'g', 'pos': 1},
            {'title': 'Coplanarity Score', 'data': cop_data, 'mean': mean_cop, 'color': 'b', 'pos': 2},
            {'title': 'Radius Consistency Score', 'data': rad_data, 'mean': mean_rad, 'color': 'm', 'pos': 3},
            {'title': 'Zero Pitch Score', 'data': zp_data, 'mean': mean_zp, 'color': 'c', 'pos': 4}
        ]

        for params in plot_params:
            plt.subplot(2, 2, params['pos'])
            plt.plot(x_data, params['data'], f"{params['color']}-", linewidth=1.5)
            plt.plot(x_data, np.ones_like(x_data) * params['mean'], 'r--', linewidth=1)
            plt.title(params['title'])
            plt.xlabel('Time Frame')
            plt.ylabel('Score')
            plt.ylim(-0.1, 1.1)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.text(0.05, 0.05, f"Mean: {params['mean']:.3f}",
                     transform=plt.gca().transAxes, fontsize=10,
                     bbox=dict(boxstyle='round', alpha=0.1))

        plt.tight_layout()
        plt.suptitle(f'Joint Motion Quality Metrics: {mode}', y=1.02, fontsize=14)
        plt.savefig(os.path.join(mode_dir, f'basic_scores{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_joint_probability_plots(self, mode, joint_probs, app_instance=None):
        """
        Save joint probability plots with scientific formatting.

        Args:
            mode (str): Joint mode name
            joint_probs (dict): Dictionary mapping joint types to probability data
            app_instance: JointAnalysisApp instance to get parameters from
        """
        # Create mode-specific directory
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # Get file name suffix
        suffix = self._get_filename_suffix(app_instance)

        # Define consistent colors and styles for joint types
        colors = {
            'prismatic': '#1f77b4',  # blue
            'planar': '#2ca02c',  # green
            'revolute': '#d62728',  # red
            'screw': '#9467bd',  # purple
            'ball': '#17becf'  # cyan
        }

        # Calculate final probabilities for display
        final_probs = {}
        for jt_name, probs in joint_probs.items():
            if len(probs) > 0:
                final_probs[jt_name] = probs[-1]

        # Sort joint types by final probability
        sorted_joints = sorted(final_probs.items(), key=lambda x: x[1], reverse=True)

        # Plot joint probabilities with scientific formatting
        plt.figure(figsize=(12, 8))

        # Get common x axis
        max_len = 0
        for jt_name, probs in joint_probs.items():
            max_len = max(max_len, len(probs))

        x_data = list(range(max_len))

        legend_handles = []

        for jt_name, probs in joint_probs.items():
            if len(probs) > 0:
                # Pad with last value if needed
                padded_probs = probs + [probs[-1]] * (max_len - len(probs)) if len(probs) < max_len else probs
                line, = plt.plot(x_data, padded_probs, color=colors.get(jt_name, "k"),
                                 linewidth=2, label=f"{jt_name.capitalize()} ({padded_probs[-1]:.3f})")
                legend_handles.append(line)

        plt.title(f'Joint Type Probabilities: {mode}')
        plt.xlabel('Time Frame')
        plt.ylabel('Probability')
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add a text box with the most probable joint type
        if sorted_joints:
            best_joint, best_prob = sorted_joints[0]
            plt.text(0.05, 0.05, f"Best Joint: {best_joint.capitalize()}\nProbability: {best_prob:.3f}",
                     transform=plt.gca().transAxes, fontsize=12,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.legend(handles=legend_handles, loc='upper right')
        plt.savefig(os.path.join(mode_dir, f'joint_probabilities{suffix}.png'), dpi=300)
        plt.close()

    def save_error_plots(self, mode, position_error_data, angular_error_data, app_instance=None):
        """
        Save error plots with proper scientific formatting.

        Args:
            mode (str): Joint mode name
            position_error_data (list): Position error data (in meters)
            angular_error_data (list): Angular error data (in radians)
            app_instance: JointAnalysisApp instance to get parameters from
        """
        # Create mode-specific directory
        mode_dir = os.path.join(self.output_dir, mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # Get file name suffix
        suffix = self._get_filename_suffix(app_instance)

        # Convert units: m to mm for position error
        position_error_mm = np.array(position_error_data) * 1000.0

        # Calculate mean errors for display
        mean_pos_err_mm = np.mean(position_error_mm) if len(position_error_mm) > 0 else 0.0
        mean_ang_err_rad = np.mean(angular_error_data) if len(angular_error_data) > 0 else 0.0
        mean_ang_err_deg = np.degrees(mean_ang_err_rad)

        # Draw error plots with scientific formatting
        plt.figure(figsize=(12, 5))
        x_data = list(range(len(position_error_data)))

        # Position error plot (in mm)
        plt.subplot(1, 2, 1)
        plt.plot(x_data, position_error_mm, 'b-', linewidth=1.5)
        plt.plot(x_data, np.ones_like(x_data) * mean_pos_err_mm, 'r--', linewidth=1)
        plt.title('Position Error')
        plt.xlabel('Time Frame')
        plt.ylabel('Error (mm)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.text(0.05, 0.95, f"Mean Error: {mean_pos_err_mm:.2f} mm",
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

        # Angular error plot (in rad, with degree conversion shown)
        plt.subplot(1, 2, 2)
        plt.plot(x_data, angular_error_data, 'r-', linewidth=1.5)
        plt.plot(x_data, np.ones_like(x_data) * mean_ang_err_rad, 'r--', linewidth=1)
        plt.title('Angular Error')
        plt.xlabel('Time Frame')
        plt.ylabel('Error (rad)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.text(0.05, 0.95, f"Mean Error: {mean_ang_err_rad:.3f} rad ({mean_ang_err_deg:.2f}°)",
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

        # Overall formatting
        plt.tight_layout()
        plt.suptitle(f'Error Metrics for {mode}', y=1.05, fontsize=14)

        # Save with higher resolution for paper quality
        plt.savefig(os.path.join(mode_dir, f'error_metrics{suffix}.png'), dpi=300, bbox_inches='tight')
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