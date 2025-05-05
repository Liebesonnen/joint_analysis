"""
Main application for joint analysis.
"""
import dearpygui.dearpygui as dpg
import os
import time
import numpy as np
import threading
import polyscope as ps
import polyscope.imgui as psim
from typing import Dict, List, Optional, Callable, Any, Union

from .core.geometry import generate_ball_joint_points, generate_hollow_cylinder
from .core.joint_estimation import compute_joint_info_all_types
from .viz.polyscope_viz import PolyscopeVisualizer
from .viz.gui import JointAnalysisGUI
from .synthetic.data_generator import SyntheticJointGenerator
from .data.data_loader import load_numpy_sequence, load_pytorch_data, load_real_datasets
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from .core.geometry import rotate_points_y, rotate_points, translate_points, apply_screw_motion, rotate_points_xyz
from .core.scoring import compute_velocity_angular_one_step_3d, compute_position_average_3d
import os
from .viz.plot_saver import PlotSaver
import torch
class JointAnalysisApp:
    """
    Main application for joint analysis.
    """

    def __init__(self, output_dir: str = "exported_pointclouds"):
        """
        Initialize the application.

        Args:
            output_dir (str): Directory to save output files
        """
        self.use_gui = True
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create plots directory
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

        # Initialize visualizers
        self.ps_viz = PolyscopeVisualizer()
        self.gui = JointAnalysisGUI(output_dir=self.plots_dir)  # Pass plots directory to GUI

        # Create a PlotSaver instance
        self.plot_saver = PlotSaver(output_dir=self.plots_dir)
        # Define available modes
        self.modes = [
            "Prismatic Door", "Prismatic Door 2", "Prismatic Door 3",
            "Revolute Door", "Revolute Door 2", "Revolute Door 3",
            "Planar Mouse", "Planar Mouse 2", "Planar Mouse 3",
            "Ball Joint", "Ball Joint 2", "Ball Joint 3",
            "Screw Joint", "Screw Joint 2", "Screw Joint 3"
        ]

        # Try to load real datasets
        self.real_data = load_real_datasets()
        for name in self.real_data:
            self.modes.append(name)

        # Set up GUI
        self.gui.setup_modes(self.modes)
        self.gui.set_result_callback(self._handle_gui_result)

        # Initialize synthetic data generator
        self.synth_gen = SyntheticJointGenerator(output_dir=os.path.join(output_dir, "synthetic"))

        # Current state
        self.current_mode = "Prismatic Door"
        self.previous_mode = None
        self.current_best_joint = "Unknown"
        self.current_joint_params = None
        self.running = False
        self.frame_count_per_mode = {m: 0 for m in self.modes}
        self.total_frames_per_mode = {m: 50 for m in self.modes}

        # For real datasets, set the actual number of frames
        for name, data in self.real_data.items():
            if data is not None and len(data) > 0:
                self.total_frames_per_mode[name] = data.shape[0]

        # Analysis parameters
        self.noise_sigma = 0.000
        self.col_sigma = 0.2
        self.col_order = 4.0
        self.cop_sigma = 0.2
        self.cop_order = 4.0
        self.rad_sigma = 0.2
        self.rad_order = 4.0
        self.zp_sigma = 0.2
        self.zp_order = 4.0
        self.prob_sigma = 0.2
        self.prob_order = 4.0
        self.neighbor_k = 10
        self.use_savgol_filter = False
        self.savgol_window_length = 10
        self.savgol_polyorder = 2
        self.use_multi_frame_fit = False
        self.multi_frame_radius = 20
        self.prismatic_sigma = 0.08
        self.prismatic_order = 5.0
        self.planar_sigma = 0.12
        self.planar_order = 4.0
        self.revolute_sigma = 0.08
        self.revolute_order = 5.0
        self.screw_sigma = 0.15
        self.screw_order = 4.0
        self.ball_sigma = 0.12
        self.ball_order = 4.0
        self.auto_save_plots = False
        # Ground truth data
        self._init_ground_truth()

    def _init_ground_truth(self) -> None:
        """Initialize ground truth data for synthetic joints."""
        # Prismatic door ground truth
        self.gt_data = {}

        # Prismatic doors
        self.gt_data["Prismatic Door"] = {
            "type": "prismatic",
            "params": {
                "axis": np.array([1., 0., 0.]),
                "origin": np.array([0., 0., 0.]),
                "motion_limit": (0., 5.0)
            }
        }

        self.gt_data["Prismatic Door 2"] = {
            "type": "prismatic",
            "params": {
                "axis": np.array([0., 1., 0.]),
                "origin": np.array([1.5, 0., 0.]),
                "motion_limit": (0., 4.0)
            }
        }

        self.gt_data["Prismatic Door 3"] = {
            "type": "prismatic",
            "params": {
                "axis": np.array([0., 0., 1.]),
                "origin": np.array([-1., 1., 0.]),
                "motion_limit": (0., 3.0)
            }
        }

        # Revolute doors
        self.gt_data["Revolute Door"] = {
            "type": "revolute",
            "params": {
                "axis": np.array([0., 1., 0.]),
                "origin": np.array([1., 1.5, 0.]),
                "motion_limit": (-np.pi / 4, np.pi / 4)
            }
        }

        self.gt_data["Revolute Door 2"] = {
            "type": "revolute",
            "params": {
                "axis": np.array([1., 0., 0.]),
                "origin": np.array([0.5, 2.0, -1.0]),
                "motion_limit": (-np.pi / 6, np.pi / 3)
            }
        }

        self.gt_data["Revolute Door 3"] = {
            "type": "revolute",
            "params": {
                "axis": np.array([1., 1., 0.]) / np.sqrt(2),
                "origin": np.array([2.0, 1.0, 1.0]),
                "motion_limit": (0., np.pi / 2)
            }
        }

        # Planar mice
        self.gt_data["Planar Mouse"] = {
            "type": "planar",
            "params": {
                "normal": np.array([0., 1., 0.]),
                "motion_limit": (-1.0, 1.0)
            }
        }

        self.gt_data["Planar Mouse 2"] = {
            "type": "planar",
            "params": {
                "normal": np.array([0., 1., 0.]),
                "motion_limit": (-1.0, 1.0)
            }
        }

        self.gt_data["Planar Mouse 3"] = {
            "type": "planar",
            "params": {
                "normal": np.array([0., 1., 0.]),
                "motion_limit": (-1.0, 1.0)
            }
        }

        # Ball joints
        self.gt_data["Ball Joint"] = {
            "type": "ball",
            "params": {
                "center": np.array([0., 0., 0.]),
                "motion_limit": (0., np.pi / 2, 0.)
            }
        }

        self.gt_data["Ball Joint 2"] = {
            "type": "ball",
            "params": {
                "center": np.array([1., 0., 0.]),
                "motion_limit": (0., np.pi / 2, 0.)
            }
        }

        self.gt_data["Ball Joint 3"] = {
            "type": "ball",
            "params": {
                "center": np.array([1., 1., 0.]),
                "motion_limit": (0., np.pi / 2, 0.)
            }
        }

        # Screw joints
        self.gt_data["Screw Joint"] = {
            "type": "screw",
            "params": {
                "axis": np.array([0., 1., 0.]),
                "origin": np.array([0., 0., 0.]),
                "pitch": 0.5,
                "motion_limit": (0., 2 * np.pi)
            }
        }

        self.gt_data["Screw Joint 2"] = {
            "type": "screw",
            "params": {
                "axis": np.array([1., 0., 0.]),
                "origin": np.array([1., 0., 0.]),
                "pitch": 0.8,
                "motion_limit": (0., 2 * np.pi)
            }
        }

        self.gt_data["Screw Joint 3"] = {
            "type": "screw",
            "params": {
                "axis": np.array([1., 1., 0.]) / np.sqrt(2),
                "origin": np.array([-1., 0., 1.]),
                "pitch": 0.6,
                "motion_limit": (0., 2 * np.pi)
            }
        }

    def save_plots(self):
        """Save plots for the current mode."""
        if self.current_mode and self.gui:
            # Extract data for the current mode
            mode = self.current_mode

            # Check if there's data to save
            if (mode in self.gui.velocity_profile and
                    len(self.gui.velocity_profile[mode]) > 0):

                # Save velocity plots
                self.plot_saver.save_velocity_plots(
                    mode,
                    self.gui.velocity_profile[mode],
                    self.gui.angular_velocity_profile[mode],
                    self
                )

                # Save basic scores plots
                self.plot_saver.save_basic_scores_plots(
                    mode,
                    self.gui.col_score_profile[mode],
                    self.gui.cop_score_profile[mode],
                    self.gui.rad_score_profile[mode],
                    self.gui.zp_score_profile[mode],
                    self
                )

                # Save joint probability plots
                self.plot_saver.save_joint_probability_plots(
                    mode,
                    self.gui.joint_prob_profile[mode],
                    self
                )

                # Save error plots
                self.plot_saver.save_error_plots(
                    mode,
                    self.gui.position_error_profile[mode],
                    self.gui.angular_error_profile[mode],
                    self
                )

                print(f"Plots saved to: {os.path.join(self.plots_dir, mode.replace(' ', '_'))}")
            else:
                print(f"No data available to save plots for mode: {mode}")
    def _handle_gui_result(self, result: str) -> None:
        """
        Handle results from the GUI.

        Args:
            result (str): Result message
        """
        if result == "clear_plots":
            # Clear point cloud history and reset frame counts
            for mode in self.modes:
                self.frame_count_per_mode[mode] = 0

    def _register_point_clouds(self) -> None:
        """Register all synthetic and real point clouds with the visualizer."""
        # Generate synthetic point clouds
        # Prismatic doors
        door_width, door_height, door_thickness = 2.0, 3.0, 0.2
        prismatic_door_points = np.random.rand(self.synth_gen.num_points, 3)
        prismatic_door_points[:, 0] = prismatic_door_points[:, 0] * door_width - 0.5 * door_width
        prismatic_door_points[:, 1] = prismatic_door_points[:, 1] * door_height
        prismatic_door_points[:, 2] = prismatic_door_points[:, 2] * door_thickness - 0.5 * door_thickness

        self.ps_viz.register_point_cloud("Prismatic Door", prismatic_door_points)
        self.ps_viz.register_point_cloud("Prismatic Door 2", prismatic_door_points + np.array([1.5, 0., 0.]),
                                         enabled=False)
        self.ps_viz.register_point_cloud("Prismatic Door 3", prismatic_door_points + np.array([-1., 1., 0.]),
                                         enabled=False)

        # Revolute doors (use same base geometry as prismatic doors)
        self.ps_viz.register_point_cloud("Revolute Door", prismatic_door_points, enabled=False)
        self.ps_viz.register_point_cloud("Revolute Door 2", prismatic_door_points + np.array([0., 0., -1.]),
                                         enabled=False)
        self.ps_viz.register_point_cloud("Revolute Door 3", prismatic_door_points + np.array([0., -0.5, 1.0]),
                                         enabled=False)

        # Planar mouse
        mouse_length, mouse_width, mouse_height = 1.0, 0.6, 0.3
        mouse_points = np.zeros((self.synth_gen.num_points, 3))
        mouse_points[:, 0] = np.random.rand(self.synth_gen.num_points) * mouse_length - 0.5 * mouse_length
        mouse_points[:, 2] = np.random.rand(self.synth_gen.num_points) * mouse_width - 0.5 * mouse_width
        mouse_points[:, 1] = np.random.rand(self.synth_gen.num_points) * mouse_height

        self.ps_viz.register_point_cloud("Planar Mouse", mouse_points, enabled=False)
        self.ps_viz.register_point_cloud("Planar Mouse 2", mouse_points + np.array([1., 0., 1.]), enabled=False)
        self.ps_viz.register_point_cloud("Planar Mouse 3", mouse_points + np.array([-1., 0., 1.]), enabled=False)

        # Ball joint
        sphere_radius = 0.3
        rod_length = sphere_radius * 10.0
        rod_radius = 0.05
        ball_joint_points = generate_ball_joint_points(
            np.array([0., 0., 0.]), sphere_radius, rod_length, rod_radius, 250, 250
        )

        self.ps_viz.register_point_cloud("Ball Joint", ball_joint_points, enabled=False)
        self.ps_viz.register_point_cloud("Ball Joint 2",
                                         generate_ball_joint_points(
                                             np.array([1., 0., 0.]), sphere_radius, rod_length, rod_radius, 250, 250
                                         ),
                                         enabled=False)
        self.ps_viz.register_point_cloud("Ball Joint 3",
                                         generate_ball_joint_points(
                                             np.array([1., 1., 0.]), sphere_radius, rod_length, rod_radius, 250, 250
                                         ),
                                         enabled=False)

        # Screw joint
        screw_joint_points = generate_hollow_cylinder(
            radius=0.4, height=0.2, thickness=0.05,
            num_points=500, cap_position="top", cap_points_ratio=0.2
        )

        self.ps_viz.register_point_cloud("Screw Joint", screw_joint_points, enabled=False)
        self.ps_viz.register_point_cloud("Screw Joint 2", screw_joint_points + np.array([1., 0., 0.]), enabled=False)
        self.ps_viz.register_point_cloud("Screw Joint 3", screw_joint_points + np.array([-1., 0., 1.]), enabled=False)

        # Register real datasets if available
        for name, data in self.real_data.items():
            if data is not None and len(data) > 0:
                self.ps_viz.register_point_cloud(name, data[0], enabled=False)

    def _compute_error_for_mode(self, mode: str, param_dict: Dict[str, Dict[str, Any]]) -> Tuple[float, float]:
        """
        Compute position and angular errors for a given mode compared to ground truth.

        Args:
            mode (str): Mode name
            param_dict (Dict[str, Dict[str, Any]]): Dictionary of joint parameters

        Returns:
            Tuple[float, float]: Position error, angular error
        """
        # For real data, we don't have ground truth
        if mode in self.real_data:
            return 0.0, 0.0

        # Check if we have ground truth for this mode
        if mode not in self.gt_data:
            return 0.0, 0.0

        gt_info = self.gt_data[mode]
        gt_type = gt_info["type"]
        gt_params = gt_info["params"]

        # Check if estimated parameters include this joint type
        if gt_type not in param_dict:
            return 1.0, 1.0

        info = param_dict[gt_type]

        # Handle different joint types
        if gt_type == "prismatic":
            # For prismatic joints, compare axes
            gt_axis = gt_params["axis"]
            est_axis = info["axis"]

            # Normalize axes
            gt_axis_norm = gt_axis / np.linalg.norm(gt_axis)
            est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)

            # Angular error - angle between axes
            dotv = np.dot(est_axis_norm, gt_axis_norm)
            angle_err = abs(1.0 - abs(dotv))

            # Position error - for prismatic joint, use origin distance
            pos_err = 0

            return pos_err, angle_err

        elif gt_type == "revolute":
            # For revolute joints, compare axes and origins
            gt_axis = gt_params["axis"]
            gt_origin = gt_params["origin"]
            est_axis = info["axis"]
            est_origin = info["origin"]

            # Normalize axes
            gt_axis_norm = gt_axis / np.linalg.norm(gt_axis)
            est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)

            # Angular error - angle between axes
            dotv = np.dot(est_axis_norm, gt_axis_norm)
            angle_err = abs(1.0 - abs(dotv))

            # Position error - distance between axis lines
            d12 = gt_origin - est_origin
            cross_ = np.cross(est_axis_norm, gt_axis_norm)
            cross_norm = np.linalg.norm(cross_)

            if cross_norm < 1e-9:
                # Parallel axes, use perpendicular distance
                line_dist = np.linalg.norm(np.cross(d12, gt_axis_norm))
            else:
                # Non-parallel axes, use distance between closest points
                n_ = cross_ / cross_norm
                line_dist = abs(np.dot(d12, n_))

            # Normalize position error (arbitrary scale)
            pos_err = line_dist / 2.0

            return pos_err, angle_err

        elif gt_type == "planar":
            # For planar joints, compare normals
            gt_normal = gt_params["normal"]
            est_normal = info["normal"]

            # Normalize normals
            gt_normal_norm = gt_normal / np.linalg.norm(gt_normal)
            est_normal_norm = est_normal / (np.linalg.norm(est_normal) + 1e-9)

            # Angular error - angle between normals
            dotv = np.dot(est_normal_norm, gt_normal_norm)
            angle_err = abs(1.0 - abs(dotv))

            # Position error - plane distance to origin
            # This is simplified and might not be the best metric
            pos_err = 0

            return pos_err, angle_err

        elif gt_type == "ball":
            # For ball joints, compare centers
            gt_center = gt_params["center"]
            est_center = info["center"]

            # Position error - distance between centers
            pos_err = np.linalg.norm(est_center - gt_center) / 2.0

            # No meaningful angular error for ball joints
            angle_err = 0.0

            return pos_err, angle_err

        elif gt_type == "screw":
            # For screw joints, compare axes, origins, and pitches
            gt_axis = gt_params["axis"]
            gt_origin = gt_params["origin"]
            gt_pitch = gt_params["pitch"]

            est_axis = info["axis"]
            est_origin = info["origin"]
            est_pitch = info.get("pitch", 0.0)

            # Normalize axes
            gt_axis_norm = gt_axis / np.linalg.norm(gt_axis)
            est_axis_norm = est_axis / (np.linalg.norm(est_axis) + 1e-9)

            # Angular error - angle between axes
            dotv = np.dot(est_axis_norm, gt_axis_norm)
            angle_err = abs(1.0 - abs(dotv))

            # Position error - distance between axis lines
            d12 = gt_origin - est_origin
            cross_ = np.cross(est_axis_norm, gt_axis_norm)
            cross_norm = np.linalg.norm(cross_)

            if cross_norm < 1e-9:
                # Parallel axes, use perpendicular distance
                line_dist = np.linalg.norm(np.cross(d12, gt_axis_norm))
            else:
                # Non-parallel axes, use distance between closest points
                n_ = cross_ / cross_norm
                line_dist = abs(np.dot(d12, n_))

            # Could also incorporate pitch error here
            pitch_err = abs(est_pitch - gt_pitch) / max(abs(gt_pitch), 1e-6)

            # Combine position and pitch errors
            pos_err = (line_dist / 2.0 + pitch_err * 0.5) / 1.5

            return pos_err, angle_err

        return 0.0, 0.0

    def update_motion_and_store(self, mode: str) -> None:
        """
        Update the motion for the current mode and store the frame.

        Args:
            mode (str): Mode name
        """
        fidx = self.frame_count_per_mode[mode]
        limit = self.total_frames_per_mode[mode]

        if fidx >= limit:
            return

        prev_points = None
        current_points = None

        # Handle real datasets
        if mode in self.real_data and self.real_data[mode] is not None:
            data = self.real_data[mode]
            if fidx < data.shape[0]:
                prev_positions = data[fidx - 1] if fidx > 0 else None
                current_positions = data[fidx]
                self.ps_viz.update_point_cloud(mode, current_positions)
                self.ps_viz.store_frame(current_positions)
                self.ps_viz.highlight_point_differences(mode, current_positions, prev_positions)
                prev_points = prev_positions
                current_points = current_positions

        # Handle synthetic datasets
        else:
            # Get the original points from the generator
            if mode == "Prismatic Door":
                # Prismatic Door 1
                prev_points = self.synth_gen.prismatic_door_points.copy() if fidx > 0 else None
                pos = (fidx / (limit - 1)) * 5.0
                current_points = self.synth_gen.prismatic_door_points.copy()
                current_points = translate_points(current_points, pos, np.array([1., 0., 0.]))

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Prismatic Door 2":
                prev_points = self.synth_gen.prismatic_door_points_2.copy() if fidx > 0 else None
                pos = (fidx / (limit - 1)) * 4.0
                current_points = self.synth_gen.prismatic_door_points_2.copy()
                current_points = translate_points(current_points, pos, np.array([0., 1., 0.]))

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Prismatic Door 3":
                prev_points = self.synth_gen.prismatic_door_points_3.copy() if fidx > 0 else None
                pos = (fidx / (limit - 1)) * 3.0
                current_points = self.synth_gen.prismatic_door_points_3.copy()
                current_points = translate_points(current_points, pos, np.array([0., 0., 1.]))

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Revolute Door":
                prev_points = self.synth_gen.revolute_door_points.copy() if fidx > 0 else None
                angle_min = -np.pi / 4
                angle_max = np.pi / 4
                angle = angle_min + (angle_max - angle_min) * (fidx / (limit - 1))
                current_points = self.synth_gen.revolute_door_points.copy()
                current_points = rotate_points(
                    current_points, angle, np.array([0., 1., 0.]), np.array([1., 1.5, 0.])
                )

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Revolute Door 2":
                prev_points = self.synth_gen.revolute_door_points_2.copy() if fidx > 0 else None
                angle_min = -np.pi / 6
                angle_max = np.pi / 3
                angle = angle_min + (angle_max - angle_min) * (fidx / (limit - 1))
                current_points = self.synth_gen.revolute_door_points_2.copy()
                current_points = rotate_points(
                    current_points, angle, np.array([1., 0., 0.]), np.array([0.5, 2.0, -1.0])
                )

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Revolute Door 3":
                prev_points = self.synth_gen.revolute_door_points_3.copy() if fidx > 0 else None
                angle_min = 0.0
                angle_max = np.pi / 2
                angle = angle_min + (angle_max - angle_min) * (fidx / (limit - 1))
                current_points = self.synth_gen.revolute_door_points_3.copy()
                current_points = rotate_points(
                    current_points, angle, np.array([1., 1., 0.]) / np.sqrt(2), np.array([2.0, 1.0, 1.0])
                )

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Planar Mouse":
                prev_points = self.synth_gen.planar_mouse_points.copy() if fidx > 0 else None
                current_points = self.synth_gen.planar_mouse_points.copy()

                if fidx < limit // 2:
                    # First half: translation
                    alpha = fidx / (limit // 2 - 1) if limit // 2 > 1 else 0
                    tx = -1.0 + alpha * (0.0 - (-1.0))
                    tz = 1.0 + alpha * (0.0 - 1.0)
                    ry = 0.0
                    current_points += np.array([tx, 0., tz])
                    current_points = rotate_points_y(current_points, ry, [0., 0., 0.])
                else:
                    # Second half: rotation and translation
                    alpha = (fidx - limit // 2) / (limit - limit // 2 - 1) if (limit - limit // 2 - 1) > 0 else 0
                    ry = np.radians(40.0) * alpha
                    tx = alpha * 1.0
                    tz = alpha * (-1.0)
                    current_points += np.array([tx, 0., tz])
                    current_points = rotate_points_y(current_points, ry, [0., 0., 0.])

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Planar Mouse 2":
                prev_points = self.synth_gen.planar_mouse_points_2.copy() if fidx > 0 else None

                if fidx < limit // 2:
                    # First half: translation
                    alpha = fidx / (limit // 2 - 1) if limit // 2 > 1 else 0
                    dy = alpha * 1.0
                    dz = alpha * 1.0
                    current_points = self.synth_gen.planar_mouse_points_2.copy()
                    current_points += np.array([0., dy, dz])
                else:
                    # Second half: rotation and translation
                    alpha = (fidx - limit // 2) / (limit - limit // 2 - 1) if (limit - limit // 2 - 1) > 0 else 0
                    mp = self.synth_gen.planar_mouse_points_2.copy()
                    mp += np.array([0., 1.0, 1.0])
                    rx = np.radians(30.0) * alpha
                    current_points = rotate_points(
                        mp, rx, np.array([1.0, 0., 0.]), np.array([0., 1.0, 1.0])
                    )

                    dy = alpha * 1.0
                    dz = alpha * 0.5
                    current_points += np.array([0., dy, dz])

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Planar Mouse 3":
                prev_points = self.synth_gen.planar_mouse_points_3.copy() if fidx > 0 else None

                if fidx < limit // 2:
                    # First half: translation
                    alpha = fidx / (limit // 2 - 1) if limit // 2 > 1 else 0
                    dx = alpha * 1.0
                    dz = alpha * 0.5
                    current_points = self.synth_gen.planar_mouse_points_3.copy()
                    current_points += np.array([dx, 0., dz])
                else:
                    # Second half: rotation and translation
                    alpha = (fidx - limit // 2) / (limit - limit // 2 - 1) if (limit - limit // 2 - 1) > 0 else 0
                    mp = self.synth_gen.planar_mouse_points_3.copy()
                    mp += np.array([1.0, 0., 0.5])
                    ry = np.radians(30.0) * alpha
                    current_points = rotate_points(
                        mp, ry, np.array([0., 1., 0.]), np.array([1.0, 0., 0.5])
                    )

                    dx = alpha * 1.0
                    dz = alpha * 0.5
                    current_points += np.array([dx, 0., dz])

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Ball Joint":
                prev_points = self.synth_gen.ball_joint_points.copy() if fidx > 0 else None

                if fidx < limit // 3:
                    alpha = fidx / (limit // 3 - 1) if limit // 3 > 1 else 0
                    ax = np.radians(60.0) * alpha
                    ay = 0.0
                    az = 0.0
                elif fidx < 2 * limit // 3:
                    alpha = (fidx - limit // 3) / (limit // 3 - 1) if limit // 3 > 1 else 0
                    ax = np.radians(60.0)
                    ay = np.radians(40.0) * alpha
                    az = 0.0
                else:
                    alpha = (fidx - 2 * limit // 3) / (limit - 2 * limit // 3 - 1) if (
                                                                                                  limit - 2 * limit // 3 - 1) > 0 else 0
                    ax = np.radians(60.0)
                    ay = np.radians(40.0)
                    az = np.radians(70.0) * alpha

                current_points = self.synth_gen.ball_joint_points.copy()
                current_points = rotate_points_xyz(
                    current_points, ax, ay, az, np.array([0., 0., 0.])
                )

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Ball Joint 2":
                prev_points = self.synth_gen.ball_joint_points_2.copy() if fidx > 0 else None

                if fidx < limit // 2:
                    alpha = fidx / (limit // 2 - 1) if limit // 2 > 1 else 0
                    rx = np.radians(50.0) * alpha
                    ry = np.radians(10.0) * alpha
                    rz = 0.0
                else:
                    alpha = (fidx - limit // 2) / (limit - limit // 2 - 1) if (limit - limit // 2 - 1) > 0 else 0
                    rx = np.radians(50.0)
                    ry = np.radians(10.0)
                    rz = np.radians(45.0) * alpha

                current_points = self.synth_gen.ball_joint_points_2.copy()
                current_points = rotate_points_xyz(
                    current_points, rx, ry, rz, np.array([1., 0., 0.])
                )

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Ball Joint 3":
                prev_points = self.synth_gen.ball_joint_points_3.copy() if fidx > 0 else None

                if fidx < limit // 3:
                    alpha = fidx / (limit // 3 - 1) if limit // 3 > 1 else 0
                    ax = np.radians(30.0) * alpha
                    ay = 0.0
                    az = 0.0
                elif fidx < 2 * limit // 3:
                    alpha = (fidx - limit // 3) / (limit // 3 - 1) if limit // 3 > 1 else 0
                    ax = np.radians(30.0)
                    ay = np.radians(50.0) * alpha
                    az = 0.0
                else:
                    alpha = (fidx - 2 * limit // 3) / (limit - 2 * limit // 3 - 1) if (
                                                                                                  limit - 2 * limit // 3 - 1) > 0 else 0
                    ax = np.radians(30.0)
                    ay = np.radians(50.0)
                    az = np.radians(80.0) * alpha

                current_points = self.synth_gen.ball_joint_points_3.copy()
                current_points = rotate_points_xyz(
                    current_points, ax, ay, az, np.array([1., 1., 0.])
                )

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Screw Joint":
                prev_points = self.synth_gen.screw_joint_points.copy() if fidx > 0 else None
                angle = 2 * np.pi * (fidx / (limit - 1))
                current_points = self.synth_gen.screw_joint_points.copy()
                current_points = apply_screw_motion(
                    current_points, angle, np.array([0., 1., 0.]), np.array([0., 0., 0.]), 0.5
                )

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Screw Joint 2":
                prev_points = self.synth_gen.screw_joint_points_2.copy() if fidx > 0 else None
                angle = 2 * np.pi * (fidx / (limit - 1))
                current_points = self.synth_gen.screw_joint_points_2.copy()
                current_points = apply_screw_motion(
                    current_points, angle, np.array([1., 0., 0.]), np.array([1., 0., 0.]), 0.8
                )

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

            elif mode == "Screw Joint 3":
                prev_points = self.synth_gen.screw_joint_points_3.copy() if fidx > 0 else None
                angle = 2 * np.pi * (fidx / (limit - 1))
                current_points = self.synth_gen.screw_joint_points_3.copy()
                current_points = apply_screw_motion(
                    current_points, angle, np.array([1., 1., 0.]) / np.sqrt(2), np.array([-1., 0., 1.]), 0.6
                )

                if self.noise_sigma > 1e-6:
                    current_points += np.random.normal(0, self.noise_sigma, current_points.shape)

                self.ps_viz.update_point_cloud(mode, current_points)
                self.ps_viz.store_frame(current_points)
                self.ps_viz.highlight_point_differences(mode, current_points, prev_points)

        # Increment frame counter
        self.frame_count_per_mode[mode] += 1
        if fidx >= 0 and current_points is not None and prev_points is not None:
            # Convert NumPy arrays to PyTorch tensors
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            prev_points_tensor = torch.tensor(prev_points, dtype=torch.float32, device=device)
            current_points_tensor = torch.tensor(current_points, dtype=torch.float32, device=device)

            # Call the original function with PyTorch tensors
            v_meas_3d, w_meas_3d = compute_velocity_angular_one_step_3d(
                prev_points_tensor, current_points_tensor, dt=0.1, num_neighbors=self.neighbor_k
            )

            # The rest of your code remains the same
            vel_mag = np.linalg.norm(v_meas_3d)
            ang_mag = np.linalg.norm(w_meas_3d)

            # Update GUI data - still showing linearly increasing values
            if self.use_gui and self.gui:
                self.gui.add_velocity_data(mode, vel_mag, ang_mag)
            # 运行关节类型估计（如果我们有序列）
            sequence = self.ps_viz.get_point_cloud_sequence()
            if sequence is not None and sequence.shape[0] >= 2:
                # 运行关节估计
                param_dict, best_type, scores_info = compute_joint_info_all_types(
                    sequence,
                    neighbor_k=self.neighbor_k,
                    col_sigma=self.col_sigma, col_order=self.col_order,
                    cop_sigma=self.cop_sigma, cop_order=self.cop_order,
                    rad_sigma=self.rad_sigma, rad_order=self.rad_order,
                    zp_sigma=self.zp_sigma, zp_order=self.zp_order,
                    prob_sigma=self.prob_sigma, prob_order=self.prob_order,
                    prismatic_sigma=self.prismatic_sigma, prismatic_order=self.prismatic_order,
                    planar_sigma=self.planar_sigma, planar_order=self.planar_order,
                    revolute_sigma=self.revolute_sigma, revolute_order=self.revolute_order,
                    screw_sigma=self.screw_sigma, screw_order=self.screw_order,
                    ball_sigma=self.ball_sigma, ball_order=self.ball_order,
                    use_savgol=self.use_savgol_filter,
                    savgol_window=self.savgol_window_length,
                    savgol_poly=self.savgol_polyorder,
                    use_multi_frame=self.use_multi_frame_fit,
                    multi_frame_window_radius=self.multi_frame_radius
                )

                # Update current best joint
                self.current_best_joint = best_type

                if best_type in param_dict:
                    self.current_joint_params = param_dict[best_type]

                    # Update joint visualization
                    self.ps_viz.set_joint_estimation_result(best_type, param_dict[best_type])

                    # Show ground truth for comparison (for synthetic data)
                    if mode in self.gt_data:
                        gt_info = self.gt_data[mode]
                        self.ps_viz.show_ground_truth(gt_info["type"], gt_info["params"])

                # Compute errors compared to ground truth
                pos_err, ang_err = self._compute_error_for_mode(mode, param_dict)

                # Update GUI with analysis results
                if self.use_gui and self.gui:
                    self.gui.add_velocity_data(mode, vel_mag, ang_mag)
                if scores_info is not None:
                    # Prepare analysis data for GUI
                    analysis_data = {
                        "basic_score_avg": scores_info["basic_score_avg"],
                        "joint_probs": scores_info["joint_probs"],
                        "position_error": pos_err,
                        "angular_error": ang_err
                    }

                    self.gui.add_analysis_results(mode, analysis_data)
        else:
            # First frame, zero velocity
            self.gui.add_velocity_data(mode, 0.0, 0.0)

    def polyscope_callback(self) -> None:
        """Callback function for the Polyscope UI."""
        # Mode selection
        changed = psim.BeginCombo("Object Mode", self.current_mode)
        if changed:
            for mode in self.modes:
                _, selected = psim.Selectable(mode, self.current_mode == mode)
                if selected and mode != self.current_mode:
                    # Remove joint visualization
                    self.ps_viz.remove_joint_visualization()

                    # Set current mode
                    self.previous_mode = self.current_mode
                    self.current_mode = mode
                    self.ps_viz.set_current_mode(mode)

                    # Reset frame index
                    self.frame_count_per_mode[mode] = 0

                    # Show ground truth if available
                    if mode in self.gt_data:
                        gt_info = self.gt_data[mode]
                        self.ps_viz.show_ground_truth(gt_info["type"], gt_info["params"])
            psim.EndCombo()

        psim.Separator()
        if psim.TreeNodeEx("Joint Type Probability Parameters", flags=psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Columns(2, "probcolumns", False)
            psim.SetColumnWidth(0, 230)

            # Prismatic joint parameters
            changed_pris_sigma, new_pris_sigma = psim.InputFloat("prismatic_sigma", self.prismatic_sigma, 0.001)
            if changed_pris_sigma:
                self.prismatic_sigma = max(1e-6, new_pris_sigma)

            changed_pris_order, new_pris_order = psim.InputFloat("prismatic_order", self.prismatic_order, 0.1)
            if changed_pris_order:
                self.prismatic_order = max(0.1, new_pris_order)

            # Planar joint parameters
            changed_plan_sigma, new_plan_sigma = psim.InputFloat("planar_sigma", self.planar_sigma, 0.001)
            if changed_plan_sigma:
                self.planar_sigma = max(1e-6, new_plan_sigma)

            changed_plan_order, new_plan_order = psim.InputFloat("planar_order", self.planar_order, 0.1)
            if changed_plan_order:
                self.planar_order = max(0.1, new_plan_order)

            psim.NextColumn()

            # Revolute joint parameters
            changed_rev_sigma, new_rev_sigma = psim.InputFloat("revolute_sigma", self.revolute_sigma, 0.001)
            if changed_rev_sigma:
                self.revolute_sigma = max(1e-6, new_rev_sigma)

            changed_rev_order, new_rev_order = psim.InputFloat("revolute_order", self.revolute_order, 0.1)
            if changed_rev_order:
                self.revolute_order = max(0.1, new_rev_order)

            # Screw joint parameters
            changed_screw_sigma, new_screw_sigma = psim.InputFloat("screw_sigma", self.screw_sigma, 0.001)
            if changed_screw_sigma:
                self.screw_sigma = max(1e-6, new_screw_sigma)

            changed_screw_order, new_screw_order = psim.InputFloat("screw_order", self.screw_order, 0.1)
            if changed_screw_order:
                self.screw_order = max(0.1, new_screw_order)

            # Ball joint parameters
            changed_ball_sigma, new_ball_sigma = psim.InputFloat("ball_sigma", self.ball_sigma, 0.001)
            if changed_ball_sigma:
                self.ball_sigma = max(1e-6, new_ball_sigma)

            changed_ball_order, new_ball_order = psim.InputFloat("ball_order", self.ball_order, 0.1)
            if changed_ball_order:
                self.ball_order = max(0.1, new_ball_order)
            psim.Columns(1)
            psim.TreePop()
        psim.Separator()
        # Parameters UI
        if psim.TreeNodeEx("Noise & Analysis Parameters", flags=psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Columns(2, "mycolumns", False)
            psim.SetColumnWidth(0, 230)

            # Neighbor K
            changed_k, new_k = psim.InputInt("Neighbor K", self.neighbor_k, 10)
            if changed_k:
                self.neighbor_k = max(1, new_k)

            # Noise sigma
            changed_noise, new_noise_sigma = psim.InputFloat("Noise Sigma", self.noise_sigma, 0.001)
            if changed_noise:
                self.noise_sigma = max(0.0, new_noise_sigma)

            # Score parameters
            changed_col_sigma, new_cs = psim.InputFloat("col_sigma", self.col_sigma, 0.001)
            if changed_col_sigma:
                self.col_sigma = max(1e-6, new_cs)

            changed_col_order, new_co = psim.InputFloat("col_order", self.col_order, 0.1)
            if changed_col_order:
                self.col_order = max(0.1, new_co)

            changed_cop_sigma, new_cops = psim.InputFloat("cop_sigma", self.cop_sigma, 0.001)
            if changed_cop_sigma:
                self.cop_sigma = max(1e-6, new_cops)

            changed_cop_order, new_copo = psim.InputFloat("cop_order", self.cop_order, 0.1)
            if changed_cop_order:
                self.cop_order = max(0.1, new_copo)

            changed_rad_sigma, new_rs = psim.InputFloat("rad_sigma", self.rad_sigma, 0.001)
            if changed_rad_sigma:
                self.rad_sigma = max(1e-6, new_rs)

            changed_rad_order, new_ro = psim.InputFloat("rad_order", self.rad_order, 0.1)
            if changed_rad_order:
                self.rad_order = max(0.1, new_ro)

            changed_zp_sigma, new_zs = psim.InputFloat("zp_sigma", self.zp_sigma, 0.001)
            if changed_zp_sigma:
                self.zp_sigma = max(1e-6, new_zs)

            changed_zp_order, new_zo = psim.InputFloat("zp_order", self.zp_order, 0.1)
            if changed_zp_order:
                self.zp_order = max(0.1, new_zo)

            changed_prob_sigma, new_ps = psim.InputFloat("prob_sigma", self.prob_sigma, 0.001)
            if changed_prob_sigma:
                self.prob_sigma = max(1e-6, new_ps)

            changed_prob_order, new_po = psim.InputFloat("prob_order", self.prob_order, 0.1)
            if changed_prob_order:
                self.prob_order = max(0.1, new_po)

            psim.NextColumn()

            # Filter parameters
            changed_sg_win, new_sg_win = psim.InputInt("SG Window", self.savgol_window_length, 1)
            if changed_sg_win:
                self.savgol_window_length = max(3, new_sg_win)

            changed_sg_poly, new_sg_poly = psim.InputInt("SG PolyOrder", self.savgol_polyorder, 1)
            if changed_sg_poly:
                self.savgol_polyorder = max(1, new_sg_poly)

            _, use_sg_filter_new = psim.Checkbox("Use SG Filter?", self.use_savgol_filter)
            if use_sg_filter_new != self.use_savgol_filter:
                self.use_savgol_filter = use_sg_filter_new

            _, use_mf_new = psim.Checkbox("Use Multi-Frame Fit?", self.use_multi_frame_fit)
            if use_mf_new != self.use_multi_frame_fit:
                self.use_multi_frame_fit = use_mf_new

            changed_mfr, new_mfr = psim.InputInt("MultiFrame Radius", self.multi_frame_radius, 1)
            if changed_mfr:
                self.multi_frame_radius = max(1, new_mfr)
            psim.Columns(1)
            psim.TreePop()

        psim.Separator()

        # Control buttons
        if psim.Button("Start"):
            self.frame_count_per_mode[self.current_mode] = 0
            self.running = True

        psim.SameLine()

        if psim.Button("Stop"):
            self.running = False

        psim.SameLine()

        if psim.Button("Save .npy"):
            filepath = self.ps_viz.save_sequence(self.output_dir)
            if filepath:
                psim.TextUnformatted(f"Saved to: {filepath}")

        psim.SameLine()

        if psim.Button("Save Plots"):
            self.save_plots()
        # Update motion when running
        if self.running:
            limit = self.total_frames_per_mode[self.current_mode]
            if self.frame_count_per_mode[self.current_mode] < limit:
                self.update_motion_and_store(self.current_mode)
            else:
                self.running = False

        # Display current joint info
        psim.Separator()

        psim.TextUnformatted(f"Current Mode: {self.current_mode}")
        psim.TextUnformatted(
            f"Frame: {self.frame_count_per_mode[self.current_mode]} / {self.total_frames_per_mode[self.current_mode]}")

        # Get sequence for analysis
        sequence = self.ps_viz.get_point_cloud_sequence()
        if sequence is not None and sequence.shape[0] >= 2:
            psim.TextUnformatted(f"Sequence Size: {sequence.shape[0]} frames, {sequence.shape[1]} points")
            psim.TextUnformatted(f"Best Joint Type: {self.current_best_joint}")

            # Display joint parameters
            if self.current_best_joint in ["planar", "ball", "screw", "prismatic",
                                           "revolute"] and self.current_joint_params is not None:
                if self.current_best_joint == "planar":
                    n_ = self.current_joint_params["normal"]
                    lim = self.current_joint_params["motion_limit"]
                    psim.TextUnformatted(f"  Normal=({n_[0]:.2f}, {n_[1]:.2f}, {n_[2]:.2f})")
                    psim.TextUnformatted(f"  MotionLimit=({lim[0]:.2f}, {lim[1]:.2f})")

                elif self.current_best_joint == "ball":
                    c_ = self.current_joint_params["center"]
                    lim = self.current_joint_params["motion_limit"]
                    psim.TextUnformatted(f"  Center=({c_[0]:.2f}, {c_[1]:.2f}, {c_[2]:.2f})")
                    psim.TextUnformatted(f"  MotionLimit=Rx:{lim[0]:.2f}, Ry:{lim[1]:.2f}, Rz:{lim[2]:.2f}")

                elif self.current_best_joint == "screw":
                    a_ = self.current_joint_params["axis"]
                    o_ = self.current_joint_params["origin"]
                    p_ = self.current_joint_params["pitch"]
                    lim = self.current_joint_params["motion_limit"]
                    psim.TextUnformatted(f"  Axis=({a_[0]:.2f}, {a_[1]:.2f}, {a_[2]:.2f}), pitch={p_:.3f}")
                    psim.TextUnformatted(f"  Origin=({o_[0]:.2f}, {o_[1]:.2f}, {o_[2]:.2f})")
                    psim.TextUnformatted(f"  MotionLimit=({lim[0]:.2f} rad, {lim[1]:.2f} rad)")

                elif self.current_best_joint == "prismatic":
                    a_ = self.current_joint_params["axis"]
                    lim = self.current_joint_params["motion_limit"]
                    psim.TextUnformatted(f"  Axis=({a_[0]:.2f}, {a_[1]:.2f}, {a_[2]:.2f})")
                    psim.TextUnformatted(f"  MotionLimit=({lim[0]:.2f}, {lim[1]:.2f})")

                elif self.current_best_joint == "revolute":
                    a_ = self.current_joint_params["axis"]
                    o_ = self.current_joint_params["origin"]
                    lim = self.current_joint_params["motion_limit"]
                    psim.TextUnformatted(f"  Axis=({a_[0]:.2f}, {a_[1]:.2f}, {a_[2]:.2f})")
                    psim.TextUnformatted(f"  Origin=({o_[0]:.2f}, {o_[1]:.2f}, {o_[2]:.2f})")
                    psim.TextUnformatted(f"  MotionLimit=({lim[0]:.2f} rad, {lim[1]:.2f} rad)")
        else:
            psim.TextUnformatted("Not enough frames to do joint classification.")

    def run(self) -> None:
        """Run the application."""
        print("Starting JointAnalysisApp...")

        # 注册点云
        print("Registering point clouds...")
        self._register_point_clouds()

        # 初始化 GUI 但不启动事件循环
        print("Starting GUI...")
        self.gui.start(width=1250, height=900)

        # 创建停止事件
        self.stop_event = threading.Event()

        # 定义 Polyscope 线程函数
        def run_polyscope():
            print("Setting up Polyscope...")
            self.ps_viz.setup_camera()
            ps.set_user_callback(self.polyscope_callback)

            print("Showing Polyscope...")
            ps.show()

            # 当 Polyscope 窗口关闭时通知主线程
            self.stop_event.set()
            print("Polyscope window closed")

        # 启动 Polyscope 线程
        ps_thread = threading.Thread(target=run_polyscope)
        ps_thread.daemon = True
        ps_thread.start()

        # 主循环
        print("Entering main loop...")
        try:
            while dpg.is_dearpygui_running() and not self.stop_event.is_set():
                # 更新 DearPyGUI
                dpg.render_dearpygui_frame()
                time.sleep(0.016)  # ~60fps
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            # 清理
            print("Cleaning up...")
            self.gui.shutdown()


def run_application(use_gui=True, output_dir="exported_pointclouds", auto_save_plots=False):
    """
    Run the joint analysis application.

    Args:
        use_gui (bool): Whether to use the GUI
        output_dir (str): Directory to save output files
        auto_save_plots (bool): Whether to automatically save plots at the end
    """
    print(f"Starting joint analysis application with GUI: {use_gui}")
    app = JointAnalysisApp(output_dir=output_dir)
    app.use_gui = use_gui

    # Store auto_save_plots setting
    app.auto_save_plots = auto_save_plots

    # Run the application
    app.run()

    # If auto_save_plots is enabled, save plots before exiting
    if auto_save_plots and app.gui:
        print("Auto-saving plots for all modes with data...")
        app.plot_saver.save_all_plots(app.gui)
        print(f"Plots saved to: {app.plots_dir}")


if __name__ == "__main__":
    run_application(use_gui=True)