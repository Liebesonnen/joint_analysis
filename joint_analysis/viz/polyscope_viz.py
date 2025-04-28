"""
Polyscope visualization for joint analysis.
"""

import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from typing import Dict, List, Tuple, Optional, Callable, Any, Union


class PolyscopeVisualizer:
    """
    Class for visualizing joint data with Polyscope.
    """

    def __init__(self):
        """Initialize the visualizer."""
        # Initialize Polyscope if not already initialized
        if not ps.is_initialized():
            ps.init()

        # Disable the ground plane
        ps.set_ground_plane_mode("none")

        # Store registered point clouds
        self.point_clouds = {}

        # Store curve networks for joint visualization
        self.joint_curves = {}

        # Joint visualization reference objects
        self.planar_normal_reference = None
        self.planar_axis1_reference = None
        self.planar_axis2_reference = None
        self.screw_axis_reference = None
        self.prismatic_axis_reference = None
        self.revolute_axis_reference = None

        # Current state
        self.current_mode = None
        self.current_frame_index = 0
        self.point_cloud_history = {}
        self.current_best_joint = "Unknown"
        self.current_joint_params = None
        self.running = False

    def register_point_cloud(self, name: str, points: np.ndarray, enabled: bool = True) -> None:
        """
        Register a point cloud with Polyscope.

        Args:
            name (str): Name of the point cloud
            points (ndarray): Points of shape (N, 3)
            enabled (bool): Whether the point cloud is initially enabled
        """
        if name in self.point_clouds:
            # Update existing point cloud
            self.point_clouds[name].update_point_positions(points)
        else:
            # Register new point cloud
            self.point_clouds[name] = ps.register_point_cloud(name, points, enabled=enabled)

    def update_point_cloud(self, name: str, points: np.ndarray) -> None:
        """
        Update an existing point cloud.

        Args:
            name (str): Name of the point cloud
            points (ndarray): New points of shape (N, 3)
        """
        if name in self.point_clouds:
            self.point_clouds[name].update_point_positions(points)
        else:
            self.register_point_cloud(name, points)

    def highlight_point_differences(self, name: str, current_points: np.ndarray,
                                    previous_points: Optional[np.ndarray] = None) -> None:
        """
        Highlight differences between consecutive frames by coloring points.

        Args:
            name (str): Name of the point cloud
            current_points (ndarray): Current points of shape (N, 3)
            previous_points (ndarray, optional): Previous points of shape (N, 3)
        """
        if name not in self.point_clouds:
            return

        if previous_points is None or current_points.shape != previous_points.shape:
            # Initialize with uniform color if no previous points
            P = current_points.shape[0]
            colors = np.full((P, 3), 0.7, dtype=np.float32)
            self.point_clouds[name].add_color_quantity("Deviation Highlight", colors, enabled=True)
            return

        # Compute distances between current and previous points
        dists = np.linalg.norm(current_points - previous_points, axis=1)
        max_dist = np.max(dists)

        if max_dist < 1e-3:
            # No significant movement, use uniform color
            P = current_points.shape[0]
            colors = np.full((P, 3), 0.7, dtype=np.float32)
        else:
            # Color based on movement magnitude
            ratio = dists / max_dist
            r = ratio
            g = 0.5 * (1.0 - ratio)
            b = 1.0 - ratio

            colors = np.stack([r, g, b], axis=-1)

        self.point_clouds[name].add_color_quantity("Deviation Highlight", colors, enabled=True)

    def remove_joint_visualization(self) -> None:
        """Remove all joint visualization elements."""
        # Remove curve networks
        for name in [
            "Planar Normal", "Ball Center", "Screw Axis", "Screw Axis Pitch",
            "Prismatic Axis", "Revolute Axis", "Revolute Origin", "Planar Axes",
            "GT Planar Normal", "GT Ball Center", "GT Screw Axis", "GT Screw Axis Pitch",
            "GT Prismatic Axis", "GT Revolute Axis", "GT Revolute Origin", "GT Planar Axes"
        ]:
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)

        # Remove point clouds used for visualization
        for name in ["BallCenterPC", "GT BallCenterPC"]:
            if ps.has_point_cloud(name):
                ps.remove_point_cloud(name)

        # Reset joint visualization references
        self.joint_curves = {}

    def show_joint_visualization(self, joint_type: str, joint_params: Dict[str, Any],
                                 is_ground_truth: bool = False) -> None:
        """
        Show visualization for a specific joint type.

        Args:
            joint_type (str): Type of joint ('planar', 'ball', 'screw', 'prismatic', 'revolute')
            joint_params (dict): Dictionary of joint parameters
            is_ground_truth (bool): Whether this visualization is for ground truth
        """
        prefix = "GT " if is_ground_truth else ""

        if joint_type == "planar":
            # Extract parameters
            n_np = joint_params.get("normal", np.array([0., 0., 1.]))

            # Visualize normal
            seg_nodes = np.array([[0, 0, 0], n_np])
            seg_edges = np.array([[0, 1]])
            name = f"{prefix}Planar Normal"
            planarnet = ps.register_curve_network(name, seg_nodes, seg_edges)
            planarnet.set_color((1.0, 0.0, 0.0))
            planarnet.set_radius(0.02)
            self.joint_curves[name] = planarnet

            # Compute orthogonal axes
            x_axis = np.array([1., 0., 0.])
            y_axis = np.array([0., 1., 0.])

            # Choose a reference vector that's not parallel to the normal
            ref = y_axis if np.abs(np.dot(n_np, x_axis)) < np.abs(np.dot(n_np, y_axis)) else x_axis

            # Compute orthogonal vectors
            axis1 = np.cross(n_np, ref)
            axis1 = axis1 / (np.linalg.norm(axis1) + 1e-6)
            axis2 = np.cross(n_np, axis1)
            axis2 = axis2 / (np.linalg.norm(axis2) + 1e-6)

            # Visualize orthogonal axes
            seg_nodes2 = np.array([[0, 0, 0], axis1, [0, 0, 0], axis2], dtype=np.float32)
            seg_edges2 = np.array([[0, 1], [2, 3]])
            name = f"{prefix}Planar Axes"
            planarex = ps.register_curve_network(name, seg_nodes2, seg_edges2)
            planarex.set_color((0., 1., 0.))
            planarex.set_radius(0.02)
            self.joint_curves[name] = planarex

        elif joint_type == "ball":
            # Extract parameters
            center_np = joint_params.get("center", np.array([0., 0., 0.]))

            # Visualize center
            name = f"{prefix}BallCenterPC"
            c_pc = ps.register_point_cloud(name, center_np.reshape(1, 3))
            c_pc.set_radius(0.05)
            c_pc.set_enabled(True)

            # Visualize coordinate axes at the center
            x_ = np.array([1., 0., 0.])
            y_ = np.array([0., 1., 0.])
            z_ = np.array([0., 0., 1.])

            seg_nodes = np.array([
                center_np, center_np + x_,
                center_np, center_np + y_,
                center_np, center_np + z_
            ])
            seg_edges = np.array([[0, 1], [2, 3], [4, 5]])

            name = f"{prefix}Ball Center"
            axisviz = ps.register_curve_network(name, seg_nodes, seg_edges)
            axisviz.set_radius(0.02)
            axisviz.set_color((1., 0., 1.))
            self.joint_curves[name] = axisviz

        elif joint_type == "screw":
            # Extract parameters
            axis_np = joint_params.get("axis", np.array([0., 1., 0.]))
            origin_np = joint_params.get("origin", np.array([0., 0., 0.]))
            pitch_ = joint_params.get("pitch", 0.0)

            # Normalize axis if needed
            axis_norm = np.linalg.norm(axis_np)
            if axis_norm > 1e-6:
                axis_np = axis_np / axis_norm

            # Visualize axis
            seg_nodes = np.array([origin_np - axis_np * 0.5, origin_np + axis_np * 0.5])
            seg_edges = np.array([[0, 1]])

            name = f"{prefix}Screw Axis"
            scv = ps.register_curve_network(name, seg_nodes, seg_edges)
            scv.set_radius(0.02)
            scv.set_color((0., 0., 1.0))
            self.joint_curves[name] = scv

            # Visualize pitch
            # Create an arrow in a direction perpendicular to the axis
            pitch_arrow_start = origin_np + axis_np * 0.6

            # Find a perpendicular vector
            perp_vec = np.array([1, 0, 0])
            if np.abs(np.dot(axis_np, perp_vec)) > 0.9:
                perp_vec = np.array([0, 1, 0])

            perp_vec = perp_vec - np.dot(perp_vec, axis_np) * axis_np
            perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-6)

            pitch_arrow_end = pitch_arrow_start + 0.2 * pitch_ * perp_vec

            seg_nodes2 = np.array([pitch_arrow_start, pitch_arrow_end])
            seg_edges2 = np.array([[0, 1]])

            name = f"{prefix}Screw Axis Pitch"
            pitch_net = ps.register_curve_network(name, seg_nodes2, seg_edges2)
            pitch_net.set_color((1., 0., 0.))
            pitch_net.set_radius(0.02)
            self.joint_curves[name] = pitch_net

        elif joint_type == "prismatic":
            # Extract parameters
            axis_np = joint_params.get("axis", np.array([1., 0., 0.]))

            # Normalize axis if needed
            axis_norm = np.linalg.norm(axis_np)
            if axis_norm > 1e-6:
                axis_np = axis_np / axis_norm

            # Visualize axis
            seg_nodes = np.array([[0., 0., 0.], axis_np])
            seg_edges = np.array([[0, 1]])

            name = f"{prefix}Prismatic Axis"
            pcv = ps.register_curve_network(name, seg_nodes, seg_edges)
            pcv.set_radius(0.01)
            pcv.set_color((0., 1., 1.))
            self.joint_curves[name] = pcv

        elif joint_type == "revolute":
            # Extract parameters
            axis_np = joint_params.get("axis", np.array([0., 1., 0.]))
            origin_np = joint_params.get("origin", np.array([0., 0., 0.]))

            # Normalize axis if needed
            axis_norm = np.linalg.norm(axis_np)
            if axis_norm > 1e-6:
                axis_np = axis_np / axis_norm

            # Visualize axis
            seg_nodes = np.array([origin_np - axis_np * 0.5, origin_np + axis_np * 0.5])
            seg_edges = np.array([[0, 1]])

            name = f"{prefix}Revolute Axis"
            rvnet = ps.register_curve_network(name, seg_nodes, seg_edges)
            rvnet.set_radius(0.01)
            rvnet.set_color((1., 1., 0.))
            self.joint_curves[name] = rvnet

            # Visualize origin
            seg_nodes2 = np.array([origin_np, origin_np + 1e-5 * axis_np])
            seg_edges2 = np.array([[0, 1]])

            name = f"{prefix}Revolute Origin"
            origin_net = ps.register_curve_network(name, seg_nodes2, seg_edges2)
            origin_net.set_radius(0.015)
            origin_net.set_color((1., 0., 0.))
            self.joint_curves[name] = origin_net

    def set_current_mode(self, mode: str) -> None:
        """
        Set the current visualization mode.

        Args:
            mode (str): Name of the mode
        """
        if self.current_mode == mode:
            return

        # Update current mode
        self.current_mode = mode

        # Reset frame index
        self.current_frame_index = 0

        # Clear point cloud history
        self.point_cloud_history = {}

        # Disable all point clouds
        for name, pc in self.point_clouds.items():
            pc.set_enabled(name == mode)

    def store_frame(self, points: np.ndarray) -> None:
        """
        Store the current frame's points.

        Args:
            points (ndarray): Points of shape (N, 3)
        """
        if self.current_mode is None:
            return

        if self.current_mode not in self.point_cloud_history:
            self.point_cloud_history[self.current_mode] = []

        self.point_cloud_history[self.current_mode].append(points.copy())

    def save_sequence(self, output_dir: str = "exported_pointclouds") -> str:
        """
        Save the current point cloud sequence to a numpy file.

        Args:
            output_dir (str): Directory to save the file

        Returns:
            str: Path to the saved file
        """
        if self.current_mode is None or self.current_mode not in self.point_cloud_history:
            return ""

        # Create directory if it doesn't exist
        mode_dir = os.path.join(output_dir, self.current_mode.replace(" ", "_"))
        os.makedirs(mode_dir, exist_ok=True)

        # Create filename
        counter = 0
        while True:
            filename = f"{self.current_mode.replace(' ', '_')}_{counter}.npy"
            filepath = os.path.join(mode_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1

        # Stack and save
        all_points = np.stack(self.point_cloud_history[self.current_mode], axis=0)
        np.save(filepath, all_points)

        return filepath

    def get_point_cloud_sequence(self) -> Optional[np.ndarray]:
        """
        Get the current point cloud sequence.

        Returns:
            ndarray: Point cloud sequence of shape (T, N, 3) or None
        """
        if self.current_mode is None or self.current_mode not in self.point_cloud_history:
            return None

        if len(self.point_cloud_history[self.current_mode]) == 0:
            return None

        return np.stack(self.point_cloud_history[self.current_mode], axis=0)

    def set_joint_estimation_result(self, joint_type: str, joint_params: Dict[str, Any]) -> None:
        """
        Set the current joint estimation result.

        Args:
            joint_type (str): Type of joint ('planar', 'ball', 'screw', 'prismatic', 'revolute')
            joint_params (dict): Dictionary of joint parameters
        """
        self.current_best_joint = joint_type
        self.current_joint_params = joint_params

        # Remove existing joint visualization
        self.remove_joint_visualization()

        # Show new joint visualization
        if joint_type in ["planar", "ball", "screw", "prismatic", "revolute"] and joint_params is not None:
            self.show_joint_visualization(joint_type, joint_params)

    def show_ground_truth(self, joint_type: str, joint_params: Dict[str, Any], enable: bool = True) -> None:
        """
        Show ground truth visualization.

        Args:
            joint_type (str): Type of joint ('planar', 'ball', 'screw', 'prismatic', 'revolute')
            joint_params (dict): Dictionary of joint parameters
            enable (bool): Whether to enable the visualization
        """
        if not enable:
            # Remove ground truth visualization
            for name in self.joint_curves:
                if name.startswith("GT "):
                    ps.remove_curve_network(name)
            if ps.has_point_cloud("GT BallCenterPC"):
                ps.remove_point_cloud("GT BallCenterPC")
            return

        # Show ground truth visualization
        if joint_type in ["planar", "ball", "screw", "prismatic", "revolute"] and joint_params is not None:
            self.show_joint_visualization(joint_type, joint_params, is_ground_truth=True)

    def setup_camera(self, center: Optional[np.ndarray] = None):
        """Set up the camera view."""
        # 不进行任何设置，让Polyscope自动处理视角
        # 或者可以尝试最基本的函数（如果有的话）
        try:
            ps.set_view_angle(45.0)
        except:
            pass

        # 注释掉所有可能不兼容的函数调用
        # ps.look_at(center)
        # ps.reset_camera()

    def show(self, callback: Optional[Callable] = None) -> None:
        """
        Show the visualization window.

        Args:
            callback (Callable, optional): Callback function to run in the UI loop
        """
        if callback is not None:
            ps.set_user_callback(callback)

        ps.show()