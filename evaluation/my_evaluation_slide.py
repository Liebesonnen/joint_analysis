import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from io import BytesIO
import os
import json
import re
from datetime import datetime

from robot_utils import console
from robot_utils.viz.polyscope import PolyscopeUtils, ps, psim, register_point_cloud, draw_frame_3d
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from robot_utils.path import add_deps_path
import sys


def quick_setup_joint_analysis() -> bool:
    """Quick setup function to import joint analysis dependencies.

    Attempts to add the joint-analysis package to the path and import it.
    Falls back to a hardcoded path if the package method fails.

    Returns:
        bool: True if successful, raises exception if both methods fail.
    """
    try:
        add_deps_path(pkg_name="joint-analysis")
        import joint_analysis
        return True
    except:
        # Fallback to hardcoded path method
        sys.path.insert(0, '/common/homes/all/uksqc_chen/projects/control')
        import joint_analysis
        return True


quick_setup_joint_analysis()
# Import joint analysis project modules
from joint_analysis.core.joint_estimation import compute_joint_info_all_types


class EnhancedViz:
    """Enhanced visualization class for joint analysis of point cloud data.

    This class provides comprehensive visualization and analysis capabilities for
    understanding joint motion in point cloud sequences. It supports multiple
    joint types (revolute, prismatic, ball, screw, planar) and includes ground
    truth comparison functionality.

    Attributes:
        file_paths (List[str]): List of file paths to point cloud data files.
        pu (PolyscopeUtils): Polyscope utilities instance for 3D visualization.
        t (int): Current time frame index.
        t_changed (bool): Flag indicating if time frame has changed.
        idx_point (int): Currently selected point index.
        idx_point_changed (bool): Flag indicating if point selection has changed.
        current_dataset (int): Index of currently active dataset.
        noise_sigma (float): Standard deviation for optional noise addition.
        ground_truth_json (str): Path to ground truth joint data JSON file.
        ground_truth_data (Dict): Loaded ground truth data structure.
        show_ground_truth (bool): Flag to show/hide ground truth visualization.
        ground_truth_scale (float): Scale factor for ground truth visualization.
        datasets (Dict[str, Dict]): Dictionary containing all loaded datasets.
        T (int): Number of time frames in current dataset.
        N (int): Number of points per frame in current dataset.
        dt_mean (float): Average time step between frames.
        num_neighbors (int): Number of neighbors for SVD computation.
        output_dir (str): Directory for saving visualization outputs.
    """

    def __init__(self, file_paths: Optional[List[str]] = None) -> None:
        """Initialize the EnhancedViz visualization system.

        Args:
            file_paths: List of paths to .npy files containing point cloud sequences.
                      If None, uses default test file.
        """
        # Initialize default file paths list if none provided
        if file_paths is None:
            self.file_paths: List[str] = [".parahome_data-slides/s1_gasstove_part2_1110_1170.npy"]
        else:
            self.file_paths: List[str] = file_paths

        # Create Polyscope utils instance for 3D visualization
        self.pu: PolyscopeUtils = PolyscopeUtils()

        # Initialize current time frame index and change flag
        self.t: int = 0
        self.t_changed: bool = False

        # Initialize current point index and change flag
        self.idx_point: int = 0
        self.idx_point_changed: bool = False

        # Current selected dataset index
        self.current_dataset: int = 0
        self.noise_sigma: float = 0.00

        # Load ground truth joint data from specified location
        self.ground_truth_json: str = "./parahome_data_slide/all_scenes_transformed_axis_pivot_data.json"
        self.ground_truth_data: Dict = self.load_ground_truth()
        self.show_ground_truth: bool = False
        self.ground_truth_scale: float = 0.5

        # Track ground truth visualization elements for cleanup
        self.gt_curve_networks: List[str] = []
        self.gt_point_clouds: List[str] = []

        # Coordinate system display settings
        self.coord_scale: float = 0.1

        # Store multiple datasets dictionary
        self.datasets: Dict[str, Dict] = {}

        # Load all specified files and process them
        for i, file_path in enumerate(self.file_paths):
            # Give each dataset a unique identifier
            dataset_name: str = f"dataset_{i}"

            # Load numpy array data
            data: np.ndarray = np.load(file_path)

            # Create display name from filename for UI
            display_name: str = os.path.basename(file_path).split('.')[0]

            # Add optional noise to data if specified
            if self.noise_sigma > 0:
                data += np.random.randn(*data.shape) * self.noise_sigma

            # Store dataset information in structured format
            self.datasets[dataset_name] = {
                "path": file_path,
                "display_name": display_name,
                "data": data,
                "T": data.shape[0],  # Number of time frames
                "N": data.shape[1],  # Number of points per frame
                "visible": True  # Whether to display in visualization
            }

            # Apply Savitzky-Golay filter to smooth trajectory data
            # This reduces noise while preserving important motion characteristics
            self.datasets[dataset_name]["data_filter"] = savgol_filter(
                x=data, window_length=21, polyorder=2, deriv=0, axis=0, delta=0.1
            )

            # Use Savitzky-Golay filter to calculate velocity (first derivative)
            # This provides smooth velocity estimates from position data
            self.datasets[dataset_name]["dv_filter"] = savgol_filter(
                x=data, window_length=21, polyorder=2, deriv=1, axis=0, delta=0.1
            )

        # Set first dataset as current active dataset
        dataset_keys: List[str] = list(self.datasets.keys())
        if dataset_keys:
            self.current_dataset_key: str = dataset_keys[0]
            current_data: Dict = self.datasets[self.current_dataset_key]

            # Get dimensions of current dataset
            self.T: int = current_data["T"]
            self.N: int = current_data["N"]

            # Set current active data references
            self.d: np.ndarray = current_data["data"]
            self.d_filter: np.ndarray = current_data["data_filter"]
            self.dv_filter: np.ndarray = current_data["dv_filter"]
        else:
            ic("No datasets loaded.")
            return

        # Average time step between frames (currently unused but available for timing)
        self.dt_mean: float = 0

        # Number of neighbors for SVD-based rotation computation
        self.num_neighbors: int = 50

        # Joint analysis parameter configuration
        # These parameters control the sensitivity and behavior of joint detection algorithms
        self.col_sigma: float = 0.2  # Collinearity analysis standard deviation
        self.col_order: float = 4.0  # Collinearity analysis order parameter
        self.cop_sigma: float = 0.2  # Coplanarity analysis standard deviation
        self.cop_order: float = 4.0  # Coplanarity analysis order parameter
        self.rad_sigma: float = 0.2  # Radius consistency standard deviation
        self.rad_order: float = 4.0  # Radius consistency order parameter
        self.zp_sigma: float = 0.2  # Zero pitch analysis standard deviation
        self.zp_order: float = 4.0  # Zero pitch analysis order parameter
        self.use_savgol: bool = True  # Whether to use Savitzky-Golay filtering
        self.savgol_window: int = 21  # Savitzky-Golay filter window length
        self.savgol_poly: int = 2  # Savitzky-Golay polynomial order

        # Create output directory for saving visualization images
        self.output_dir: str = "visualization_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Calculate angular velocity for all datasets
        for dataset_key in self.datasets:
            dataset: Dict = self.datasets[dataset_key]
            angular_velocity_raw: np.ndarray
            angular_velocity_filtered: np.ndarray
            angular_velocity_raw, angular_velocity_filtered = self.calculate_angular_velocity(
                dataset["data_filter"], dataset["N"]
            )
            dataset["angular_velocity_raw"] = angular_velocity_raw
            dataset["angular_velocity_filtered"] = angular_velocity_filtered

            # Perform comprehensive joint analysis on filtered data
            joint_params: Dict
            best_joint: str
            info_dict: Dict
            joint_params, best_joint, info_dict = self.perform_joint_analysis(dataset["data_filter"])
            dataset["joint_params"] = joint_params
            dataset["best_joint"] = best_joint
            dataset["joint_info"] = info_dict

            # Attempt to map dataset to corresponding ground truth entry
            dataset["ground_truth_key"] = self.map_dataset_to_ground_truth(dataset["display_name"])

        # Set current dataset's angular velocity references
        self.angular_velocity_raw: np.ndarray = self.datasets[self.current_dataset_key]["angular_velocity_raw"]
        self.angular_velocity_filtered: np.ndarray = self.datasets[self.current_dataset_key][
            "angular_velocity_filtered"]

        # Set current dataset's joint analysis results
        self.current_joint_params: Dict = self.datasets[self.current_dataset_key]["joint_params"]
        self.current_best_joint: str = self.datasets[self.current_dataset_key]["best_joint"]
        self.current_joint_info: Dict = self.datasets[self.current_dataset_key]["joint_info"]

        # Draw coordinate frame at origin for reference
        draw_frame_3d(np.zeros(6), label="origin", scale=0.1)

        # Register initial point clouds for all visible datasets
        for dataset_key, dataset in self.datasets.items():
            if dataset["visible"]:
                register_point_cloud(
                    dataset["display_name"],
                    dataset["data_filter"][self.t],
                    radius=0.01,
                    enabled=True
                )

        # Reset viewport to encompass all visible point clouds
        visible_point_clouds: List[np.ndarray] = [
            dataset["data_filter"][self.t] for dataset in self.datasets.values()
            if dataset["visible"]
        ]
        if visible_point_clouds:
            self.pu.reset_bbox_from_pcl_list(visible_point_clouds)

        # Visualize estimated joint parameters for current dataset
        self.visualize_joint_parameters()

        # Set user interaction callback function for UI
        ps.set_user_callback(self.callback)

        # Launch Polyscope interactive interface
        ps.show()

    def load_ground_truth(self) -> Dict:
        """Load ground truth joint data from JSON configuration file.

        Attempts to load and parse the ground truth data file containing
        known joint parameters for comparison with estimated results.

        Returns:
            Dict: Parsed ground truth data structure, empty dict if loading fails.
        """
        try:
            with open(self.ground_truth_json, 'r') as f:
                data: Dict = json.load(f)
                print(f"Successfully loaded ground truth data from: {self.ground_truth_json}")
                ic(f"Available scenes: {list(data.keys())}")
                return data
        except FileNotFoundError:
            print(f"Warning: Ground truth file not found at {self.ground_truth_json}")
            return {}
        except Exception as e:
            print(f"Error loading ground truth data: {e}")
            return {}

    def extract_scene_info_from_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """Extract scene and object information from standardized filename format.

        Parses filenames following the pattern: "s{scene_id}_{object}_{part}_{start}_{end}"
        to extract structured information for ground truth mapping.

        Args:
            filename: Input filename to parse (e.g., "s204_drawer_part2_1320_1440").

        Returns:
            Dict containing "scene", "object", and "part" keys, or None if parsing fails.
        """
        # Primary pattern to match filenames like "s204_drawer_part2_1320_1440"
        pattern: str = r'(s\d+)_([^_]+)_(part\d+)_\d+_\d+'
        match = re.match(pattern, filename)

        if match:
            scene: str
            object_type: str
            part: str
            scene, object_type, part = match.groups()
            return {
                "scene": scene,
                "object": object_type,
                "part": part
            }

        # Alternative pattern for base parts or different naming conventions
        pattern2: str = r'(s\d+)_([^_]+)_(base)_\d+_\d+'
        match2 = re.match(pattern2, filename)

        if match2:
            scene, object_type, part = match2.groups()
            return {
                "scene": scene,
                "object": object_type,
                "part": part
            }

        print(f"Warning: Could not extract scene info from filename: {filename}")
        return None

    def map_dataset_to_ground_truth(self, display_name: str) -> Optional[Dict[str, str]]:
        """Map dataset display name to corresponding ground truth entry.

        Uses filename parsing to identify the correct ground truth data
        entry for comparison with estimated joint parameters.

        Args:
            display_name: Display name of dataset (typically filename without extension).

        Returns:
            Dict containing scene, object, and part mapping, or None if not found.
        """
        scene_info: Optional[Dict[str, str]] = self.extract_scene_info_from_filename(display_name)

        if not scene_info:
            return None

        scene: str = scene_info["scene"]
        object_type: str = scene_info["object"]
        part: str = scene_info["part"]

        # Verify scene exists in ground truth data
        if scene not in self.ground_truth_data:
            print(f"Warning: Scene {scene} not found in ground truth data")
            return None

        # Verify object exists in the scene
        if object_type not in self.ground_truth_data[scene]:
            print(f"Warning: Object {object_type} not found in scene {scene} ground truth data")
            return None

        # Verify part exists for the object
        if part not in self.ground_truth_data[scene][object_type]:
            print(f"Warning: Part {part} not found for {object_type} in scene {scene}")
            return None

        print(f"Mapped {display_name} -> Scene: {scene}, Object: {object_type}, Part: {part}")
        return {
            "scene": scene,
            "object": object_type,
            "part": part
        }

    def perform_joint_analysis(self, point_history: np.ndarray) -> Tuple[Dict, str, Dict]:
        """Perform comprehensive joint analysis on point trajectory data.

        Analyzes point motion patterns to identify the most likely joint type
        and estimate joint parameters. Supports multiple joint types including
        revolute, prismatic, ball, screw, and planar joints.

        Args:
            point_history: 3D array of shape (T, N, 3) containing point trajectories
                         over time, where T is time frames, N is points, 3 is coordinates.

        Returns:
            Tuple containing:
                - joint_params: Dictionary of estimated parameters for each joint type
                - best_joint: String identifier of most likely joint type
                - info_dict: Detailed analysis information including scores and probabilities
        """
        joint_params: Dict
        best_joint: str
        info_dict: Dict
        joint_params, best_joint, info_dict = compute_joint_info_all_types(
            point_history,
            neighbor_k=self.num_neighbors,
            col_sigma=self.col_sigma,
            col_order=self.col_order,
            cop_sigma=self.cop_sigma,
            cop_order=self.cop_order,
            rad_sigma=self.rad_sigma,
            rad_order=self.rad_order,
            zp_sigma=self.zp_sigma,
            zp_order=self.zp_order,
            use_savgol=self.use_savgol,
            savgol_window=self.savgol_window,
            savgol_poly=self.savgol_poly
        )

        # Print comprehensive joint analysis results to console
        console.rule()
        print(f"Joint Type: {best_joint}")
        console.rule()

        # Display basic scoring metrics for joint classification
        if info_dict and "basic_score_avg" in info_dict:
            basic_scores: Dict = info_dict["basic_score_avg"]
            print(f"Basic Scores:\n"
                  f"Collinearity Score: {basic_scores.get('col_mean', 0.0):.16f}\n"
                  f"Coplanarity Score: {basic_scores.get('cop_mean', 0.0):.16f}\n"
                  f"Radius Consistency Score: {basic_scores.get('rad_mean', 0.0):.16f}\n"
                  f"Zero Pitch Score: {basic_scores.get('zp_mean', 0.0):.16f}")

        # Display joint type probabilities
        if info_dict and "joint_probs" in info_dict:
            joint_probs: Dict = info_dict["joint_probs"]
            console.log("[bold green]Joint Probabilities:")
            for joint_type, prob in joint_probs.items():
                console.log(f"{joint_type.capitalize()}: {prob:.16f}")

        # Display detailed parameters for the best matching joint type
        if best_joint in joint_params:
            params: Dict = joint_params[best_joint]
            print(f"\n{best_joint.capitalize()} Joint Parameters:")

            if best_joint == "planar":
                normal: List[float] = params.get("normal", [0, 0, 0])
                motion_limit: Tuple[float, float] = params.get("motion_limit", (0, 0))
                print(f"Normal Vector: [{normal[0]:.16f}, {normal[1]:.16f}, {normal[2]:.16f}]\n"
                      f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f})"
                      )

            elif best_joint == "ball":
                center: List[float] = params.get("center", [0, 0, 0])
                radius: float = params.get("radius", 0)
                motion_limit: Tuple[float, float, float] = params.get("motion_limit", (0, 0, 0))
                print(f"Center: [{center[0]:.16f}, {center[1]:.16f}, {center[2]:.16f}]\n",
                      f"Radius: {radius:.16f}\n",
                      f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}, {motion_limit[2]:.16f}) rad"
                      )

            elif best_joint == "screw":
                axis: List[float] = params.get("axis", [0, 0, 0])
                origin: List[float] = params.get("origin", [0, 0, 0])
                pitch: float = params.get("pitch", 0)
                motion_limit: Tuple[float, float] = params.get("motion_limit", (0, 0))
                print(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]\n",
                      f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]\n",
                      f"Pitch: {pitch:.16f}\n",
                      f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) rad"
                      )

            elif best_joint == "prismatic":
                axis: List[float] = params.get("axis", [0, 0, 0])
                origin: List[float] = params.get("origin", [0, 0, 0])
                motion_limit: Tuple[float, float] = params.get("motion_limit", (0, 0))
                print(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]\n",
                      f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]\n",
                      f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) m\n"
                      )

            elif best_joint == "revolute":
                axis: List[float] = params.get("axis", [0, 0, 0])
                origin: List[float] = params.get("origin", [0, 0, 0])
                motion_limit: Tuple[float, float] = params.get("motion_limit", (0, 0))
                print(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]\n",
                      f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]\n",
                      f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) rad\n",
                      f"Motion Range: {np.degrees(motion_limit[1] - motion_limit[0]):.16f}°\n"
                      )
        console.rule()
        return joint_params, best_joint, info_dict

    def visualize_joint_parameters(self) -> None:
        """Visualize estimated joint parameters in the 3D Polyscope interface.

        Creates 3D visualizations of joint axes, origins, and other geometric
        elements based on the estimated joint type and parameters.
        """
        # Remove any existing joint visualizations to prevent overlap
        self.remove_joint_visualization()

        joint_type: str = self.current_best_joint
        joint_params: Optional[Dict] = None

        if joint_type in self.current_joint_params:
            joint_params = self.current_joint_params[joint_type]
            self.show_joint_visualization(joint_type, joint_params)

    def remove_joint_visualization(self) -> None:
        """Remove all existing joint visualization elements from the 3D scene.

        Cleans up curve networks and point clouds created for joint parameter
        visualization to prevent visual clutter when updating displays.
        """
        # Remove all possible joint visualization curve networks
        visualization_names: List[str] = [
            "Planar Normal", "Ball Center", "Screw Axis", "Screw Axis Pitch",
            "Prismatic Axis", "Revolute Axis", "Revolute Origin", "Planar Axes"
        ]
        for name in visualization_names:
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)

        # Remove point clouds used for joint visualization
        point_cloud_names: List[str] = ["BallCenterPC"]
        for name in point_cloud_names:
            if ps.has_point_cloud(name):
                ps.remove_point_cloud(name)

    def show_joint_visualization(self, joint_type: str, joint_params: Dict) -> None:
        """Display 3D visualization for a specific joint type and its parameters.

        Creates appropriate geometric representations for different joint types
        including axes, origins, centers, and directional indicators.

        Args:
            joint_type: Type of joint to visualize ("planar", "ball", "screw", etc.).
            joint_params: Dictionary containing joint-specific parameters.
        """
        if joint_type == "planar":
            # Extract normal vector for planar joint
            n_np: np.ndarray = joint_params.get("normal", np.array([0., 0., 1.]))

            # Create line segment representing the normal direction
            seg_nodes: np.ndarray = np.array([[0, 0, 0], n_np])
            seg_edges: np.ndarray = np.array([[0, 1]])
            name: str = "Planar Normal"
            planarnet = ps.register_curve_network(name, seg_nodes, seg_edges)
            planarnet.set_color((1.0, 0.0, 0.0))  # Red color for normal vector
            planarnet.set_radius(0.02)

        elif joint_type == "ball":
            # Extract center point for ball joint
            center_np: np.ndarray = joint_params.get("center", np.array([0., 0., 0.]))

            # Visualize center as a point cloud
            name = "BallCenterPC"
            c_pc = ps.register_point_cloud(name, center_np.reshape(1, 3))
            c_pc.set_radius(0.05)
            c_pc.set_enabled(True)

            # Create coordinate axes at the ball joint center
            x_axis: np.ndarray = np.array([1., 0., 0.])
            y_axis: np.ndarray = np.array([0., 1., 0.])
            z_axis: np.ndarray = np.array([0., 0., 1.])

            # Build line segments for all three coordinate axes
            seg_nodes = np.array([
                center_np, center_np + x_axis,
                center_np, center_np + y_axis,
                center_np, center_np + z_axis
            ])
            seg_edges = np.array([[0, 1], [2, 3], [4, 5]])

            name = "Ball Center"
            axisviz = ps.register_curve_network(name, seg_nodes, seg_edges)
            axisviz.set_radius(0.02)
            axisviz.set_color((1., 0., 1.))  # Magenta color for ball joint axes

        elif joint_type == "screw":
            # Extract screw joint parameters
            axis_np: np.ndarray = joint_params.get("axis", np.array([0., 1., 0.]))
            origin_np: np.ndarray = joint_params.get("origin", np.array([0., 0., 0.]))
            pitch_: float = joint_params.get("pitch", 0.0)

            # Normalize axis vector for consistent visualization
            axis_norm: float = np.linalg.norm(axis_np)
            if axis_norm > 1e-6:
                axis_np = axis_np / axis_norm

            # Visualize screw axis as a line segment
            seg_nodes = np.array([origin_np - axis_np * 0.5, origin_np + axis_np * 0.5])
            seg_edges = np.array([[0, 1]])

            name = "Screw Axis"
            scv = ps.register_curve_network(name, seg_nodes, seg_edges)
            scv.set_radius(0.02)
            scv.set_color((0., 0., 1.0))  # Blue color for screw axis

            # Visualize pitch parameter using a directional arrow
            pitch_arrow_start: np.ndarray = origin_np + axis_np * 0.6

            # Find a perpendicular vector to the axis for pitch visualization
            perp_vec: np.ndarray = np.array([1, 0, 0])
            if np.abs(np.dot(axis_np, perp_vec)) > 0.9:
                perp_vec = np.array([0, 1, 0])

            # Make perpendicular vector orthogonal to axis
            perp_vec = perp_vec - np.dot(perp_vec, axis_np) * axis_np
            perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-9)

            pitch_arrow_end: np.ndarray = pitch_arrow_start + 0.2 * pitch_ * perp_vec

            seg_nodes2 = np.array([pitch_arrow_start, pitch_arrow_end])
            seg_edges2 = np.array([[0, 1]])

            name = "Screw Axis Pitch"
            pitch_net = ps.register_curve_network(name, seg_nodes2, seg_edges2)
            pitch_net.set_color((1., 0., 0.))  # Red color for pitch indicator
            pitch_net.set_radius(0.02)

        elif joint_type == "prismatic":
            # Extract prismatic joint parameters
            axis_np: np.ndarray = joint_params.get("axis", np.array([1., 0., 0.]))
            origin_np: np.ndarray = joint_params.get("origin", np.array([0., 0., 0.]))

            # Normalize axis vector for consistent visualization
            axis_norm: float = np.linalg.norm(axis_np)
            if axis_norm > 1e-6:
                axis_np = axis_np / axis_norm

            # Visualize prismatic motion axis
            seg_nodes = np.array([origin_np, origin_np + axis_np])
            seg_edges = np.array([[0, 1]])

            name = "Prismatic Axis"
            pcv = ps.register_curve_network(name, seg_nodes, seg_edges)
            pcv.set_radius(0.01)
            pcv.set_color((0., 1., 1.))  # Cyan color for prismatic axis

        elif joint_type == "revolute":
            # Extract revolute joint parameters
            axis_np: np.ndarray = joint_params.get("axis", np.array([0., 1., 0.]))
            origin_np: np.ndarray = joint_params.get("origin", np.array([0., 0., 0.]))

            # Normalize axis vector for consistent visualization
            axis_norm: float = np.linalg.norm(axis_np)
            if axis_norm > 1e-6:
                axis_np = axis_np / axis_norm

            # Visualize revolute rotation axis
            seg_nodes = np.array([origin_np - axis_np * 0.5, origin_np + axis_np * 0.5])
            seg_edges = np.array([[0, 1]])

            name = "Revolute Axis"
            rvnet = ps.register_curve_network(name, seg_nodes, seg_edges)
            rvnet.set_radius(0.01)
            rvnet.set_color((1., 1., 0.))  # Yellow color for revolute axis

            # Visualize revolute joint origin point
            seg_nodes2 = np.array([origin_np, origin_np + 1e-5 * axis_np])
            seg_edges2 = np.array([[0, 1]])

            name = "Revolute Origin"
            origin_net = ps.register_curve_network(name, seg_nodes2, seg_edges2)
            origin_net.set_radius(0.015)
            origin_net.set_color((1., 0., 0.))  # Red color for origin point

    def visualize_ground_truth(self) -> None:
        """Visualize ground truth joint information for comparison with estimates.

        Displays known joint parameters from ground truth data alongside
        estimated parameters for visual comparison and validation.
        """
        # Remove existing ground truth visualization elements
        self.remove_ground_truth_visualization()

        # Process all visible datasets that have ground truth mappings
        for dataset_key, dataset in self.datasets.items():
            if not dataset["visible"] or dataset["ground_truth_key"] is None:
                continue

            gt_key: Dict[str, str] = dataset["ground_truth_key"]
            scene: str = gt_key["scene"]
            object_type: str = gt_key["object"]
            part_info: str = gt_key["part"]

            # Verify ground truth data availability
            if scene not in self.ground_truth_data:
                continue

            if object_type not in self.ground_truth_data[scene]:
                continue

            if part_info not in self.ground_truth_data[scene][object_type]:
                continue

            self.visualize_ground_truth_joint(scene, object_type, part_info, dataset["display_name"])

    def visualize_ground_truth_joint(self, scene: str, object_type: str, part_name: str, display_name: str) -> None:
        """Visualize a specific ground truth joint with axis and pivot information.

        Creates 3D visualization elements for known joint parameters from
        ground truth data, using distinct colors to differentiate from estimates.

        Args:
            scene: Scene identifier string.
            object_type: Type of object (e.g., "drawer", "door").
            part_name: Specific part identifier (e.g., "part1", "base").
            display_name: Display name for labeling visualization elements.
        """
        # Verify data hierarchy exists
        if scene not in self.ground_truth_data:
            return

        if object_type not in self.ground_truth_data[scene]:
            return

        if part_name not in self.ground_truth_data[scene][object_type]:
            return

        joint_info: Dict = self.ground_truth_data[scene][object_type][part_name]

        # Extract axis and pivot information from ground truth
        if "axis" not in joint_info or "pivot" not in joint_info:
            return

        axis: np.ndarray = np.array(joint_info["axis"])

        # Handle different pivot data formats (scalar or vector)
        pivot = joint_info["pivot"]
        if isinstance(pivot, list):
            if len(pivot) == 1:
                # Single parameter pivot - position along axis from origin
                pivot_point: np.ndarray = np.array([0., 0., 0.]) + float(pivot[0]) * axis
            elif len(pivot) == 3:
                # Full 3D pivot point coordinates
                pivot_point = np.array(pivot)
            else:
                return
        else:
            # Numeric pivot value - position along axis
            pivot_point = np.array([0., 0., 0.]) + float(pivot) * axis

        # Normalize axis vector for consistent visualization
        axis_norm: float = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis = axis / axis_norm

        # Apply visualization scale factor
        axis = axis * self.ground_truth_scale

        # Create axis visualization with green color for ground truth
        seg_nodes: np.ndarray = np.array([pivot_point - axis, pivot_point + axis])
        seg_edges: np.ndarray = np.array([[0, 1]])

        name: str = f"GT_{display_name}_Axis"
        axis_viz = ps.register_curve_network(name, seg_nodes, seg_edges)
        axis_viz.set_radius(0.015)
        axis_viz.set_color((0.0, 0.8, 0.2))  # Green color for ground truth
        self.gt_curve_networks.append(name)

        # Create pivot point visualization
        name = f"GT_{display_name}_Pivot"
        pivot_viz = ps.register_curve_network(
            name,
            np.array([pivot_point, pivot_point + 0.01 * axis]),
            np.array([[0, 1]])
        )
        pivot_viz.set_radius(0.02)
        pivot_viz.set_color((0.2, 0.8, 0.2))  # Green color for ground truth
        self.gt_curve_networks.append(name)

    def remove_ground_truth_visualization(self) -> None:
        """Remove all ground truth visualization elements from the 3D scene.

        Cleans up ground truth visualization elements to prevent clutter
        when updating or toggling ground truth display.
        """
        # Remove tracked curve networks
        for name in self.gt_curve_networks:
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)

        # Remove tracked point clouds
        for name in self.gt_point_clouds:
            if ps.has_point_cloud(name):
                ps.remove_point_cloud(name)

        # Clear tracking lists for future use
        self.gt_curve_networks = []
        self.gt_point_clouds = []

    def find_neighbors(self, points: np.ndarray, num_neighbors: int) -> np.ndarray:
        """Find the closest neighboring points for each point in the dataset.

        Computes distance-based neighborhoods for SVD analysis of local
        point configurations and motion patterns.

        Args:
            points: Array of 3D points with shape (N, 3).
            num_neighbors: Number of closest neighbors to find for each point.

        Returns:
            Array of neighbor indices with shape (N, num_neighbors).
        """
        # Calculate pairwise distance matrix for all points
        N: int = points.shape[0]
        dist_matrix: np.ndarray = np.zeros((N, N))
        for i in range(N):
            dist_matrix[i] = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))

        # Find closest neighbors (excluding self) for each point
        neighbors: np.ndarray = np.zeros((N, num_neighbors), dtype=int)
        for i in range(N):
            # Sort by distance and take closest neighbors (skip index 0 which is self)
            indices: np.ndarray = np.argsort(dist_matrix[i])[1:num_neighbors + 1]
            neighbors[i] = indices

        return neighbors

    def compute_rotation_matrix(self, src_points: np.ndarray, dst_points: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Compute optimal rotation matrix between two corresponding point sets using SVD.

        Uses Singular Value Decomposition to find the best rotation matrix
        that aligns source points with destination points, accounting for
        translation and potential reflection.

        Args:
            src_points: Source point set with shape (N, 3).
            dst_points: Destination point set with shape (N, 3).

        Returns:
            Tuple containing:
                - R: 3x3 rotation matrix
                - src_center: Centroid of source points
                - dst_center: Centroid of destination points
        """
        # Center both point sets at their centroids
        src_center: np.ndarray = np.mean(src_points, axis=0)
        dst_center: np.ndarray = np.mean(dst_points, axis=0)

        src_centered: np.ndarray = src_points - src_center
        dst_centered: np.ndarray = dst_points - dst_center

        # Compute cross-covariance matrix
        H: np.ndarray = np.dot(src_centered.T, dst_centered)

        # Perform SVD decomposition
        U: np.ndarray
        Vt: np.ndarray
        U, _, Vt = np.linalg.svd(H)

        # Construct rotation matrix from SVD components
        R: np.ndarray = np.dot(Vt.T, U.T)

        # Handle reflection case to ensure proper rotation
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        return R, src_center, dst_center

    def rotation_matrix_to_angular_velocity(self, R: np.ndarray, dt: float) -> np.ndarray:
        """Extract angular velocity vector from rotation matrix.

        Converts a rotation matrix to the corresponding angular velocity
        vector using axis-angle representation and time step information.

        Args:
            R: 3x3 rotation matrix representing frame-to-frame rotation.
            dt: Time step between frames in seconds.

        Returns:
            3D angular velocity vector in rad/s.
        """
        # Ensure R is a valid rotation matrix through SVD cleanup
        U: np.ndarray
        Vt: np.ndarray
        U, _, Vt = np.linalg.svd(R)
        R = np.dot(U, Vt)

        # Compute rotation angle using trace formula
        cos_theta: float = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta: float = np.arccos(cos_theta)

        # Handle small angle case to avoid numerical issues
        if abs(theta) < 1e-6:
            return np.zeros(3)

        # Calculate sine of rotation angle
        sin_theta: float = np.sin(theta)
        if abs(sin_theta) < 1e-6:
            return np.zeros(3)

        # Extract angular velocity from skew-symmetric component of rotation matrix
        W: np.ndarray = (R - R.T) / (2 * sin_theta)
        omega: np.ndarray = np.zeros(3)
        omega[0] = W[2, 1]  # Extract x-component from skew-symmetric matrix
        omega[1] = W[0, 2]  # Extract y-component from skew-symmetric matrix
        omega[2] = W[1, 0]  # Extract z-component from skew-symmetric matrix

        # Convert to angular velocity: axis * angle / time
        omega = omega * theta / (dt + 1e-15)  # Add small epsilon to prevent division by zero

        return omega

    def calculate_angular_velocity(self, d_filter: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute angular velocity for each point using SVD on local neighborhoods.

        Analyzes local point motion patterns to estimate angular velocities
        by fitting rotation matrices to neighborhood deformation between frames.

        Args:
            d_filter: Filtered point trajectory data with shape (T, N, 3).
            N: Number of points per frame.

        Returns:
            Tuple containing:
                - angular_velocity: Raw angular velocity estimates
                - angular_velocity_filtered: Smoothed angular velocity data
        """
        T: int = d_filter.shape[0]

        # Initialize angular velocity storage array
        angular_velocity: np.ndarray = np.zeros((T - 1, N, 3))

        # Process each consecutive frame pair
        for t in range(T - 1):
            # Get point positions for current and next frames
            current_points: np.ndarray = d_filter[t]
            next_points: np.ndarray = d_filter[t + 1]

            # Find neighborhood structure for all points
            neighbors: np.ndarray = self.find_neighbors(current_points, self.num_neighbors)

            # Compute angular velocity for each individual point
            for i in range(N):
                # Extract local neighborhood around current point
                src_neighborhood: np.ndarray = current_points[neighbors[i]]
                dst_neighborhood: np.ndarray = next_points[neighbors[i]]

                # Compute optimal rotation between neighborhood configurations
                R: np.ndarray
                R, _, _ = self.compute_rotation_matrix(src_neighborhood, dst_neighborhood)

                # Extract angular velocity from rotation matrix
                omega: np.ndarray = self.rotation_matrix_to_angular_velocity(R, self.dt_mean)

                # Store computed angular velocity
                angular_velocity[t, i] = omega

        # Apply smoothing filter to angular velocity data
        angular_velocity_filtered: np.ndarray = np.zeros_like(angular_velocity)

        if T - 1 >= 5:  # Ensure sufficient data for filtering
            for i in range(N):
                for dim in range(3):
                    # Apply Savitzky-Golay filter to each component
                    angular_velocity_filtered[:, i, dim] = savgol_filter(
                        angular_velocity[:, i, dim],
                        window_length=11,  # Window size for smoothing
                        polyorder=2  # Polynomial order for fitting
                    )
        else:
            # Use raw data if insufficient frames for filtering
            angular_velocity_filtered = angular_velocity.copy()

        return angular_velocity, angular_velocity_filtered

    def switch_dataset(self, new_dataset_key: str) -> bool:
        """Switch the currently active dataset for analysis and visualization.

        Updates all current data references and refreshes visualizations
        to reflect the newly selected dataset.

        Args:
            new_dataset_key: Identifier key for the dataset to switch to.

        Returns:
            True if switch was successful, False if dataset key not found.
        """
        if new_dataset_key in self.datasets:
            self.current_dataset_key = new_dataset_key
            dataset: Dict = self.datasets[new_dataset_key]

            # Update current data references for new dataset
            self.d = dataset["data"]
            self.d_filter = dataset["data_filter"]
            self.dv_filter = dataset["dv_filter"]
            self.angular_velocity_raw = dataset["angular_velocity_raw"]
            self.angular_velocity_filtered = dataset["angular_velocity_filtered"]

            # Update joint analysis results for new dataset
            self.current_joint_params = dataset["joint_params"]
            self.current_best_joint = dataset["best_joint"]
            self.current_joint_info = dataset["joint_info"]

            # Update temporal and spatial dimensions
            self.T = dataset["T"]
            self.N = dataset["N"]

            # Ensure current indices are within valid ranges for new dataset
            self.t = min(self.t, self.T - 1)
            self.idx_point = min(self.idx_point, self.N - 1)

            # Refresh visualization components
            self.plot_image()
            self.visualize_joint_parameters()

            # Update ground truth visualization if currently displayed
            if self.show_ground_truth:
                self.visualize_ground_truth()

            return True
        return False

    def toggle_visibility(self, dataset_key: str) -> bool:
        """Toggle the visibility state of a specific dataset in the 3D visualization.

        Args:
            dataset_key: Identifier for the dataset to toggle.

        Returns:
            True if toggle was successful, False if dataset key not found.
        """
        if dataset_key in self.datasets:
            self.datasets[dataset_key]["visible"] = not self.datasets[dataset_key]["visible"]
            return True
        return False

    def plot_image(self) -> None:
        """Generate comprehensive trajectory plots for the currently selected point.

        Creates a multi-panel plot showing position, velocity, and angular velocity
        components over time for detailed motion analysis.
        """
        dataset: Dict = self.datasets[self.current_dataset_key]

        # Create time axis arrays for plotting
        t: np.ndarray = np.arange(self.T) * self.dt_mean
        t_angular: np.ndarray = np.arange(self.T - 1) * self.dt_mean

        # Create comprehensive figure with five analysis subplots
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), dpi=100)

        # First subplot: Position trajectory comparison (original vs filtered)
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

        # Second subplot: Linear velocity components
        ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity")

        ax2.set_title(f"Linear Velocity of Point #{self.idx_point} - {dataset['display_name']}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlim(0, max(t))
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        # Third subplot: Angular velocity X component (raw vs filtered)
        ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red", label="Raw ωx")
        ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred", label="Filtered ωx")

        ax3.set_title(f"Angular Velocity X Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angular Velocity (rad/s)")
        ax3.set_xlim(0, max(t))
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right')

        # Fourth subplot: Angular velocity Y component (raw vs filtered)
        ax4.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green", label="Raw ωy")
        ax4.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
                 label="Filtered ωy")

        ax4.set_title(f"Angular Velocity Y Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angular Velocity (rad/s)")
        ax4.set_xlim(0, max(t))
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')

        # Fifth subplot: Angular velocity Z component (raw vs filtered)
        ax5.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 2], "--", c="blue", label="Raw ωz")
        ax5.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 2], "-", c="darkblue",
                 label="Filtered ωz")

        ax5.set_title(f"Angular Velocity Z Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Angular Velocity (rad/s)")
        ax5.set_xlim(0, max(t))
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend(loc='upper right')

        # Optimize subplot spacing for readability
        plt.tight_layout()

        # Convert plot to image for display in Polyscope interface
        buf: BytesIO = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        image: Image.Image = Image.open(buf).convert('RGB')
        rgb_array: np.ndarray = np.array(image)

        # Display image in Polyscope interface
        ps.add_color_image_quantity("plot", rgb_array / 255.0, enabled=True)
        plt.close(fig)

    def save_current_plot(self) -> str:
        """Save the current point's trajectory analysis plot to a file.

        Generates and saves a comprehensive plot file with timestamp
        for documentation and further analysis.

        Returns:
            String path to the saved plot file.
        """
        dataset: Dict = self.datasets[self.current_dataset_key]

        # Create time axis arrays
        t: np.ndarray = np.arange(self.T) * self.dt_mean
        t_angular: np.ndarray = np.arange(self.T - 1) * self.dt_mean

        # Create comprehensive analysis figure
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), dpi=100)

        # Plot position data with original and filtered comparisons
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

        # Plot linear velocity components
        ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity")
        ax2.set_title(f"Linear Velocity of Point #{self.idx_point} - {dataset['display_name']}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlim(0, max(t))
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        # Plot angular velocity X component with raw and filtered data
        ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red", label="Raw ωx")
        ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred", label="Filtered ωx")
        ax3.set_title(f"Angular Velocity X Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angular Velocity (rad/s)")
        ax3.set_xlim(0, max(t))
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right')

        # Plot angular velocity Y component with raw and filtered data
        ax4.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green", label="Raw ωy")
        ax4.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
                 label="Filtered ωy")
        ax4.set_title(f"Angular Velocity Y Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angular Velocity (rad/s)")
        ax4.set_xlim(0, max(t))
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')

        # Plot angular velocity Z component with raw and filtered data
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

        # Generate timestamped filename for unique file identification
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename: str = f"{self.output_dir}/{dataset['display_name']}_point_{self.idx_point}_t{self.t}_{timestamp}.png"

        # Save high-resolution plot to file
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Plot saved to: {filename}")
        return filename

    def callback(self) -> None:
        """Main Polyscope UI callback function for interactive parameter control.

        Handles all user interface elements including sliders, buttons, checkboxes,
        and tree nodes for comprehensive visualization and analysis control.
        """
        # Display main title and current dataset information
        psim.Text("Point Cloud Joint Analysis")
        psim.Text(f"Active Dataset: {self.datasets[self.current_dataset_key]['display_name']}")
        psim.Text(f"Detected Joint Type: {self.current_best_joint}")

        # Ground truth visualization controls
        changed_gt: bool
        changed_gt, self.show_ground_truth = psim.Checkbox("Show Ground Truth", self.show_ground_truth)
        if changed_gt:
            if self.show_ground_truth:
                self.visualize_ground_truth()
            else:
                self.remove_ground_truth_visualization()

        # Ground truth scale adjustment
        changed_gt_scale: bool
        changed_gt_scale, self.ground_truth_scale = psim.SliderFloat("Ground Truth Scale", self.ground_truth_scale, 0.1,
                                                                     2.0)
        if changed_gt_scale and self.show_ground_truth:
            self.remove_ground_truth_visualization()
            self.visualize_ground_truth()

        psim.Separator()

        # Main navigation controls for time and point selection
        self.t_changed, self.t = psim.SliderInt("Time Frame", self.t, 0, self.T - 1)
        self.idx_point_changed, self.idx_point = psim.SliderInt("Point Index", self.idx_point, 0, self.N - 1)

        # SVD analysis parameter control
        changed_neighbors: bool
        changed_neighbors, self.num_neighbors = psim.SliderInt("Neighbors for SVD", self.num_neighbors, 5, 30)

        # Collapsible joint analysis parameter section
        if psim.TreeNode("Joint Analysis Parameters"):
            # Collinearity analysis parameters
            changed_col_sigma: bool
            changed_col_sigma, self.col_sigma = psim.SliderFloat("Collinearity Sigma", self.col_sigma, 0.05, 0.5)
            changed_col_order: bool
            changed_col_order, self.col_order = psim.SliderFloat("Collinearity Order", self.col_order, 1.0, 10.0)

            # Coplanarity analysis parameters
            changed_cop_sigma: bool
            changed_cop_sigma, self.cop_sigma = psim.SliderFloat("Coplanarity Sigma", self.cop_sigma, 0.05, 0.5)
            changed_cop_order: bool
            changed_cop_order, self.cop_order = psim.SliderFloat("Coplanarity Order", self.cop_order, 1.0, 10.0)

            # Radius consistency analysis parameters
            changed_rad_sigma: bool
            changed_rad_sigma, self.rad_sigma = psim.SliderFloat("Radius Sigma", self.rad_sigma, 0.05, 0.5)
            changed_rad_order: bool
            changed_rad_order, self.rad_order = psim.SliderFloat("Radius Order", self.rad_order, 1.0, 10.0)

            # Zero pitch analysis parameters
            changed_zp_sigma: bool
            changed_zp_sigma, self.zp_sigma = psim.SliderFloat("Zero Pitch Sigma", self.zp_sigma, 0.05, 0.5)
            changed_zp_order: bool
            changed_zp_order, self.zp_order = psim.SliderFloat("Zero Pitch Order", self.zp_order, 1.0, 10.0)

            # Savitzky-Golay filter parameters
            changed_savgol_window: bool
            changed_savgol_window, self.savgol_window = psim.SliderInt("SG Window", self.savgol_window, 3, 31)
            changed_savgol_poly: bool
            changed_savgol_poly, self.savgol_poly = psim.SliderInt("SG Poly Order", self.savgol_poly, 1, 5)
            changed_savgol: bool
            changed_savgol, self.use_savgol = psim.Checkbox("Use SG Filter", self.use_savgol)

            # Check if any joint analysis parameter has changed
            params_changed: bool = (changed_col_sigma or changed_col_order or
                                    changed_cop_sigma or changed_cop_order or
                                    changed_rad_sigma or changed_rad_order or
                                    changed_zp_sigma or changed_zp_order or
                                    changed_savgol_window or changed_savgol_poly or
                                    changed_savgol)

            # Apply parameter changes button
            if params_changed:
                if psim.Button("Apply Parameter Changes"):
                    # Rerun joint analysis with updated parameters
                    joint_params: Dict
                    best_joint: str
                    info_dict: Dict
                    joint_params, best_joint, info_dict = self.perform_joint_analysis(
                        self.datasets[self.current_dataset_key]["data_filter"])

                    # Update dataset with new results
                    self.datasets[self.current_dataset_key]["joint_params"] = joint_params
                    self.datasets[self.current_dataset_key]["best_joint"] = best_joint
                    self.datasets[self.current_dataset_key]["joint_info"] = info_dict

                    # Update current analysis results
                    self.current_joint_params = joint_params
                    self.current_best_joint = best_joint
                    self.current_joint_info = info_dict

                    # Refresh visualization with new parameters
                    self.visualize_joint_parameters()

            psim.TreePop()

        # Update point cloud visualization when time frame changes
        if self.t_changed:
            for dataset_key, dataset in self.datasets.items():
                if dataset["visible"] and self.t < dataset["T"]:
                    register_point_cloud(
                        dataset["display_name"],
                        dataset["data_filter"][min(self.t, dataset["T"] - 1)],
                        radius=0.01,
                        enabled=True
                    )

        # Update trajectory plot when point selection changes
        if self.idx_point_changed:
            self.plot_image()

        # Dataset management section
        if psim.TreeNode("Dataset Selection"):
            # Provide controls for dataset selection and visibility
            for dataset_key, dataset in self.datasets.items():
                # Dataset selection button
                if psim.Button(f"Select {dataset['display_name']}"):
                    self.switch_dataset(dataset_key)

                psim.SameLine()

                # Visibility toggle button
                vis_text: str = "Visible" if dataset["visible"] else "Hidden"
                if psim.Button(f"{vis_text}###{dataset_key}"):
                    self.toggle_visibility(dataset_key)

                    # Update point cloud display based on visibility
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

        # Plot saving functionality
        if psim.Button("Save Current Plot"):
            saved_path: str = self.save_current_plot()
            psim.Text(f"Saved to: {saved_path}")

        # Joint reanalysis controls
        if changed_neighbors or psim.Button("Reanalyze Joint"):
            # Recalculate angular velocity for all datasets
            for dataset_key, dataset in self.datasets.items():
                angular_velocity_raw: np.ndarray
                angular_velocity_filtered: np.ndarray
                angular_velocity_raw, angular_velocity_filtered = self.calculate_angular_velocity(
                    dataset["data_filter"], dataset["N"]
                )
                dataset["angular_velocity_raw"] = angular_velocity_raw
                dataset["angular_velocity_filtered"] = angular_velocity_filtered

                # Perform fresh joint analysis
                joint_params: Dict
                best_joint: str
                info_dict: Dict
                joint_params, best_joint, info_dict = self.perform_joint_analysis(dataset["data_filter"])
                dataset["joint_params"] = joint_params
                dataset["best_joint"] = best_joint
                dataset["joint_info"] = info_dict

            # Update current dataset references
            self.angular_velocity_raw = self.datasets[self.current_dataset_key]["angular_velocity_raw"]
            self.angular_velocity_filtered = self.datasets[self.current_dataset_key]["angular_velocity_filtered"]
            self.current_joint_params = self.datasets[self.current_dataset_key]["joint_params"]
            self.current_best_joint = self.datasets[self.current_dataset_key]["best_joint"]
            self.current_joint_info = self.datasets[self.current_dataset_key]["joint_info"]

            # Refresh all visualizations
            self.plot_image()
            self.visualize_joint_parameters()

            psim.Text("Joint analysis recalculated")

        # Ground truth comparison section
        if psim.TreeNode("Ground Truth Information"):
            dataset: Dict = self.datasets[self.current_dataset_key]
            gt_key: Optional[Dict[str, str]] = dataset["ground_truth_key"]

            if gt_key:
                scene: str = gt_key["scene"]
                object_type: str = gt_key["object"]
                part_info: str = gt_key["part"]

                # Display ground truth metadata
                psim.Text(f"Scene: {scene}")
                psim.Text(f"Object Type: {object_type}")
                psim.Text(f"Part: {part_info}")

                # Display ground truth parameters if available
                if scene in self.ground_truth_data and object_type in self.ground_truth_data[scene]:
                    if part_info in self.ground_truth_data[scene][object_type]:
                        joint_info: Dict = self.ground_truth_data[scene][object_type][part_info]

                        # Display ground truth axis information
                        if "axis" in joint_info:
                            axis: List[float] = joint_info["axis"]
                            psim.Text(f"Ground Truth Axis: [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")

                            # Calculate and display normalized axis
                            axis_norm: float = np.linalg.norm(axis)
                            if axis_norm > 1e-6:
                                norm_axis: List[float] = [ax / axis_norm for ax in axis]
                                psim.Text(f"Normalized: [{norm_axis[0]:.4f}, {norm_axis[1]:.4f}, {norm_axis[2]:.4f}]")

                        # Display ground truth pivot information
                        if "pivot" in joint_info:
                            pivot = joint_info["pivot"]
                            if isinstance(pivot, list):
                                if len(pivot) == 1:
                                    psim.Text(f"Ground Truth Pivot (parameter): {pivot[0]:.4f}")
                                elif len(pivot) == 3:
                                    psim.Text(f"Ground Truth Pivot: [{pivot[0]:.4f}, {pivot[1]:.4f}, {pivot[2]:.4f}]")
                            else:
                                psim.Text(f"Ground Truth Pivot (parameter): {pivot:.4f}")

                        # Comparison with estimated joint parameters
                        if self.current_best_joint in self.current_joint_params:
                            psim.Text("\nComparison with Estimated Joint:")
                            params: Dict = self.current_joint_params[self.current_best_joint]

                            if self.current_best_joint == "revolute":
                                psim.Text(f"Estimated Type: {self.current_best_joint.capitalize()}")

                                # Compare axes if ground truth axis data exists
                                if "axis" in joint_info:
                                    gt_axis: np.ndarray = np.array(joint_info["axis"])
                                    gt_axis_norm: float = np.linalg.norm(gt_axis)
                                    if gt_axis_norm > 1e-6:
                                        gt_axis = gt_axis / gt_axis_norm

                                    est_axis: np.ndarray = np.array(params.get("axis", [0, 0, 0]))
                                    est_axis_norm: float = np.linalg.norm(est_axis)
                                    if est_axis_norm > 1e-6:
                                        est_axis = est_axis / est_axis_norm

                                    # Calculate angular difference between axes
                                    dot_product: float = np.clip(np.abs(np.dot(gt_axis, est_axis)), 0.0, 1.0)
                                    angle_diff: float = np.arccos(dot_product)
                                    angle_diff_deg: float = np.degrees(angle_diff)

                                    psim.Text(f"Axis Angle Difference: {angle_diff_deg:.2f}°")

                                    # Handle near-opposite axes case
                                    if angle_diff_deg > 90:
                                        angle_diff_deg = 180 - angle_diff_deg
                                        psim.Text(f"Axes are nearly opposite. Adjusted angle: {angle_diff_deg:.2f}°")

                                    # Calculate spatial distance between axis lines
                                    est_origin: np.ndarray = np.array(params.get("origin", [0, 0, 0]))

                                    # Process ground truth pivot data
                                    gt_pivot = joint_info.get("pivot", [0, 0, 0])
                                    if isinstance(gt_pivot, list):
                                        if len(gt_pivot) == 1:
                                            # Single parameter pivot along axis from origin
                                            gt_origin: np.ndarray = np.array([0., 0., 0.]) + float(
                                                gt_pivot[0]) * gt_axis
                                        elif len(gt_pivot) == 3:
                                            gt_origin = np.array(gt_pivot)
                                        else:
                                            gt_origin = np.array([0., 0., 0.])
                                    else:
                                        # Numeric pivot value
                                        gt_origin = np.array([0., 0., 0.]) + float(gt_pivot) * gt_axis

                                    # Calculate distance between axis lines
                                    cross_product: np.ndarray = np.cross(est_axis, gt_axis)
                                    cross_norm: float = np.linalg.norm(cross_product)

                                    if cross_norm < 1e-6:
                                        # Parallel axes - calculate point-to-line distance
                                        vec_to_point: np.ndarray = est_origin - gt_origin
                                        proj_on_axis: np.ndarray = np.dot(vec_to_point, gt_axis) * gt_axis
                                        perpendicular: np.ndarray = vec_to_point - proj_on_axis
                                        axis_distance: float = np.linalg.norm(perpendicular)
                                        psim.Text(f"Axis Distance (parallel): {axis_distance:.4f} m")
                                    else:
                                        # Skew axes - calculate minimum distance between lines
                                        vec_between: np.ndarray = gt_origin - est_origin
                                        axis_distance = abs(np.dot(vec_between, cross_product)) / cross_norm
                                        psim.Text(f"Axis Distance (skew): {axis_distance:.4f} m")

                                    # Calculate distance between origin points
                                    origin_distance: float = np.linalg.norm(est_origin - gt_origin)
                                    psim.Text(f"Origin Distance: {origin_distance:.4f} m")
                    else:
                        psim.Text(f"Part {part_info} not found in ground truth for {object_type}")
                else:
                    psim.Text("Scene or object not found in ground truth data")
            else:
                psim.Text("No ground truth mapping found for this dataset")

            psim.TreePop()

        # Coordinate system and display settings
        if psim.TreeNode("Coordinate System Settings"):
            # Coordinate system orientation options
            options: List[str] = ["y_up", "z_up"]
            for i, option in enumerate(options):
                is_selected: bool = (ps.get_up_dir() == option)
                changed: bool
                changed, is_selected = psim.Checkbox(f"{option}", is_selected)
                if changed and is_selected:
                    ps.set_up_dir(option)

            # Coordinate frame scale adjustment
            changed_scale: bool
            changed_scale, self.coord_scale = psim.SliderFloat("Coord Frame Scale", self.coord_scale, 0.05, 1.0)
            if changed_scale:
                # Redraw origin coordinate frame with new scale
                draw_frame_3d(np.zeros(6), label="origin", scale=self.coord_scale)

            # Ground plane visualization options
            ground_modes: List[str] = ["none", "tile", "shadow_only"]
            current_mode: str = ps.get_ground_plane_mode()
            for mode in ground_modes:
                is_selected = (current_mode == mode)
                changed, is_selected = psim.Checkbox(f"Ground: {mode}", is_selected)
                if changed and is_selected:
                    ps.set_ground_plane_mode(mode)

            psim.TreePop()

        # Detailed joint analysis information display
        if psim.TreeNode("Joint Information"):
            # Display basic scoring metrics
            if self.current_joint_info and "basic_score_avg" in self.current_joint_info:
                psim.Text("Basic Scores:")
                basic_scores: Dict = self.current_joint_info["basic_score_avg"]
                psim.Text(f"Collinearity: {basic_scores.get('col_mean', 0.0):.3f}")
                psim.Text(f"Coplanarity: {basic_scores.get('cop_mean', 0.0):.3f}")
                psim.Text(f"Radius Consistency: {basic_scores.get('rad_mean', 0.0):.3f}")
                psim.Text(f"Zero Pitch: {basic_scores.get('zp_mean', 0.0):.3f}")

                # Display joint type probabilities
                psim.Text("\nJoint Probabilities:")
                joint_probs: Dict = self.current_joint_info.get("joint_probs", {})
                for joint_type, prob in joint_probs.items():
                    psim.Text(f"{joint_type.capitalize()}: {prob:.3f}")

                # Display detailed parameters for best matching joint
                if self.current_best_joint in self.current_joint_params:
                    psim.Text(f"\n{self.current_best_joint.capitalize()} Joint Parameters:")
                    params = self.current_joint_params[self.current_best_joint]

                    # Display joint-specific parameter information
                    if self.current_best_joint == "planar":
                        normal: List[float] = params.get("normal", [0, 0, 0])
                        motion_limit: Tuple[float, float] = params.get("motion_limit", (0, 0))
                        psim.Text(f"Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
                        psim.Text(f"Motion Limit: ({motion_limit[0]:.3f}, {motion_limit[1]:.3f})")

                    elif self.current_best_joint == "ball":
                        center: List[float] = params.get("center", [0, 0, 0])
                        radius: float = params.get("radius", 0)
                        motion_limit: Tuple[float, float, float] = params.get("motion_limit", (0, 0, 0))
                        psim.Text(f"Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
                        psim.Text(f"Radius: {radius:.3f}")
                        psim.Text(
                            f"Motion Limit: ({motion_limit[0]:.3f}, {motion_limit[1]:.3f}, {motion_limit[2]:.3f}) rad")

                    elif self.current_best_joint == "screw":
                        axis: List[float] = params.get("axis", [0, 0, 0])
                        origin: List[float] = params.get("origin", [0, 0, 0])
                        pitch: float = params.get("pitch", 0)
                        motion_limit: Tuple[float, float] = params.get("motion_limit", (0, 0))
                        psim.Text(f"Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
                        psim.Text(f"Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
                        psim.Text(f"Pitch: {pitch:.3f}")
                        psim.Text(f"Motion Limit: ({motion_limit[0]:.3f}, {motion_limit[1]:.3f}) rad")

                    elif self.current_best_joint == "prismatic":
                        axis: List[float] = params.get("axis", [0, 0, 0])
                        origin: List[float] = params.get("origin", [0, 0, 0])
                        motion_limit: Tuple[float, float] = params.get("motion_limit", (0, 0))
                        psim.Text(f"Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
                        psim.Text(f"Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
                        psim.Text(f"Motion Limit: ({motion_limit[0]:.3f}, {motion_limit[1]:.3f}) m")

                    elif self.current_best_joint == "revolute":
                        axis: List[float] = params.get("axis", [0, 0, 0])
                        origin: List[float] = params.get("origin", [0, 0, 0])
                        motion_limit: Tuple[float, float] = params.get("motion_limit", (0, 0))
                        psim.Text(f"Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
                        psim.Text(f"Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
                        psim.Text(f"Motion Limit: ({motion_limit[0]:.3f}, {motion_limit[1]:.3f}) rad")
                        psim.Text(f"Motion Range: {np.degrees(motion_limit[1] - motion_limit[0]):.2f}°")

            psim.TreePop()


# Program entry point and initialization
if __name__ == "__main__":
    """Main execution block for running the enhanced joint analysis visualization.

    Initializes the visualization system with specified data files and launches
    the interactive Polyscope interface for comprehensive joint motion analysis.
    """
    # Specify multiple data file paths for analysis
    file_paths: List[str] = [
        "./parahome_data_slide//s190_chair_base_270_390.npy"
    ]

    # Create and launch the enhanced visualization system
    viz: EnhancedViz = EnhancedViz(file_paths)