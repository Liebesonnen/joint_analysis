import copy
import time
import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add joint_analysis module to path
sys.path.append('/common/homes/all/uksqc_chen/projects/control/')

import torch
import click
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

from robot_utils import console
from robot_utils.cv.image.masks import erode_mask
from robot_utils.cv.io.io_cv_batch import load_rgb_batch, load_depth_batch, load_mask_batch
from robot_utils.cv.geom.projection import pinhole_projection_image_to_camera as proj_img_to_cam
from robot_utils.py.interact import ask_checkbox_with_all
from robot_utils.py.filesystem import get_validate_file, get_validate_path, get_ordered_subdirs
from robot_utils.serialize.dataclass import load_dict_from_yaml
from robot_utils.serialize.schema_numpy import DictNumpyArray
from robot_vision.human_pose.utils.viz_hand import create_local_frame_on_hand_heuristic
from robot_vision.human_pose.utils.constants import mano_hand_edges as hand_edges
from robot_utils.viz.polyscope import ps, psim, PolyscopeUtils, register_point_cloud, register_curve_network, \
    draw_frame_3d, ps_select
from robot_vision.cam.data import DemoMetaInfo
from kvil_demo.cfg.preprocessing import DemoFileStructure
from obj_can_space.template.obj_template import compute_neighbor_health_score
from obj_can_space.template.obj_template import RepairedObjData

# Import joint analysis module
try:
    from joint_analysis0.core.joint_estimation import compute_joint_info_all_types

    JOINT_ANALYSIS_AVAILABLE = True
    console.log("Successfully imported joint_analysis module")
except ImportError as e:
    console.log(f"Warning: Could not import joint_analysis module: {e}")
    console.log("Falling back to simple joint analyzer")
    JOINT_ANALYSIS_AVAILABLE = False


class SimpleJointAnalyzer:
    """Simple joint analysis without external dependencies"""

    def __init__(self):
        self.joint_types = ["revolute", "prismatic", "planar", "ball", "unknown"]

    def analyze_joint(self, trajectory_data):
        """
        Analyze joint type from trajectory data
        Args:
            trajectory_data: numpy array of shape (T, N, 3) - T timesteps, N points, 3D coordinates
        Returns:
            tuple: (joint_params, best_joint_type, info_dict)
        """
        T, N, _ = trajectory_data.shape

        if T < 5 or N < 10:
            return {}, "unknown", {}

        # Calculate motion vectors for each point across time
        motion_vectors = np.diff(trajectory_data, axis=0)  # (T-1, N, 3)

        # Calculate total displacement for each point
        total_displacement = trajectory_data[-1] - trajectory_data[0]  # (N, 3)
        displacement_magnitudes = np.linalg.norm(total_displacement, axis=1)

        # Find points that move significantly
        moving_threshold = np.percentile(displacement_magnitudes, 70)
        moving_points = displacement_magnitudes > moving_threshold

        if np.sum(moving_points) < 5:
            return {}, "unknown", {}

        # Analyze motion patterns
        joint_scores = {}
        joint_params = {}

        # Test for revolute joint
        revolute_score, revolute_params = self._test_revolute_joint(trajectory_data, moving_points)
        joint_scores["revolute"] = revolute_score
        joint_params["revolute"] = revolute_params

        # Test for prismatic joint
        prismatic_score, prismatic_params = self._test_prismatic_joint(trajectory_data, moving_points)
        joint_scores["prismatic"] = prismatic_score
        joint_params["prismatic"] = prismatic_params

        # Test for planar joint
        planar_score, planar_params = self._test_planar_joint(trajectory_data, moving_points)
        joint_scores["planar"] = planar_score
        joint_params["planar"] = planar_params

        # Test for ball joint
        ball_score, ball_params = self._test_ball_joint(trajectory_data, moving_points)
        joint_scores["ball"] = ball_score
        joint_params["ball"] = ball_params

        # Determine best joint type
        best_joint = max(joint_scores.keys(), key=lambda k: joint_scores[k])

        # Create info dictionary
        info_dict = {
            "joint_probs": joint_scores,
            "basic_score_avg": {
                "revolute_score": revolute_score,
                "prismatic_score": prismatic_score,
                "planar_score": planar_score,
                "ball_score": ball_score
            }
        }

        return joint_params, best_joint, info_dict

    def _test_revolute_joint(self, trajectory_data, moving_points):
        """Test if motion follows revolute joint pattern"""
        T, N, _ = trajectory_data.shape

        # Get trajectories of moving points
        moving_trajectories = trajectory_data[:, moving_points, :]  # (T, n_moving, 3)

        if moving_trajectories.shape[1] < 5:
            return 0.0, {}

        # For revolute joint, points should move in circular arcs around a common axis
        best_score = 0.0
        best_params = {}

        # Try to find the rotation axis by analyzing motion
        try:
            # Calculate centroid of moving points at each timestep
            centroids = np.mean(moving_trajectories, axis=1)  # (T, 3)

            # Estimate rotation center as mean of centroids
            rotation_center = np.mean(centroids, axis=0)

            # For each point, check if it follows circular motion
            scores = []
            for i in range(moving_trajectories.shape[1]):
                point_traj = moving_trajectories[:, i, :]  # (T, 3)

                # Calculate distances from rotation center
                distances = np.linalg.norm(point_traj - rotation_center, axis=1)

                # For revolute joint, distance should be relatively constant
                distance_var = np.var(distances) / (np.mean(distances) + 1e-6)
                circular_score = 1.0 / (1.0 + distance_var * 10)
                scores.append(circular_score)

            revolute_score = np.mean(scores)

            # Estimate rotation axis using PCA of the motion vectors
            motion_vectors = np.diff(moving_trajectories, axis=0)  # (T-1, n_moving, 3)
            motion_vectors_flat = motion_vectors.reshape(-1, 3)

            if motion_vectors_flat.shape[0] > 3:
                pca = PCA(n_components=3)
                pca.fit(motion_vectors_flat)

                # The rotation axis should be perpendicular to the main motion directions
                # So we take the component with smallest variance
                rotation_axis = pca.components_[-1]  # Last principal component
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

                # Calculate motion limits
                angles = []
                for i in range(T - 1):
                    p1 = moving_trajectories[i, :, :] - rotation_center
                    p2 = moving_trajectories[i + 1, :, :] - rotation_center

                    # Project points onto plane perpendicular to rotation axis
                    p1_proj = p1 - np.outer(np.dot(p1, rotation_axis), rotation_axis)
                    p2_proj = p2 - np.outer(np.dot(p2, rotation_axis), rotation_axis)

                    # Calculate angle between projections
                    for j in range(p1_proj.shape[0]):
                        if np.linalg.norm(p1_proj[j]) > 1e-6 and np.linalg.norm(p2_proj[j]) > 1e-6:
                            cos_angle = np.dot(p1_proj[j], p2_proj[j]) / (
                                        np.linalg.norm(p1_proj[j]) * np.linalg.norm(p2_proj[j]))
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.arccos(cos_angle)
                            angles.append(angle)

                total_angle = np.sum(angles) if angles else 0.0
                motion_limit = (0.0, total_angle)

                best_params = {
                    "axis": rotation_axis.tolist(),
                    "origin": rotation_center.tolist(),
                    "motion_limit": motion_limit
                }
                best_score = revolute_score

        except Exception as e:
            console.log(f"Error in revolute joint analysis: {e}")
            best_score = 0.0
            best_params = {}

        return best_score, best_params

    def _test_prismatic_joint(self, trajectory_data, moving_points):
        """Test if motion follows prismatic joint pattern"""
        T, N, _ = trajectory_data.shape

        # Get trajectories of moving points
        moving_trajectories = trajectory_data[:, moving_points, :]

        if moving_trajectories.shape[1] < 5:
            return 0.0, {}

        try:
            # For prismatic joint, all points should move in the same direction
            # Calculate displacement vectors for each point
            displacements = moving_trajectories[-1] - moving_trajectories[0]  # (n_moving, 3)

            # Normalize displacements
            displacement_magnitudes = np.linalg.norm(displacements, axis=1)
            valid_displacements = displacement_magnitudes > 1e-6

            if np.sum(valid_displacements) < 3:
                return 0.0, {}

            normalized_displacements = displacements[valid_displacements] / displacement_magnitudes[
                valid_displacements, np.newaxis]

            # Calculate consistency of displacement directions
            mean_direction = np.mean(normalized_displacements, axis=0)
            mean_direction = mean_direction / np.linalg.norm(mean_direction)

            # Calculate how well each displacement aligns with mean direction
            alignment_scores = np.dot(normalized_displacements, mean_direction)
            prismatic_score = np.mean(np.abs(alignment_scores))

            # Calculate motion limits
            projected_distances = np.dot(displacements, mean_direction)
            motion_range = (np.min(projected_distances), np.max(projected_distances))

            # Estimate origin as mean of initial positions
            origin = np.mean(moving_trajectories[0], axis=0)

            params = {
                "axis": mean_direction.tolist(),
                "origin": origin.tolist(),
                "motion_limit": motion_range
            }

        except Exception as e:
            console.log(f"Error in prismatic joint analysis: {e}")
            prismatic_score = 0.0
            params = {}

        return prismatic_score, params

    def _test_planar_joint(self, trajectory_data, moving_points):
        """Test if motion is constrained to a plane"""
        T, N, _ = trajectory_data.shape

        moving_trajectories = trajectory_data[:, moving_points, :]

        if moving_trajectories.shape[1] < 5:
            return 0.0, {}

        try:
            # Flatten all positions of moving points
            all_positions = moving_trajectories.reshape(-1, 3)

            # Use PCA to find the plane that best fits all positions
            pca = PCA(n_components=3)
            pca.fit(all_positions)

            # For planar motion, one principal component should have very small variance
            explained_variance_ratio = pca.explained_variance_ratio_

            # If the third component has very small variance, motion is likely planar
            planar_score = 1.0 - explained_variance_ratio[2]

            # The normal vector is the third principal component
            normal_vector = pca.components_[2]
            normal_vector = normal_vector / np.linalg.norm(normal_vector)

            # Calculate motion limits in the plane
            projected_positions = all_positions @ pca.components_[:2].T
            x_range = (np.min(projected_positions[:, 0]), np.max(projected_positions[:, 0]))
            y_range = (np.min(projected_positions[:, 1]), np.max(projected_positions[:, 1]))

            params = {
                "normal": normal_vector.tolist(),
                "motion_limit": (np.sqrt((x_range[1] - x_range[0]) ** 2 + (y_range[1] - y_range[0]) ** 2), 0)
            }

        except Exception as e:
            console.log(f"Error in planar joint analysis: {e}")
            planar_score = 0.0
            params = {}

        return planar_score, params

    def _test_ball_joint(self, trajectory_data, moving_points):
        """Test if motion follows ball joint pattern"""
        T, N, _ = trajectory_data.shape

        moving_trajectories = trajectory_data[:, moving_points, :]

        if moving_trajectories.shape[1] < 5:
            return 0.0, {}

        try:
            # For ball joint, points should maintain relatively constant distance from joint center
            # but can move in multiple directions

            # Estimate joint center as the point that minimizes distance variance across all trajectories
            all_positions = moving_trajectories.reshape(-1, 3)
            center_estimate = np.mean(all_positions, axis=0)

            # Calculate distances from center for each point at each timestep
            distances = np.linalg.norm(moving_trajectories - center_estimate, axis=2)

            # For ball joint, each point should maintain relatively constant distance
            distance_consistency_scores = []
            for i in range(moving_trajectories.shape[1]):
                point_distances = distances[:, i]
                consistency = 1.0 / (1.0 + np.var(point_distances) / (np.mean(point_distances) + 1e-6))
                distance_consistency_scores.append(consistency)

            ball_score = np.mean(distance_consistency_scores)

            # Calculate average radius
            avg_radius = np.mean(distances)

            # Calculate motion limits (angular ranges)
            angular_ranges = []
            for i in range(moving_trajectories.shape[1]):
                point_traj = moving_trajectories[:, i, :] - center_estimate
                # Calculate angles between consecutive positions
                angles = []
                for t in range(T - 1):
                    v1 = point_traj[t]
                    v2 = point_traj[t + 1]
                    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)
                        angles.append(angle)
                if angles:
                    angular_ranges.append(np.sum(angles))

            max_angular_range = np.max(angular_ranges) if angular_ranges else 0.0

            params = {
                "center": center_estimate.tolist(),
                "radius": float(avg_radius),
                "motion_limit": (0.0, 0.0, max_angular_range)
            }

        except Exception as e:
            console.log(f"Error in ball joint analysis: {e}")
            ball_score = 0.0
            params = {}

        return ball_score, params


class EnhancedVizKVILData:
    def __init__(
            self,
            path: Path,
            namespace: str,
            shared_data: dict = None,
            auto_viz_all: bool = False
    ):
        self.shared_data = shared_data
        self.pu = PolyscopeUtils()
        console.log("visualize demo in 3D with joint analysis")

        self.p = DemoFileStructure(
            task_path=path, namespace=namespace
        )
        self.p.trial_idx = 0

        self._load_traj()
        self._chose_objects(auto_viz_all)
        self._load_images()

        self.ps_group = {}
        self.ps_group_flag = {}
        for obj in self.obj_list:
            self.ps_group[obj] = ps.create_group(obj)
            self.ps_group_flag[obj] = True
            self.ps_group[obj].set_enabled(True)

        self.selected_point_idx, self.selected_point_idx_changed = 0, False

        self._load_can_space_cmap()
        self._load_can_neighbors()
        self._reset_track_data()
        self._load_intrinsics()
        self._load_obj_raw_pcl()

        # Joint analysis initialization
        self._init_joint_analysis()

        # Note: set bbox of viz scene and reset_cam_orbit
        self.pu.reset_bbox_from_pcl_list([self.pcd_raw[obj][0] for obj in self.obj_list if "hand_person" not in obj])
        self.pu.viz_bounding_box()
        self.pu.reset_cam_orbit_from_bbox()

        origin = draw_frame_3d(np.zeros(6, dtype=float), scale=0.1, radius=0.005, alpha=0.8)
        self.current_pcl_points = {}
        self.current_pcl_points_new = {}
        self.current_pcl_points_fixed = {}
        self.frame_mat_list = {}
        self.hand_curve_network = {}

        # Note: viz candidate points and hand skeletons
        for i, obj_name in enumerate(self.obj_list):
            traj = self.xyz_traj_filtered[obj_name]
            self.t_max, self.n_points = traj.shape[:2]
            self.update_filtered_traj(obj_name, 0)

        self.pt_idx_max = self.xyz_traj_filtered[self.obj_list[0]].shape[1]

        self._load_global_frame()
        self.pu.look_at(
            cam_position=np.array([0, 0, -0.4]), target=self.center
        )

        self.time_changed, self.current_time = False, 0
        self.pt_idx_changed, self.pt_idx = False, 0
        self.flag_local_frame_changed, self.flag_local_frame = False, False
        self.master_obj = self.obj_list[0]
        self.master_obj_changed: bool = False

        self.enable_raw_flag_changed = False
        self.enable_raw_flag = self.shared_data["enable_raw_flag"] if self.shared_data is not None else False
        self.enable_filtered_flag_changed = False
        self.enable_filtered_flag = self.shared_data["enable_filtered_flag"] if self.shared_data is not None else False

        # Perform joint analysis after loading all data
        self._perform_joint_analysis_all_objects()

    def _init_joint_analysis(self):
        """Initialize joint analysis parameters and data structures"""
        # Joint analysis parameters (for both simple and advanced analyzers)
        self.num_neighbors = 50
        self.col_sigma = 1
        self.col_order = 4.0
        self.cop_sigma = 1
        self.cop_order = 4.0
        self.rad_sigma = 0.2
        self.rad_order = 4.0
        self.zp_sigma = 1
        self.zp_order = 4.0
        self.use_savgol = True
        self.savgol_window = 21
        self.savgol_poly = 2

        # Initialize the appropriate joint analyzer
        if JOINT_ANALYSIS_AVAILABLE:
            console.log("Using advanced joint_analysis module")
            self.use_advanced_analyzer = True
        else:
            console.log("Using simple joint analyzer")
            self.joint_analyzer = SimpleJointAnalyzer()
            self.use_advanced_analyzer = False

        # Joint analysis results storage
        self.joint_params = {}  # {obj_name: joint_params}
        self.best_joint = {}  # {obj_name: best_joint_type}
        self.joint_info = {}  # {obj_name: info_dict}

        # Joint visualization flags
        self.show_joint_viz = {}  # {obj_name: bool}
        self.joint_viz_scale = 0.2

        # Initialize show_joint_viz for all objects
        for obj in self.obj_list:
            self.show_joint_viz[obj] = False

        # Ground truth data (if available)
        self.ground_truth_json = "./all_scenes_transformed_axis_pivot_data.json"
        self.ground_truth_data = self.load_ground_truth()
        self.show_ground_truth = False
        self.ground_truth_scale = 0.5
        self.gt_curve_networks = []
        self.gt_point_clouds = []

    def load_ground_truth(self):
        """Load ground truth data from JSON file"""
        try:
            if os.path.exists(self.ground_truth_json):
                with open(self.ground_truth_json, 'r') as f:
                    data = json.load(f)
                    console.log(f"Successfully loaded ground truth data from: {self.ground_truth_json}")
                    console.log(f"Available scenes: {list(data.keys())}")
                    return data
            else:
                console.log(f"Ground truth file not found at {self.ground_truth_json}")
                return {}
        except Exception as e:
            console.log(f"Error loading ground truth data: {e}")
            return {}

    def extract_scene_info_from_namespace(self, namespace):
        """Extract scene and object information from namespace"""
        # Pattern to match namespaces like "s204_drawer_part2" or similar
        pattern = r'(s\d+)_([^_]+)_(part\d+|base)'
        match = re.match(pattern, namespace)

        if match:
            scene, object_type, part = match.groups()
            return {
                "scene": scene,
                "object": object_type,
                "part": part
            }

        console.log(f"Warning: Could not extract scene info from namespace: {namespace}")
        return None

    def map_namespace_to_ground_truth(self, namespace):
        """Map namespace to ground truth key"""
        scene_info = self.extract_scene_info_from_namespace(namespace)

        if not scene_info or not self.ground_truth_data:
            return None

        scene = scene_info["scene"]
        object_type = scene_info["object"]
        part = scene_info["part"]

        # Check if this scene exists in ground truth data
        if scene not in self.ground_truth_data:
            console.log(f"Warning: Scene {scene} not found in ground truth data")
            return None

        # Check if this object exists in the scene
        if object_type not in self.ground_truth_data[scene]:
            console.log(f"Warning: Object {object_type} not found in scene {scene} ground truth data")
            return None

        # Check if this part exists for the object
        if part not in self.ground_truth_data[scene][object_type]:
            console.log(f"Warning: Part {part} not found for {object_type} in scene {scene}")
            return None

        console.log(f"Mapped {namespace} -> Scene: {scene}, Object: {object_type}, Part: {part}")
        return {
            "scene": scene,
            "object": object_type,
            "part": part
        }

    def _perform_joint_analysis_all_objects(self):
        """Perform joint analysis for all objects"""
        console.log("Starting joint analysis for all objects...")

        for obj_name in self.obj_list:
            if "hand_person" in obj_name:
                continue  # Skip hand objects

            console.log(f"Analyzing joint for object: {obj_name}")

            # Get trajectory data for this object
            traj_data = self.xyz_traj_filtered[obj_name]  # Shape: (T, N, 3)

            # Skip if not enough data
            if traj_data.shape[0] < 10:
                console.log(f"Not enough trajectory data for {obj_name}, skipping joint analysis")
                continue

            try:
                if self.use_advanced_analyzer:
                    # Use advanced joint analysis module
                    joint_params, best_joint, info_dict = self.perform_advanced_joint_analysis(traj_data)
                else:
                    # Use simple joint analyzer
                    joint_params, best_joint, info_dict = self.joint_analyzer.analyze_joint(traj_data)

                # Store results
                self.joint_params[obj_name] = joint_params
                self.best_joint[obj_name] = best_joint
                self.joint_info[obj_name] = info_dict

                console.log(f"Joint analysis completed for {obj_name}: {best_joint}")

                # Print joint scores
                if "joint_probs" in info_dict:
                    for joint_type, score in info_dict["joint_probs"].items():
                        console.log(f"  {joint_type}: {score:.3f}")

            except Exception as e:
                console.log(f"Error in joint analysis for {obj_name}: {e}")
                continue

    def perform_advanced_joint_analysis(self, point_history):
        """Perform joint analysis using the advanced joint_analysis module"""
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

        # Print joint analysis results
        console.log("\n" + "=" * 80)
        console.log(f"Joint Type: {best_joint}")
        console.log("=" * 80)

        # Print basic scores
        if info_dict and "basic_score_avg" in info_dict:
            basic_scores = info_dict["basic_score_avg"]
            console.log("\nBasic Scores:")
            console.log(f"Collinearity Score: {basic_scores.get('col_mean', 0.0):.16f}")
            console.log(f"Coplanarity Score: {basic_scores.get('cop_mean', 0.0):.16f}")
            console.log(f"Radius Consistency Score: {basic_scores.get('rad_mean', 0.0):.16f}")
            console.log(f"Zero Pitch Score: {basic_scores.get('zp_mean', 0.0):.16f}")

        # Print joint probabilities
        if info_dict and "joint_probs" in info_dict:
            joint_probs = info_dict["joint_probs"]
            console.log("\nJoint Probabilities:")
            for joint_type, prob in joint_probs.items():
                console.log(f"{joint_type.capitalize()}: {prob:.16f}")

        # Print joint parameters
        if best_joint in joint_params:
            params = joint_params[best_joint]
            console.log(f"\n{best_joint.capitalize()} Joint Parameters:")

            if best_joint == "planar":
                normal = params.get("normal", [0, 0, 0])
                motion_limit = params.get("motion_limit", (0, 0))
                console.log(f"Normal Vector: [{normal[0]:.16f}, {normal[1]:.16f}, {normal[2]:.16f}]")
                console.log(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f})")

            elif best_joint == "ball":
                center = params.get("center", [0, 0, 0])
                radius = params.get("radius", 0)
                motion_limit = params.get("motion_limit", (0, 0, 0))
                console.log(f"Center: [{center[0]:.16f}, {center[1]:.16f}, {center[2]:.16f}]")
                console.log(f"Radius: {radius:.16f}")
                console.log(
                    f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}, {motion_limit[2]:.16f}) rad")

            elif best_joint == "screw":
                axis = params.get("axis", [0, 0, 0])
                origin = params.get("origin", [0, 0, 0])
                pitch = params.get("pitch", 0)
                motion_limit = params.get("motion_limit", (0, 0))
                console.log(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]")
                console.log(f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]")
                console.log(f"Pitch: {pitch:.16f}")
                console.log(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) rad")

            elif best_joint == "prismatic":
                axis = params.get("axis", [0, 0, 0])
                origin = params.get("origin", [0, 0, 0])
                motion_limit = params.get("motion_limit", (0, 0))
                console.log(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]")
                console.log(f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]")
                console.log(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) m")

            elif best_joint == "revolute":
                axis = params.get("axis", [0, 0, 0])
                origin = params.get("origin", [0, 0, 0])
                motion_limit = params.get("motion_limit", (0, 0))
                console.log(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]")
                console.log(f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]")
                console.log(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) rad")
                console.log(f"Motion Range: {np.degrees(motion_limit[1] - motion_limit[0]):.16f}°")

        console.log("=" * 80 + "\n")

        return joint_params, best_joint, info_dict

    def visualize_joint_parameters(self, obj_name):
        """Visualize estimated joint parameters in Polyscope for a specific object"""
        if obj_name not in self.best_joint:
            return

        # Remove any existing joint visualizations for this object
        self.remove_joint_visualization(obj_name)

        joint_type = self.best_joint[obj_name]

        if joint_type in self.joint_params[obj_name]:
            joint_params = self.joint_params[obj_name][joint_type]
            self.show_joint_visualization(obj_name, joint_type, joint_params)

    def remove_joint_visualization(self, obj_name):
        """Remove joint visualization elements for a specific object"""
        joint_viz_names = [
            f"{obj_name}_joint_axis",
            f"{obj_name}_joint_origin",
            f"{obj_name}_joint_normal",
            f"{obj_name}_joint_center",
            f"{obj_name}_joint_pitch"
        ]

        for name in joint_viz_names:
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)
            if ps.has_point_cloud(name):
                ps.remove_point_cloud(name)

    def show_joint_visualization(self, obj_name, joint_type, joint_params):
        """Show visualization for a specific joint type and object"""
        scale = self.joint_viz_scale

        if joint_type == "revolute":
            # Extract parameters
            axis_np = np.array(joint_params.get("axis", [0., 1., 0.]))
            origin_np = np.array(joint_params.get("origin", [0., 0., 0.]))

            # Normalize axis if needed
            axis_norm = np.linalg.norm(axis_np)
            if axis_norm > 1e-6:
                axis_np = axis_np / axis_norm

            # Visualize axis
            seg_nodes = np.array([origin_np - axis_np * scale, origin_np + axis_np * scale])
            seg_edges = np.array([[0, 1]])

            axis_viz = ps.register_curve_network(f"{obj_name}_joint_axis", seg_nodes, seg_edges)
            axis_viz.set_radius(0.01)
            axis_viz.set_color((1., 1., 0.))  # Yellow for revolute joint
            axis_viz.add_to_group(obj_name)

            # Visualize origin point
            origin_viz = ps.register_curve_network(
                f"{obj_name}_joint_origin",
                np.array([origin_np, origin_np + 0.01 * axis_np]),
                np.array([[0, 1]])
            )
            origin_viz.set_radius(0.015)
            origin_viz.set_color((1., 0., 0.))  # Red for origin
            origin_viz.add_to_group(obj_name)

        elif joint_type == "prismatic":
            # Extract parameters
            axis_np = np.array(joint_params.get("axis", [1., 0., 0.]))
            origin_np = np.array(joint_params.get("origin", [0., 0., 0.]))

            # Normalize axis if needed
            axis_norm = np.linalg.norm(axis_np)
            if axis_norm > 1e-6:
                axis_np = axis_np / axis_norm

            # Visualize axis
            seg_nodes = np.array([origin_np, origin_np + axis_np * scale])
            seg_edges = np.array([[0, 1]])

            axis_viz = ps.register_curve_network(f"{obj_name}_joint_axis", seg_nodes, seg_edges)
            axis_viz.set_radius(0.01)
            axis_viz.set_color((0., 1., 1.))  # Cyan for prismatic joint
            axis_viz.add_to_group(obj_name)

        elif joint_type == "planar":
            # Extract parameters
            normal_np = np.array(joint_params.get("normal", [0., 0., 1.]))

            # Visualize normal vector
            seg_nodes = np.array([[0, 0, 0], normal_np * scale])
            seg_edges = np.array([[0, 1]])

            normal_viz = ps.register_curve_network(f"{obj_name}_joint_normal", seg_nodes, seg_edges)
            normal_viz.set_color((1.0, 0.0, 0.0))  # Red for planar normal
            normal_viz.set_radius(0.01)
            normal_viz.add_to_group(obj_name)

        elif joint_type == "ball":
            # Extract parameters
            center_np = np.array(joint_params.get("center", [0., 0., 0.]))

            # Visualize center point
            center_viz = ps.register_point_cloud(f"{obj_name}_joint_center", center_np.reshape(1, 3))
            center_viz.set_radius(0.02)
            center_viz.set_color((1., 0., 1.))  # Magenta for ball joint
            center_viz.add_to_group(obj_name)

        elif joint_type == "screw":
            # Extract parameters
            axis_np = np.array(joint_params.get("axis", [0., 1., 0.]))
            origin_np = np.array(joint_params.get("origin", [0., 0., 0.]))
            pitch = joint_params.get("pitch", 0.0)

            # Normalize axis if needed
            axis_norm = np.linalg.norm(axis_np)
            if axis_norm > 1e-6:
                axis_np = axis_np / axis_norm

            # Visualize axis
            seg_nodes = np.array([origin_np - axis_np * scale, origin_np + axis_np * scale])
            seg_edges = np.array([[0, 1]])

            axis_viz = ps.register_curve_network(f"{obj_name}_joint_axis", seg_nodes, seg_edges)
            axis_viz.set_radius(0.01)
            axis_viz.set_color((0., 0., 1.0))  # Blue for screw joint
            axis_viz.add_to_group(obj_name)

            # Visualize pitch using an arrow
            if abs(pitch) > 1e-6:
                pitch_arrow_start = origin_np + axis_np * scale * 0.6

                # Find a perpendicular vector
                perp_vec = np.array([1, 0, 0])
                if np.abs(np.dot(axis_np, perp_vec)) > 0.9:
                    perp_vec = np.array([0, 1, 0])

                perp_vec = perp_vec - np.dot(perp_vec, axis_np) * axis_np
                perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-9)

                pitch_arrow_end = pitch_arrow_start + 0.2 * pitch * perp_vec

                pitch_viz = ps.register_curve_network(
                    f"{obj_name}_joint_pitch",
                    np.array([pitch_arrow_start, pitch_arrow_end]),
                    np.array([[0, 1]])
                )
                pitch_viz.set_color((1., 0., 0.))  # Red for pitch arrow
                pitch_viz.set_radius(0.008)
                pitch_viz.add_to_group(obj_name)

    def visualize_ground_truth(self, obj_name):
        """Visualize ground truth joint information for a specific object"""
        if not self.ground_truth_data:
            return

        # Map object name to ground truth
        namespace = f"{self.p.namespace}_{obj_name}"  # Construct full namespace
        gt_key = self.map_namespace_to_ground_truth(namespace)

        if not gt_key:
            return

        scene = gt_key["scene"]
        object_type = gt_key["object"]
        part_info = gt_key["part"]

        if (scene not in self.ground_truth_data or
                object_type not in self.ground_truth_data[scene] or
                part_info not in self.ground_truth_data[scene][object_type]):
            return

        joint_info = self.ground_truth_data[scene][object_type][part_info]

        # Extract axis and pivot
        if "axis" not in joint_info or "pivot" not in joint_info:
            return

        axis = np.array(joint_info["axis"])

        # Handle different pivot formats
        pivot = joint_info["pivot"]
        if isinstance(pivot, list):
            if len(pivot) == 1:
                pivot_point = np.array([0., 0., 0.]) + float(pivot[0]) * axis
            elif len(pivot) == 3:
                pivot_point = np.array(pivot)
            else:
                return
        else:
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

        name = f"GT_{obj_name}_Axis"
        axis_viz = ps.register_curve_network(name, seg_nodes, seg_edges)
        axis_viz.set_radius(0.015)
        axis_viz.set_color((0.0, 0.8, 0.2))  # Green for ground truth
        axis_viz.add_to_group(obj_name)
        self.gt_curve_networks.append(name)

    def remove_ground_truth_visualization(self):
        """Remove all ground truth visualization elements"""
        for name in self.gt_curve_networks:
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)

        for name in self.gt_point_clouds:
            if ps.has_point_cloud(name):
                ps.remove_point_cloud(name)

        self.gt_curve_networks = []
        self.gt_point_clouds = []

    # Original methods from VizKVILData class (keeping essential ones)
    def _load_traj(self):
        """load xyz data, filtered one"""
        self.delta_time = torch.load(self.p.t_obj / "delta_time.pt", weights_only=False)
        self.xyz_traj_filtered = torch.load(get_validate_file(self.p.t_obj_xyz), weights_only=False).data
        self.xyz_traj_filtered_raw = copy.deepcopy(self.xyz_traj_filtered)
        self.frame_traj = DictNumpyArray.load_pt(self.p.t_local_pose).data
        self.frame_traj_raw = copy.deepcopy(self.frame_traj)

    def get_frame_traj(self):
        return self.frame_traj_raw

    def get_traj(self):
        return self.xyz_traj_filtered_raw

    def _load_images(self):
        """load rgb, depth and mask images"""
        self.rgb_overlays, _ = load_rgb_batch(self.p.v_all / "overlay", [".jpg"], bgr2rgb=True)
        self.rgb_raw, _ = load_rgb_batch(self.p.t_rgb, [".jpg"], bgr2rgb=True)
        if len(self.rgb_overlays) == 0:
            self.rgb_overlays, _ = load_rgb_batch(self.p.t_rgb, [".jpg"], bgr2rgb=True)

        self.rgb_enabled: bool = False
        self.rgb_quantity = None

        self.depths, _ = load_depth_batch(
            self.p.t_depth, [".png"], ex_pattern=["raw_", "v_"], to_meter=True)

        self.masks = {}
        mask_path = self.p.t_mask
        if self.p.t_mask_instance_idx.is_file():
            mask_instances = load_dict_from_yaml(self.p.t_mask_instance_idx)
        else:
            mask_instances = {}

        for m_dir in get_ordered_subdirs(mask_path):
            if m_dir.stem == "all":
                continue

            obj_name = m_dir.stem
            if not mask_instances:
                self.masks[m_dir.stem], _ = load_mask_batch(m_dir, pattern=[".png"])
                continue

            if obj_name not in mask_instances:
                continue

            if mask_instances:
                m_dir = m_dir / f"{mask_instances[obj_name]:>02d}"
                self.masks[obj_name], _ = load_mask_batch(m_dir, pattern=[".png"])

    def _chose_objects(self, auto_viz_all: bool):
        """select which object to visualize"""
        if not auto_viz_all:
            obj_list = ask_checkbox_with_all("Select trajectory:", list(self.xyz_traj_filtered.keys()), default=["all"])
            if len(obj_list) < 1:
                console.print(f"[red]no trajectories found/selected in your trajectory")
        else:
            obj_list = self.xyz_traj_filtered.keys()
        self.obj_list = obj_list
        self.n_obj = len(self.obj_list)
        console.log(f"objects in scene: {self.obj_list}")

    def _load_intrinsics(self):
        """loading camera intrinsics"""
        meta_info = DemoMetaInfo.load(self.p.t_meta)
        self.intrinsic_np = meta_info.get_intrinsics()

    def _load_obj_raw_pcl(self):
        """load depth image, compute raw masked PCL and viz in polyscope"""
        self.pcd_raw = {o: [] for o in self.obj_list}
        self.ps_pcl_obj = {}

        from robot_utils.py.parallel import get_multithread_pool
        from functools import partial

        def process(depth, mask, intrinsics):
            mask = erode_mask(mask, 4)
            wh = np.fliplr(np.argwhere(mask))
            return proj_img_to_cam(wh, depth, intrinsics)

        pool = get_multithread_pool()
        self.ps_group["person"] = ps.create_group("person")
        self.ps_group_flag["person"] = True
        for obj in self.obj_list + ["person", ]:
            if obj not in self.masks.keys():
                continue

            self.pcd_raw[obj] = pool.map(partial(process, intrinsics=self.intrinsic_np), self.depths, self.masks[obj])
            self.update_raw_pcl(obj, 0)

    def _reset_track_data(self):
        """load raw tracker data and data before filtering"""
        self.xyz_track = torch.load(self.p.t_flow_xyz, weights_only=False)
        self.uv_track = torch.load(self.p.t_flow_uv, weights_only=False)
        self.xyz_filtered = torch.load(self.p.t_obj_xyz, weights_only=False).data

        self.repair_result_proc: Dict[str, List[RepairedObjData]] = torch.load(
            self.p.t_template_r, weights_only=False)

        self.xyz_track_enabled: bool = False
        self.xyz_filtered_enabled: bool = False
        self.track_obj = self.obj_list[0]
        self.track_obj_changed: bool = False
        self.track_obj_pcl_init = None
        self.xyz_track_ps: Dict[str, Optional[ps.PointCloud]] = {o: None for o in self.obj_list}
        self.xyz_filtered_ps = {o: None for o in self.obj_list}
        self.obj_pcl_previous = {o: np.empty(0) for o in self.obj_list}
        self.shape_opt, self.shape_to_be_opt = None, None
        self.shape_init, self.plot_on_init = {}, None
        self.flag_show_on_init_frame: bool = True
        self.flag_show_profile: bool = True
        self.flag_show_occ: bool = True
        self.flag_show_out_of_mask: bool = True
        self.flag_viz_proc: bool = False
        self.flag_viz_online: bool = False

        self.track_obj_status_text = ""
        self.offset_radio_changed, self.offset_radio = False, 0.4

        from obj_can_space.template.obj_template import ObjTemplate
        self.obj_template = {
            obj: ObjTemplate(
                obj,
                force_redo=True,
                pcl_template=self.xyz_track[obj][0]
            )
            for obj in self.xyz_track.keys()
        }

    def _load_can_space_cmap(self):
        """load colors of each objects corresponding to the DCN canonical space"""
        self.uv_color = {}
        self.uv_color_cv = {}
        for obj in self.obj_list:
            if "hand_person" in obj:
                continue
            uv_colors = torch.load(self.p.can_dir / obj / "uv_color_plt.pt", weights_only=False)
            uv_color_cv = torch.load(self.p.can_dir / obj / "uv_color_cv.pt", weights_only=False)
            self.uv_color[obj] = uv_colors
            self.uv_color_cv[obj] = uv_color_cv

    def _load_can_neighbors(self):
        self.neighbor_idx = {}
        for obj in self.obj_list:
            if "hand_person" in obj:
                continue
            self.neighbor_idx[obj] = torch.load(self.p.can_dir / obj / "neighbor_idx.pt", weights_only=False)

    def _load_global_frame(self):
        from obj_pose.local_pose_estimation import FrameInfo
        frame_info = FrameInfo.load(self.p.t_frame_cfg)

        self.imitator_frame = frame_info.imitator
        self.imitator_frame_raw = frame_info.imitator.copy()
        self.person_frame = frame_info.get_person_frame(0)
        if self.person_frame is not None:
            self.person_frame_raw = self.person_frame.copy()

        self.z_up_dir = -frame_info.imitator[:, 1] - frame_info.imitator[:, -1]
        self.z_up_dir = self.z_up_dir / np.linalg.norm(self.z_up_dir)
        self.center = frame_info.imitator[:, -1]
        x_dir = np.array([1, 0, 0])
        self.radial_dir = frame_info.radial_dir
        for o, r_dir in self.radial_dir.items():
            r_dir[...] = r_dir.dot(x_dir) * x_dir

    @staticmethod
    def _trans_frame34(frame1, frame2):
        """frame 2 is represented in frame 1"""
        frame2 = frame2.copy()
        if frame2.ndim == 2:
            frame2 = frame2[None]

        frame2[:, :, :3] = np.einsum("ji,bnj->bin", frame1[:, :3], frame2[:, :, :3])
        frame2[:, :, -1] = np.einsum("ji,bj->bi", frame1[:, :3], frame2[:, :, -1] - frame1[:, -1])
        return frame2.squeeze(0)

    def _transform_global_frame(self, frame: np.ndarray):
        """Args: frame: (3, 4)"""
        if not self.flag_local_frame:
            self.person_frame = self.person_frame_raw.copy()
            self.imitator_frame = self.imitator_frame_raw.copy()
            return

        self.person_frame = self._trans_frame34(frame, self.person_frame_raw)
        self.imitator_frame = self._trans_frame34(frame, self.imitator_frame_raw)

    def _update_global_frame(self):
        draw_frame_3d(self.imitator_frame, scale=0.2, radius=0.005, alpha=0.8, enabled=True, label="frame_imitator")
        if self.person_frame is not None:
            draw_frame_3d(
                self.person_frame, scale=0.2, radius=0.005, alpha=0.8, enabled=True,
                label="frame_person_00")

    def update_raw_pcl(self, obj: str, timestep: int):
        if obj not in self.pcd_raw:
            return
        self.ps_pcl_obj[obj] = register_point_cloud(
            f"pcl_{obj}_raw", self.pcd_raw[obj][timestep], enabled=self.ps_group_flag.get(obj, True), radius=0.0005
        )
        self.ps_pcl_obj[obj].add_to_group(obj)

    def update_filtered_traj(self, obj_name: str, timestep: int):
        traj = self.xyz_traj_filtered[obj_name]
        if timestep == 0:
            enabled = False if "hand_person" in obj_name else True
            self.current_pcl_points[obj_name] = register_point_cloud(
                f"xyz_filtered_{obj_name}", traj[0], enabled=enabled, radius=0.007)
            self.current_pcl_points[obj_name].add_to_group(obj_name)

            if obj_name in self.uv_color.keys():
                self.current_pcl_points[obj_name].add_color_quantity(
                    "dcn_cmap", values=self.uv_color[obj_name], enabled=True)
            if "hand_person" in str(obj_name):
                if self.frame_mat_list.get(obj_name, None) is None:
                    new_frame_mat = self.frame_traj[obj_name][0, :, 0]
                    self.frame_mat_list[obj_name] = new_frame_mat
                self.hand_curve_network[obj_name] = register_curve_network(
                    f"{obj_name}_curve", nodes=traj[0], edges=hand_edges, enabled=True, radius=0.005)
                self.hand_curve_network[obj_name].add_to_group(obj_name)

        else:
            self.current_pcl_points[obj_name].update_point_positions(traj[timestep])
            if "hand_person" in str(obj_name):
                self.hand_curve_network[obj_name].update_node_positions(traj[timestep])

    def _update_obj_frame(self):
        if not (self.pt_idx_changed or self.time_changed or self.flag_local_frame_changed):
            return

        frame = self.frame_traj_raw[self.master_obj][0, self.current_time, self.pt_idx]
        self._transform_global_frame(frame)
        self._update_global_frame()
        self._transform_traj(frame)

    def _transform_traj(self, frame: np.ndarray):
        if not self.flag_local_frame:
            self.frame_traj = copy.deepcopy(self.frame_traj_raw)
            self.xyz_traj_filtered = copy.deepcopy(self.xyz_traj_filtered_raw)
            return

        t = self.current_time
        for obj in self.obj_list:
            traj_viz = self.xyz_traj_filtered_raw[obj][t]
            self.xyz_traj_filtered[obj][t] = np.einsum("ji,pj->pi", frame[:, :3], traj_viz - frame[:, -1])

    def _viz_track_results(self):
        """Visualize the tracking results and the repaired pcd during preprocessing"""
        if "hand_person" in self.track_obj:
            return

        tracker_raw_changed, self.xyz_track_enabled = psim.Checkbox("Show point tracker", self.xyz_track_enabled)
        tracker_repaired_changed, self.xyz_filtered_enabled = psim.Checkbox("Repair point tracker",
                                                                            self.xyz_filtered_enabled)
        self.xyz_track_enabled = self.xyz_track_enabled & self.ps_group_flag[self.track_obj]
        self.xyz_filtered_enabled = self.xyz_filtered_enabled & self.ps_group_flag[self.track_obj]
        if tracker_raw_changed or self.time_changed:
            self.xyz_track_ps[self.track_obj] = register_point_cloud(
                f"xyz_track_{self.track_obj}", self.xyz_track[self.track_obj][self.current_time],
                enabled=self.xyz_track_enabled, radius=0.005
            )
        if tracker_repaired_changed or self.time_changed:
            self.xyz_filtered_ps[self.track_obj] = register_point_cloud(
                f"track_3d_repair_{self.track_obj}", self.xyz_filtered[self.track_obj][self.current_time],
                enabled=self.xyz_filtered_enabled, radius=0.005
            )

    def run(self):
        ps.set_user_callback(self._callback)
        ps.show()

    def _callback(self):
        psim.PushItemWidth(300)
        psim.TextUnformatted("Enhanced KVIL GUI with Joint Analysis")
        psim.Separator()

        # Time control
        self.time_changed, self.current_time = psim.SliderInt(
            "time", v=self.current_time, v_min=0, v_max=self.t_max - 1)

        # Keyboard controls
        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_LeftArrow)):
            self.time_changed = True
            self.current_time = max(self.current_time - 1, 0)
        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_RightArrow)):
            self.time_changed = True
            self.current_time = min(self.current_time + 1, self.t_max - 1)
        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_R)):
            self.time_changed = True
            self.current_time = 0
        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Escape)):
            exit()

        # Offset ratio
        self.offset_radio_changed, self.offset_radio = psim.SliderFloat(
            "offset_ratio", v=self.offset_radio, v_min=-1, v_max=2.0)

        # Object selection
        psim.Separator()
        self.track_obj_changed, self.track_obj = ps_select(self.track_obj, self.obj_list)

        if self.track_obj_changed:
            self.time_changed = True
            self.current_time = 0

        self.master_obj_changed, self.master_obj = ps_select(
            self.master_obj, self.obj_list, label="Select Master Object")
        if self.master_obj_changed:
            self.time_changed = True
            self.current_time = 0
            if "hand_person" in self.master_obj:
                self.pt_idx_max = 1
            else:
                self.pt_idx_max = self.xyz_traj_filtered[self.master_obj][self.current_time].shape[0]
            self.pt_idx = min(self.pt_idx, self.pt_idx_max - 1)

        self.pt_idx_changed, self.pt_idx = psim.SliderInt(
            "idx", v=self.pt_idx, v_min=0, v_max=self.pt_idx_max - 1)

        self.flag_local_frame_changed, self.flag_local_frame = psim.Checkbox("Enable Local Frame",
                                                                             self.flag_local_frame)

        if self.master_obj_changed:
            self.pt_idx_changed = True

        self._update_obj_frame()
        self._viz_track_results()

        # Joint Analysis Section
        psim.Separator()
        psim.TextUnformatted("Joint Analysis")

        # Joint visualization controls for each object
        if psim.TreeNode("Joint Visualization Controls"):
            for obj_name in self.obj_list:
                if "hand_person" in obj_name or obj_name not in self.best_joint:
                    continue

                joint_type = self.best_joint.get(obj_name, "unknown")

                # Show joint type and controls
                psim.Text(f"{obj_name}: {joint_type}")
                psim.SameLine()

                changed, self.show_joint_viz[obj_name] = psim.Checkbox(
                    f"Show###{obj_name}", self.show_joint_viz.get(obj_name, False))

                if changed:
                    if self.show_joint_viz[obj_name]:
                        self.visualize_joint_parameters(obj_name)
                    else:
                        self.remove_joint_visualization(obj_name)

            # Joint visualization scale
            scale_changed, self.joint_viz_scale = psim.SliderFloat(
                "Joint Viz Scale", self.joint_viz_scale, 0.05, 1.0)
            if scale_changed:
                # Update all active joint visualizations
                for obj_name in self.obj_list:
                    if self.show_joint_viz.get(obj_name, False):
                        self.visualize_joint_parameters(obj_name)

            psim.TreePop()

        # Ground truth visualization
        if self.ground_truth_data:
            gt_changed, self.show_ground_truth = psim.Checkbox("Show Ground Truth", self.show_ground_truth)
            if gt_changed:
                if self.show_ground_truth:
                    for obj_name in self.obj_list:
                        if "hand_person" not in obj_name:
                            self.visualize_ground_truth(obj_name)
                else:
                    self.remove_ground_truth_visualization()

            if self.show_ground_truth:
                gt_scale_changed, self.ground_truth_scale = psim.SliderFloat(
                    "GT Scale", self.ground_truth_scale, 0.1, 2.0)
                if gt_scale_changed:
                    self.remove_ground_truth_visualization()
                    for obj_name in self.obj_list:
                        if "hand_person" not in obj_name:
                            self.visualize_ground_truth(obj_name)

        # Display joint information for current track object
        if psim.TreeNode("Joint Information"):
            obj_name = self.track_obj
            if obj_name in self.best_joint:
                joint_type = self.best_joint[obj_name]
                psim.Text(f"Object: {obj_name}")
                psim.Text(f"Joint Type: {joint_type}")

                if obj_name in self.joint_info and "joint_probs" in self.joint_info[obj_name]:
                    psim.Text("Joint Probabilities:")
                    joint_probs = self.joint_info[obj_name]["joint_probs"]
                    for jtype, prob in joint_probs.items():
                        psim.Text(f"  {jtype}: {prob:.3f}")

                if joint_type in self.joint_params.get(obj_name, {}):
                    params = self.joint_params[obj_name][joint_type]
                    psim.Text(f"{joint_type.capitalize()} Parameters:")

                    if joint_type == "revolute":
                        axis = params.get("axis", [0, 0, 0])
                        origin = params.get("origin", [0, 0, 0])
                        motion_limit = params.get("motion_limit", (0, 0))
                        psim.Text(f"  Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
                        psim.Text(f"  Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
                        psim.Text(f"  Range: {np.degrees(motion_limit[1] - motion_limit[0]):.1f}°")

                    elif joint_type == "prismatic":
                        axis = params.get("axis", [0, 0, 0])
                        origin = params.get("origin", [0, 0, 0])
                        motion_limit = params.get("motion_limit", (0, 0))
                        psim.Text(f"  Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
                        psim.Text(f"  Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
                        psim.Text(f"  Range: {motion_limit[1] - motion_limit[0]:.3f}m")

                    elif joint_type == "planar":
                        normal = params.get("normal", [0, 0, 0])
                        motion_limit = params.get("motion_limit", (0, 0))
                        psim.Text(f"  Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
                        psim.Text(f"  Range: {motion_limit[0]:.3f}m")

                    elif joint_type == "ball":
                        center = params.get("center", [0, 0, 0])
                        radius = params.get("radius", 0)
                        motion_limit = params.get("motion_limit", (0, 0, 0))
                        psim.Text(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
                        psim.Text(f"  Radius: {radius:.3f}m")
                        psim.Text(f"  Range: {np.degrees(motion_limit[2]):.1f}°")

                    elif joint_type == "screw":
                        axis = params.get("axis", [0, 0, 0])
                        origin = params.get("origin", [0, 0, 0])
                        pitch = params.get("pitch", 0)
                        motion_limit = params.get("motion_limit", (0, 0))
                        psim.Text(f"  Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
                        psim.Text(f"  Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
                        psim.Text(f"  Pitch: {pitch:.3f}")
                        psim.Text(f"  Range: {np.degrees(motion_limit[1] - motion_limit[0]):.1f}°")

            psim.TreePop()

        # Joint analysis parameters (only show if using advanced analyzer)
        if self.use_advanced_analyzer and psim.TreeNode("Advanced Joint Analysis Parameters"):
            changed_neighbors, self.num_neighbors = psim.SliderInt("Neighbors", self.num_neighbors, 5, 100)
            changed_col_sigma, self.col_sigma = psim.SliderFloat("Col Sigma", self.col_sigma, 0.05, 0.5)
            changed_col_order, self.col_order = psim.SliderFloat("Col Order", self.col_order, 1.0, 10.0)
            changed_cop_sigma, self.cop_sigma = psim.SliderFloat("Cop Sigma", self.cop_sigma, 0.05, 0.5)
            changed_cop_order, self.cop_order = psim.SliderFloat("Cop Order", self.cop_order, 1.0, 10.0)
            changed_rad_sigma, self.rad_sigma = psim.SliderFloat("Rad Sigma", self.rad_sigma, 0.05, 0.5)
            changed_rad_order, self.rad_order = psim.SliderFloat("Rad Order", self.rad_order, 1.0, 10.0)
            changed_zp_sigma, self.zp_sigma = psim.SliderFloat("ZP Sigma", self.zp_sigma, 0.05, 0.5)
            changed_zp_order, self.zp_order = psim.SliderFloat("ZP Order", self.zp_order, 1.0, 10.0)

            changed_savgol_window, self.savgol_window = psim.SliderInt("SG Window", self.savgol_window, 3, 31)
            changed_savgol_poly, self.savgol_poly = psim.SliderInt("SG Poly Order", self.savgol_poly, 1, 5)
            changed_savgol, self.use_savgol = psim.Checkbox("Use SG Filter", self.use_savgol)

            params_changed = (changed_neighbors or changed_col_sigma or changed_col_order or
                              changed_cop_sigma or changed_cop_order or changed_rad_sigma or
                              changed_rad_order or changed_zp_sigma or changed_zp_order or
                              changed_savgol_window or changed_savgol_poly or changed_savgol)

            if params_changed and psim.Button("Apply Parameter Changes"):
                console.log("Reanalyzing joints with new parameters...")
                self._perform_joint_analysis_all_objects()
                # Update visualizations
                for obj_name in self.obj_list:
                    if self.show_joint_viz.get(obj_name, False):
                        self.visualize_joint_parameters(obj_name)

            psim.TreePop()

        # Show which analyzer is being used
        analyzer_type = "Advanced" if self.use_advanced_analyzer else "Simple"
        psim.Text(f"Using {analyzer_type} Joint Analyzer")

        # Reanalyze button
        if psim.Button("Reanalyze All Joints"):
            console.log("Reanalyzing all joints...")
            self._perform_joint_analysis_all_objects()
            # Update visualizations
            for obj_name in self.obj_list:
                if self.show_joint_viz.get(obj_name, False):
                    self.visualize_joint_parameters(obj_name)

        # Original controls
        psim.Separator()
        show_on_init_frame_changed, self.flag_show_on_init_frame = psim.Checkbox(
            "Show on init frame", self.flag_show_on_init_frame)
        psim.Indent(50)
        _, self.flag_show_profile = psim.Checkbox("Profile", self.flag_show_profile)
        psim.SameLine()
        _, self.flag_show_occ = psim.Checkbox("Occ", self.flag_show_occ)
        psim.SameLine()
        _, self.flag_show_out_of_mask = psim.Checkbox("Mask", self.flag_show_out_of_mask)
        psim.Unindent(50)

        psim.Separator()
        psim.TextUnformatted("Viz object anomaly detection results")
        psim.Indent(50)
        _, self.flag_viz_proc = psim.Checkbox("Viz Proc Result", self.flag_viz_proc)
        psim.SameLine()
        _, self.flag_viz_online = psim.Checkbox("Viz Online Result", self.flag_viz_online)
        psim.Unindent(50)

        psim.Separator()
        i = 0
        n_obj_per_line = 3
        for o in self.ps_group_flag.keys():
            changed_, self.ps_group_flag[o] = psim.Checkbox(f"g_{o[:3]}", self.ps_group_flag.get(o, True))
            if changed_:
                self.ps_group[o].set_enabled(self.ps_group_flag[o])
            if i % 3 != n_obj_per_line - 1 and i != len(self.obj_list) - 1:
                psim.SameLine()
            i = i + 1
        psim.Separator()

        # Update visualizations
        if self.time_changed:
            if self.shared_data is not None:
                with self.shared_data["lock"]:
                    self.shared_data["time"] = self.current_time

            for obj in self.obj_list + ["person"]:
                if "hand_person" in obj:
                    continue
                self.update_raw_pcl(obj, self.current_time)

            for obj in self.obj_list:
                self.update_filtered_traj(obj, self.current_time)

            for obj, frame_mat in self.frame_mat_list.items():
                frame = draw_frame_3d(
                    frame_mat[self.current_time], label=f"frame_{obj}", scale=0.1, radius=0.01,
                    alpha=1.0, collections=None, enabled=True
                )
                frame.add_to_group(obj)

        self.pu.cam.default_monitor()


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path", "-p", type=Path, help="the absolute path to task")
@click.option("--namespace", "-n", type=str, help="the namespace")
def main(path, namespace):
    EnhancedVizKVILData(path, namespace).run()


if __name__ == "__main__":
    main()