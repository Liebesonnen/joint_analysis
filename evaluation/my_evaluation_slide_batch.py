import numpy as np
import os
import json
import re
from scipy.signal import savgol_filter
from datetime import datetime
import sys
from collections import defaultdict
import glob

sys.path.append('/common/homes/all/uksqc_chen/projects/control')
# Import joint analysis project modules
from joint_analysis.core.joint_estimation import compute_joint_info_all_types

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

        # Joint analysis parameters
        self.num_neighbors = 50
        self.col_sigma = 0.6
        self.col_order = 4.0
        self.cop_sigma = 0.6
        self.cop_order = 4.0
        self.rad_sigma = 0.55
        self.rad_order = 4.0
        self.zp_sigma = 0.8
        self.zp_order = 4.0
        self.use_savgol = True
        self.savgol_window = 21
        self.savgol_poly = 2

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
        """Perform joint analysis on point history data"""
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

        # Perform joint analysis
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
            "classification_correct": best_joint == expected_joint_type,
            "joint_params": joint_params.get(best_joint, {}),
            "all_joint_probabilities": info_dict.get("joint_probs", {}) if info_dict else {}
        }

        # Calculate errors if classification is correct
        if result["classification_correct"]:
            ground_truth = self.get_ground_truth_for_object(scene_info)
            if ground_truth:
                result["ground_truth"] = ground_truth
                result["errors"] = self.calculate_joint_errors(best_joint, joint_params[best_joint], ground_truth)

        return result

    def evaluate_all_groups(self):
        """Evaluate all groups and return comprehensive results"""
        print("Starting improved joint analysis evaluation...")
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
        print("IMPROVED EVALUATION SUMMARY")
        print("=" * 80)

        overall_stats = results["overall_statistics"]
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
        print(f"Running evaluation with noise std = {noise_std}")
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

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        noise_str = f"noise_{noise_std:g}".replace('.', 'p')
        output_file = f"joint_analysis_evaluation_{noise_str}_{timestamp}.json"

        success = evaluator.save_results(results, output_file)

        if success:
            print(f"\n✓ Evaluation completed successfully!")
            print(f"✓ Results saved to: {output_file}")
        else:
            print(f"\n✗ Failed to save results")


if __name__ == "__main__":
    main()