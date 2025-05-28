import numpy as np
import os
import json
import re
from scipy.signal import savgol_filter
from datetime import datetime

# Import joint analysis project modules
from joint_analysis.joint_analysis.core.joint_estimation import compute_joint_info_all_types


class JointAnalysisEvaluator:
    def __init__(self, file_paths=None):
        # Initialize default file paths list if none provided
        if file_paths is None:
            self.file_paths = ["./demo_data/s1_gasstove_part2_1110_1170.npy"]
        else:
            self.file_paths = file_paths

        # Load ground truth joint data
        self.ground_truth_json = "/common/homes/all/uksqc_chen/projects/control/ParaHome/all_scenes_transformed_axis_pivot_data.json"
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
        self.col_sigma = 0.2
        self.col_order = 4.0
        self.cop_sigma = 0.2
        self.cop_order = 4.0
        self.rad_sigma = 0.2
        self.rad_order = 4.0
        self.zp_sigma = 0.2
        self.zp_order = 4.0
        self.use_savgol = True
        self.savgol_window = 21
        self.savgol_poly = 2

        # Store evaluation results
        self.evaluation_results = {}

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

    def extract_scene_info_from_filename(self, filename):
        """Extract scene and object information from filename"""
        # Pattern to match filenames like "s204_drawer_part2_1320_1440"
        pattern = r'(s\d+)_([^_]+)_(part\d+)_\d+_\d+'
        match = re.match(pattern, filename)

        if match:
            scene, object_type, part = match.groups()
            return {
                "scene": scene,
                "object": object_type,
                "part": part
            }

        # Alternative pattern for base parts
        pattern2 = r'(s\d+)_([^_]+)_(base)_\d+_\d+'
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
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-6 or v2_norm < 1e-6:
            return float('inf')

        v1_normalized = v1 / v1_norm
        v2_normalized = v2 / v2_norm

        # Calculate angle (use absolute value of dot product to get minimum angle)
        dot_product = np.clip(np.abs(np.dot(v1_normalized, v2_normalized)), 0.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        # Return the smaller angle (0-90 degrees)
        return min(angle_deg, 180 - angle_deg)

    def calculate_distance_between_lines(self, axis1, origin1, axis2, origin2):
        """Calculate minimum distance between two lines in 3D space"""
        # Normalize axes
        axis1_norm = np.linalg.norm(axis1)
        axis2_norm = np.linalg.norm(axis2)

        if axis1_norm < 1e-6 or axis2_norm < 1e-6:
            return float('inf')

        axis1 = axis1 / axis1_norm
        axis2 = axis2 / axis2_norm

        # Calculate cross product
        cross_product = np.cross(axis1, axis2)
        cross_norm = np.linalg.norm(cross_product)

        if cross_norm < 1e-6:
            # Lines are parallel, calculate point-to-line distance
            vec_to_point = origin1 - origin2
            proj_on_axis = np.dot(vec_to_point, axis2) * axis2
            perpendicular = vec_to_point - proj_on_axis
            return np.linalg.norm(perpendicular)
        else:
            # Lines are skew, calculate minimum distance
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

        # Check if ground truth exists
        if scene not in self.ground_truth_data:
            return None
        if object_type not in self.ground_truth_data[scene]:
            return None
        if part not in self.ground_truth_data[scene][object_type]:
            return None

        return self.ground_truth_data[scene][object_type][part]

    def evaluate_single_file(self, file_path):
        """Evaluate a single file and return results"""
        print(f"\nEvaluating: {file_path}")

        # Extract filename
        filename = os.path.basename(file_path).split('.')[0]
        scene_info = self.extract_scene_info_from_filename(filename)

        if not scene_info:
            return {
                "filename": filename,
                "error": "Could not extract scene info from filename"
            }

        object_type = scene_info["object"]

        # Check if we have expected joint type for this object
        if object_type not in self.expected_joint_types:
            return {
                "filename": filename,
                "scene_info": scene_info,
                "error": f"Unknown object type: {object_type}"
            }

        expected_joint_type = self.expected_joint_types[object_type]

        # Load and process data
        try:
            data = np.load(file_path)
            print(f"Loaded data shape: {data.shape}")
        except Exception as e:
            return {
                "filename": filename,
                "scene_info": scene_info,
                "error": f"Could not load file: {e}"
            }

        # Apply Savitzky-Golay filter to smooth data
        data_filter = savgol_filter(
            x=data, window_length=21, polyorder=2, deriv=0, axis=0, delta=0.1
        )

        # Perform joint analysis
        try:
            joint_params, best_joint, info_dict = self.perform_joint_analysis(data_filter)
        except Exception as e:
            return {
                "filename": filename,
                "scene_info": scene_info,
                "error": f"Joint analysis failed: {e}"
            }

        print(f"Expected: {expected_joint_type}, Detected: {best_joint}")

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

        # If classification is incorrect, don't calculate errors
        if not result["classification_correct"]:
            result["error_message"] = f"Classification incorrect: expected {expected_joint_type}, got {best_joint}"
            print(f"Classification incorrect: expected {expected_joint_type}, got {best_joint}")
            return result

        # Get ground truth
        ground_truth = self.get_ground_truth_for_object(scene_info)
        if not ground_truth:
            result["error_message"] = "No ground truth data available"
            print("No ground truth data available")
            return result

        result["ground_truth"] = ground_truth

        # Calculate errors based on joint type
        if best_joint == "prismatic":
            self.calculate_prismatic_errors(result, joint_params[best_joint], ground_truth)
        elif best_joint == "revolute":
            self.calculate_revolute_errors(result, joint_params[best_joint], ground_truth)
        elif best_joint == "planar":
            self.calculate_planar_errors(result, joint_params[best_joint], ground_truth)

        return result

    def calculate_prismatic_errors(self, result, estimated_params, ground_truth):
        """Calculate errors for prismatic joints"""
        est_axis = np.array(estimated_params.get("axis", [0, 0, 0]))
        gt_axis = np.array(ground_truth["axis"])

        # Calculate angle error
        angle_error = self.calculate_angle_between_vectors(est_axis, gt_axis)

        result["errors"] = {
            "axis_angle_error_degrees": angle_error
        }

        print(f"Prismatic joint - Axis angle error: {angle_error:.2f}°")

    def calculate_revolute_errors(self, result, estimated_params, ground_truth):
        """Calculate errors for revolute joints"""
        est_axis = np.array(estimated_params.get("axis", [0, 0, 0]))
        est_origin = np.array(estimated_params.get("origin", [0, 0, 0]))
        gt_axis = np.array(ground_truth["axis"])

        # Handle ground truth pivot
        gt_pivot = ground_truth["pivot"]
        if isinstance(gt_pivot, list):
            if len(gt_pivot) == 1:
                # Single value pivot - assume it's along the axis from origin
                gt_origin = np.array([0., 0., 0.]) + float(gt_pivot[0]) * gt_axis
            elif len(gt_pivot) == 3:
                gt_origin = np.array(gt_pivot)
            else:
                gt_origin = np.array([0., 0., 0.])
        else:
            # Numeric pivot value
            gt_origin = np.array([0., 0., 0.]) + float(gt_pivot) * gt_axis

        # Calculate angle error
        angle_error = self.calculate_angle_between_vectors(est_axis, gt_axis)

        # Calculate distance between axes
        axis_distance = self.calculate_distance_between_lines(est_axis, est_origin, gt_axis, gt_origin)

        # Calculate origin distance
        origin_distance = np.linalg.norm(est_origin - gt_origin)

        result["errors"] = {
            "axis_angle_error_degrees": angle_error,
            "axis_distance_meters": axis_distance,
            "origin_distance_meters": origin_distance
        }

        print(
            f"Revolute joint - Axis angle error: {angle_error:.2f}°, Axis distance: {axis_distance:.4f}m, Origin distance: {origin_distance:.4f}m")

    def calculate_planar_errors(self, result, estimated_params, ground_truth):
        """Calculate errors for planar joints"""
        est_normal = np.array(estimated_params.get("normal", [0, 0, 0]))
        gt_normal = np.array(ground_truth["axis"])  # For planar joints, axis represents normal

        # Calculate angle error
        angle_error = self.calculate_angle_between_vectors(est_normal, gt_normal)

        result["errors"] = {
            "normal_angle_error_degrees": angle_error
        }

        print(f"Planar joint - Normal angle error: {angle_error:.2f}°")

    def evaluate_all_files(self):
        """Evaluate all files and return comprehensive results"""
        print("Starting joint analysis evaluation...")
        print(f"Total files to evaluate: {len(self.file_paths)}")

        all_results = []
        summary_stats = {
            "total_files": len(self.file_paths),
            "successful_evaluations": 0,
            "classification_correct": 0,
            "classification_incorrect": 0,
            "errors_occurred": 0,
            "by_object_type": {},
            "by_joint_type": {}
        }

        for file_path in self.file_paths:
            result = self.evaluate_single_file(file_path)
            all_results.append(result)

            # Update summary statistics
            if "error" in result:
                summary_stats["errors_occurred"] += 1
            else:
                summary_stats["successful_evaluations"] += 1

                object_type = result["scene_info"]["object"]
                expected_type = result["expected_joint_type"]
                detected_type = result["detected_joint_type"]

                # Initialize object type stats if not exist
                if object_type not in summary_stats["by_object_type"]:
                    summary_stats["by_object_type"][object_type] = {
                        "total": 0,
                        "correct_classification": 0,
                        "incorrect_classification": 0
                    }

                # Initialize joint type stats if not exist
                for jtype in [expected_type, detected_type]:
                    if jtype not in summary_stats["by_joint_type"]:
                        summary_stats["by_joint_type"][jtype] = {
                            "expected": 0,
                            "detected": 0,
                            "correct_detections": 0
                        }

                # Update stats
                summary_stats["by_object_type"][object_type]["total"] += 1
                summary_stats["by_joint_type"][expected_type]["expected"] += 1
                summary_stats["by_joint_type"][detected_type]["detected"] += 1

                if result["classification_correct"]:
                    summary_stats["classification_correct"] += 1
                    summary_stats["by_object_type"][object_type]["correct_classification"] += 1
                    summary_stats["by_joint_type"][expected_type]["correct_detections"] += 1
                else:
                    summary_stats["classification_incorrect"] += 1
                    summary_stats["by_object_type"][object_type]["incorrect_classification"] += 1

        # Calculate accuracy percentages
        if summary_stats["successful_evaluations"] > 0:
            summary_stats["classification_accuracy"] = summary_stats["classification_correct"] / summary_stats[
                "successful_evaluations"] * 100
        else:
            summary_stats["classification_accuracy"] = 0.0

        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "summary_statistics": summary_stats,
            "detailed_results": all_results
        }

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
            # Convert numpy arrays to Python lists for JSON serialization
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
        stats = results["summary_statistics"]

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        print(f"Total files evaluated: {stats['total_files']}")
        print(f"Successful evaluations: {stats['successful_evaluations']}")
        print(f"Errors occurred: {stats['errors_occurred']}")
        print(f"Classification accuracy: {stats['classification_accuracy']:.1f}%")
        print(f"Correct classifications: {stats['classification_correct']}")
        print(f"Incorrect classifications: {stats['classification_incorrect']}")

        print("\nBy Object Type:")
        for obj_type, obj_stats in stats["by_object_type"].items():
            accuracy = (obj_stats["correct_classification"] / obj_stats["total"] * 100) if obj_stats["total"] > 0 else 0
            print(f"  {obj_type}: {obj_stats['correct_classification']}/{obj_stats['total']} correct ({accuracy:.1f}%)")

        print("\nBy Joint Type:")
        for joint_type, joint_stats in stats["by_joint_type"].items():
            accuracy = (joint_stats["correct_detections"] / joint_stats["expected"] * 100) if joint_stats[
                                                                                                  "expected"] > 0 else 0
            print(
                f"  {joint_type}: {joint_stats['correct_detections']}/{joint_stats['expected']} correct ({accuracy:.1f}%)")

        print("\nFiles with Correct Classification and Calculated Errors:")
        for result in results["detailed_results"]:
            if result.get("classification_correct", False) and "errors" in result:
                print(f"  {result['filename']} ({result['detected_joint_type']}):")
                for error_name, error_value in result["errors"].items():
                    if "angle" in error_name:
                        print(f"    {error_name}: {error_value:.2f}°")
                    else:
                        print(f"    {error_name}: {error_value:.4f}m")


def main():
    # Example file paths - replace with your actual file paths
    file_paths = [
        "/common/homes/all/uksqc_chen/projects/control/ParaHome/output_specific_actions/s204_drawer_part2_1320_1440.npy"
        # Add more file paths here, for example:
        # "/path/to/s1_microwave_part1_1380_1470.npy",
        # "/path/to/s2_refrigerator_part1_3180_3240.npy",
        # "/path/to/s3_washingmachine_part1_1140_1170.npy",
        # "/path/to/s4_trashbin_part1_2400_2490.npy",
        # "/path/to/s5_chair_base_2610_2760.npy"
    ]

    # You can also specify a directory to process all .npy files
    # import glob
    # directory_path = "/path/to/your/npy/files/"
    # file_paths = glob.glob(os.path.join(directory_path, "*.npy"))

    print(f"Found {len(file_paths)} files to evaluate")
    if len(file_paths) == 0:
        print("No files found. Please check your file paths.")
        return

    # Create evaluator
    evaluator = JointAnalysisEvaluator(file_paths)

    # Perform evaluation
    results = evaluator.evaluate_all_files()

    # Print summary
    evaluator.print_summary(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"joint_analysis_evaluation_{timestamp}.json"
    success = evaluator.save_results(results, output_file)

    if success:
        print(f"\n✓ Evaluation completed successfully!")
        print(f"✓ Results saved to: {output_file}")
        print(f"✓ Classification accuracy: {results['summary_statistics']['classification_accuracy']:.1f}%")
    else:
        print("\n✗ Failed to save results")


# Alternative main function for batch processing from a directory
def main_batch_directory(directory_path):
    """Process all .npy files in a directory"""
    import glob

    file_paths = glob.glob(os.path.join(directory_path, "*.npy"))

    if len(file_paths) == 0:
        print(f"No .npy files found in directory: {directory_path}")
        return

    print(f"Found {len(file_paths)} .npy files in {directory_path}")

    # Create evaluator
    evaluator = JointAnalysisEvaluator(file_paths)

    # Perform evaluation
    results = evaluator.evaluate_all_files()

    # Print summary
    evaluator.print_summary(results)

    # Save results with directory name in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.basename(directory_path.rstrip('/'))
    output_file = f"joint_analysis_evaluation_{dir_name}_{timestamp}.json"

    success = evaluator.save_results(results, output_file)

    if success:
        print(f"\n✓ Batch evaluation completed successfully!")
        print(f"✓ Results saved to: {output_file}")
        print(f"✓ Classification accuracy: {results['summary_statistics']['classification_accuracy']:.1f}%")


if __name__ == "__main__":
    # Use this for specific files
    main()

    # Or use this for batch processing a directory
    # main_batch_directory("/path/to/your/npy/files/directory/")