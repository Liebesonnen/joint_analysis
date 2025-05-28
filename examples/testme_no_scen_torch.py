import numpy as np
import torch
import torch.nn.functional as F
import os
import json
import re
from scipy.signal import savgol_filter
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import glob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Import joint analysis project modules
from joint_analysis.joint_analysis.core.joint_estimation import compute_joint_info_all_types


class OptimizedJointAnalysisEvaluator:
    def __init__(self, file_paths=None, device=None, use_multiprocessing=True, num_workers=None):
        # Initialize default file paths list if none provided
        if file_paths is None:
            self.file_paths = ["./demo_data/s1_gasstove_part2_1110_1170.npy"]
        else:
            self.file_paths = file_paths

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Multiprocessing setup
        self.use_multiprocessing = use_multiprocessing
        if num_workers is None:
            self.num_workers = min(mp.cpu_count(), len(self.file_paths))
        else:
            self.num_workers = num_workers

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

        # Precompile torch functions for better performance
        self._compile_torch_functions()

    def _compile_torch_functions(self):
        """Precompile commonly used torch functions for better performance"""
        try:
            # For PyTorch 2.0+, use torch.compile
            if hasattr(torch, 'compile'):
                self.calculate_angles_batch_compiled = torch.compile(self._calculate_angles_batch_raw)
                self.calculate_distances_batch_compiled = torch.compile(self._calculate_distances_batch_raw)
            else:
                self.calculate_angles_batch_compiled = self._calculate_angles_batch_raw
                self.calculate_distances_batch_compiled = self._calculate_distances_batch_raw
        except:
            # Fallback if compilation fails
            self.calculate_angles_batch_compiled = self._calculate_angles_batch_raw
            self.calculate_distances_batch_compiled = self._calculate_distances_batch_raw

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

    def extract_scene_info_from_filename(self, filename: str) -> Optional[Dict]:
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

    def apply_savgol_filter_torch(self, data: torch.Tensor, window_length: int = 21,
                                  polyorder: int = 2) -> torch.Tensor:
        """Apply Savitzky-Golay filter using torch operations for better performance"""
        # For now, we'll use scipy for the filter but convert to/from torch tensors efficiently
        # This could be optimized further with a pure torch implementation
        if data.is_cuda:
            data_cpu = data.cpu().numpy()
        else:
            data_cpu = data.numpy()

        filtered_data = savgol_filter(
            x=data_cpu, window_length=window_length, polyorder=polyorder,
            deriv=0, axis=0, delta=0.1
        )

        return torch.from_numpy(filtered_data).to(self.device)

    def perform_joint_analysis(self, point_history: torch.Tensor) -> Tuple:
        """Perform joint analysis on point history data"""
        # Convert to numpy for the existing joint analysis function
        if isinstance(point_history, torch.Tensor):
            point_history_np = point_history.cpu().numpy()
        else:
            point_history_np = point_history

        joint_params, best_joint, info_dict = compute_joint_info_all_types(
            point_history_np,
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

    def _calculate_angles_batch_raw(self, v1_batch: torch.Tensor, v2_batch: torch.Tensor) -> torch.Tensor:
        """Calculate angles between batches of vectors using torch operations"""
        # v1_batch, v2_batch: (N, 3) tensors

        # Normalize vectors
        v1_norm = torch.norm(v1_batch, dim=1, keepdim=True)
        v2_norm = torch.norm(v2_batch, dim=1, keepdim=True)

        # Handle zero vectors
        valid_mask = (v1_norm.squeeze() > 1e-6) & (v2_norm.squeeze() > 1e-6)

        v1_normalized = v1_batch / (v1_norm + 1e-9)
        v2_normalized = v2_batch / (v2_norm + 1e-9)

        # Calculate angles (use absolute value of dot product to get minimum angle)
        dot_products = torch.sum(v1_normalized * v2_normalized, dim=1)
        dot_products = torch.clamp(torch.abs(dot_products), 0.0, 1.0)

        angles_rad = torch.acos(dot_products)
        angles_deg = torch.rad2deg(angles_rad)

        # Return the smaller angle (0-90 degrees)
        angles_deg = torch.minimum(angles_deg, 180 - angles_deg)

        # Set invalid angles to inf
        angles_deg[~valid_mask] = float('inf')

        return angles_deg

    def _calculate_distances_batch_raw(self, axis1_batch: torch.Tensor, origin1_batch: torch.Tensor,
                                       axis2_batch: torch.Tensor, origin2_batch: torch.Tensor) -> torch.Tensor:
        """Calculate minimum distances between batches of lines using torch operations"""
        # All inputs: (N, 3) tensors

        # Normalize axes
        axis1_norm = torch.norm(axis1_batch, dim=1, keepdim=True)
        axis2_norm = torch.norm(axis2_batch, dim=1, keepdim=True)

        valid_mask = (axis1_norm.squeeze() > 1e-6) & (axis2_norm.squeeze() > 1e-6)

        axis1_normalized = axis1_batch / (axis1_norm + 1e-9)
        axis2_normalized = axis2_batch / (axis2_norm + 1e-9)

        # Calculate cross products
        cross_products = torch.cross(axis1_normalized, axis2_normalized, dim=1)
        cross_norms = torch.norm(cross_products, dim=1)

        # For parallel lines (cross_norm ≈ 0)
        parallel_mask = cross_norms < 1e-6

        distances = torch.zeros(axis1_batch.shape[0], device=self.device)

        # Handle parallel lines
        if parallel_mask.any():
            vec_to_point = origin1_batch[parallel_mask] - origin2_batch[parallel_mask]
            proj_on_axis = torch.sum(vec_to_point * axis2_normalized[parallel_mask], dim=1, keepdim=True) * \
                           axis2_normalized[parallel_mask]
            perpendicular = vec_to_point - proj_on_axis
            distances[parallel_mask] = torch.norm(perpendicular, dim=1)

        # Handle skew lines
        skew_mask = ~parallel_mask & valid_mask
        if skew_mask.any():
            vec_between = origin2_batch[skew_mask] - origin1_batch[skew_mask]
            cross_normalized = cross_products[skew_mask] / cross_norms[skew_mask].unsqueeze(1)
            distances[skew_mask] = torch.abs(torch.sum(vec_between * cross_normalized, dim=1))

        # Set invalid distances to inf
        distances[~valid_mask] = float('inf')

        return distances

    def calculate_angles_batch(self, v1_batch: torch.Tensor, v2_batch: torch.Tensor) -> torch.Tensor:
        """Wrapper for compiled angle calculation function"""
        return self.calculate_angles_batch_compiled(v1_batch, v2_batch)

    def calculate_distances_batch(self, axis1_batch: torch.Tensor, origin1_batch: torch.Tensor,
                                  axis2_batch: torch.Tensor, origin2_batch: torch.Tensor) -> torch.Tensor:
        """Wrapper for compiled distance calculation function"""
        return self.calculate_distances_batch_compiled(axis1_batch, origin1_batch, axis2_batch, origin2_batch)

    def get_ground_truth_for_object(self, scene_info: Dict) -> Optional[Dict]:
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

    def load_and_preprocess_data(self, file_path: str) -> Tuple[torch.Tensor, str, Dict]:
        """Load and preprocess a single file"""
        filename = os.path.basename(file_path).split('.')[0]
        scene_info = self.extract_scene_info_from_filename(filename)

        # Load data
        data = np.load(file_path)
        data_tensor = torch.from_numpy(data).float().to(self.device)

        # Apply Savitzky-Golay filter
        data_filtered = self.apply_savgol_filter_torch(data_tensor)

        return data_filtered, filename, scene_info

    def evaluate_single_file_optimized(self, file_path: str) -> Dict:
        """Evaluate a single file with optimized operations"""
        print(f"\nEvaluating: {file_path}")

        try:
            data_filtered, filename, scene_info = self.load_and_preprocess_data(file_path)
            print(f"Loaded data shape: {data_filtered.shape}")
        except Exception as e:
            return {
                "filename": os.path.basename(file_path).split('.')[0],
                "error": f"Could not load/preprocess file: {e}"
            }

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

        # Perform joint analysis
        try:
            joint_params, best_joint, info_dict = self.perform_joint_analysis(data_filtered)
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

        # Calculate errors based on joint type using optimized functions
        if best_joint == "prismatic":
            self.calculate_prismatic_errors_optimized(result, joint_params[best_joint], ground_truth)
        elif best_joint == "revolute":
            self.calculate_revolute_errors_optimized(result, joint_params[best_joint], ground_truth)
        elif best_joint == "planar":
            self.calculate_planar_errors_optimized(result, joint_params[best_joint], ground_truth)

        return result

    def calculate_prismatic_errors_optimized(self, result: Dict, estimated_params: Dict, ground_truth: Dict):
        """Calculate errors for prismatic joints using optimized torch operations"""
        est_axis = torch.tensor(estimated_params.get("axis", [0, 0, 0]), dtype=torch.float32, device=self.device)
        gt_axis = torch.tensor(ground_truth["axis"], dtype=torch.float32, device=self.device)

        # Calculate angle error using batch function (even for single pair)
        angle_error = self.calculate_angles_batch(est_axis.unsqueeze(0), gt_axis.unsqueeze(0))[0]

        result["errors"] = {
            "axis_angle_error_degrees": float(angle_error.cpu())
        }

        print(f"Prismatic joint - Axis angle error: {float(angle_error):.2f}°")

    def calculate_revolute_errors_optimized(self, result: Dict, estimated_params: Dict, ground_truth: Dict):
        """Calculate errors for revolute joints using optimized torch operations"""
        est_axis = torch.tensor(estimated_params.get("axis", [0, 0, 0]), dtype=torch.float32, device=self.device)
        est_origin = torch.tensor(estimated_params.get("origin", [0, 0, 0]), dtype=torch.float32, device=self.device)
        gt_axis = torch.tensor(ground_truth["axis"], dtype=torch.float32, device=self.device)

        # Handle ground truth pivot
        gt_pivot = ground_truth["pivot"]
        if isinstance(gt_pivot, list):
            if len(gt_pivot) == 1:
                gt_origin = torch.zeros(3, device=self.device) + float(gt_pivot[0]) * gt_axis
            elif len(gt_pivot) == 3:
                gt_origin = torch.tensor(gt_pivot, dtype=torch.float32, device=self.device)
            else:
                gt_origin = torch.zeros(3, device=self.device)
        else:
            gt_origin = torch.zeros(3, device=self.device) + float(gt_pivot) * gt_axis

        # Calculate angle error
        angle_error = self.calculate_angles_batch(est_axis.unsqueeze(0), gt_axis.unsqueeze(0))[0]

        # Calculate distance between axes
        axis_distance = self.calculate_distances_batch(
            est_axis.unsqueeze(0), est_origin.unsqueeze(0),
            gt_axis.unsqueeze(0), gt_origin.unsqueeze(0)
        )[0]

        # Calculate origin distance
        origin_distance = torch.norm(est_origin - gt_origin)

        result["errors"] = {
            "axis_angle_error_degrees": float(angle_error.cpu()),
            "axis_distance_meters": float(axis_distance.cpu()),
            "origin_distance_meters": float(origin_distance.cpu())
        }

        print(f"Revolute joint - Axis angle error: {float(angle_error):.2f}°, "
              f"Axis distance: {float(axis_distance):.4f}m, "
              f"Origin distance: {float(origin_distance):.4f}m")

    def calculate_planar_errors_optimized(self, result: Dict, estimated_params: Dict, ground_truth: Dict):
        """Calculate errors for planar joints using optimized torch operations"""
        est_normal = torch.tensor(estimated_params.get("normal", [0, 0, 0]), dtype=torch.float32, device=self.device)
        gt_normal = torch.tensor(ground_truth["axis"], dtype=torch.float32, device=self.device)

        # Calculate angle error
        angle_error = self.calculate_angles_batch(est_normal.unsqueeze(0), gt_normal.unsqueeze(0))[0]

        result["errors"] = {
            "normal_angle_error_degrees": float(angle_error.cpu())
        }

        print(f"Planar joint - Normal angle error: {float(angle_error):.2f}°")

    def evaluate_batch_files(self, file_batch: List[str]) -> List[Dict]:
        """Evaluate a batch of files"""
        results = []
        for file_path in file_batch:
            result = self.evaluate_single_file_optimized(file_path)
            results.append(result)
        return results

    def evaluate_all_files_optimized(self) -> Dict:
        """Evaluate all files with optimized batch processing and multiprocessing"""
        print("Starting optimized joint analysis evaluation...")
        print(f"Total files to evaluate: {len(self.file_paths)}")
        print(f"Using device: {self.device}")
        print(f"Multiprocessing: {self.use_multiprocessing}, Workers: {self.num_workers}")

        all_results = []

        if self.use_multiprocessing and len(self.file_paths) > 1:
            # Split files into batches for multiprocessing
            batch_size = max(1, len(self.file_paths) // self.num_workers)
            file_batches = [self.file_paths[i:i + batch_size] for i in range(0, len(self.file_paths), batch_size)]

            print(f"Processing {len(file_batches)} batches with {len(file_batches[0])} files each")

            # Note: For GPU operations, we need to be careful with multiprocessing
            # Using ThreadPoolExecutor instead of ProcessPoolExecutor for GPU compatibility
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                batch_results = list(executor.map(self.evaluate_batch_files, file_batches))

            # Flatten results
            for batch_result in batch_results:
                all_results.extend(batch_result)
        else:
            # Sequential processing
            for file_path in self.file_paths:
                result = self.evaluate_single_file_optimized(file_path)
                all_results.append(result)

        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(all_results)

        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "device_used": str(self.device),
            "multiprocessing_used": self.use_multiprocessing,
            "num_workers": self.num_workers,
            "summary_statistics": summary_stats,
            "detailed_results": all_results
        }

    def _calculate_summary_statistics(self, all_results: List[Dict]) -> Dict:
        """Calculate summary statistics from results"""
        summary_stats = {
            "total_files": len(all_results),
            "successful_evaluations": 0,
            "classification_correct": 0,
            "classification_incorrect": 0,
            "errors_occurred": 0,
            "by_object_type": {},
            "by_joint_type": {}
        }

        for result in all_results:
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

        return summary_stats

    def convert_numpy_to_python(self, obj):
        """Convert numpy arrays and torch tensors to Python types for JSON serialization"""
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            if isinstance(obj, torch.Tensor):
                obj = obj.cpu().numpy()
            return obj.tolist()
        elif isinstance(obj, (np.integer, torch.IntTensor)):
            return int(obj)
        elif isinstance(obj, (np.floating, torch.FloatTensor)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_python(item) for item in obj]
        else:
            return obj

    def save_results(self, results: Dict, output_file: str) -> bool:
        """Save evaluation results to JSON file"""
        try:
            # Convert numpy arrays and torch tensors to Python lists for JSON serialization
            serializable_results = self.convert_numpy_to_python(results)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

    def print_summary(self, results: Dict):
        """Print a summary of the evaluation results"""
        stats = results["summary_statistics"]

        print("\n" + "=" * 80)
        print("OPTIMIZED EVALUATION SUMMARY")
        print("=" * 80)

        print(f"Device used: {results.get('device_used', 'Unknown')}")
        print(f"Multiprocessing: {results.get('multiprocessing_used', False)}")
        print(f"Workers: {results.get('num_workers', 1)}")
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


def main_optimized():
    """Main function for optimized evaluation"""
    # Example file paths - replace with your actual file paths
    file_paths = [
        "/common/homes/all/uksqc_chen/projects/control/ParaHome/output_specific_actions/s204_drawer_part2_1320_1440.npy"
        # Add more file paths here
    ]

    print(f"Found {len(file_paths)} files to evaluate")
    if len(file_paths) == 0:
        print("No files found. Please check your file paths.")
        return

    # Create optimized evaluator
    evaluator = OptimizedJointAnalysisEvaluator(
        file_paths=file_paths,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_multiprocessing=True,
        num_workers=4
    )

    # Perform evaluation
    import time
    start_time = time.time()
    results = evaluator.evaluate_all_files_optimized()
    end_time = time.time()

    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")

    # Print summary
    evaluator.print_summary(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"optimized_joint_analysis_evaluation_{timestamp}.json"
    success = evaluator.save_results(results, output_file)

    if success:
        print(f"\n✓ Optimized evaluation completed successfully!")
        print(f"✓ Results saved to: {output_file}")
        print(f"✓ Classification accuracy: {results['summary_statistics']['classification_accuracy']:.1f}%")
        print(f"✓ Total time: {end_time - start_time:.2f} seconds")
    else:
        print("\n✗ Failed to save results")


def main_batch_directory_optimized(directory_path: str, device: str = 'auto'):
    """Process all .npy files in a directory with optimized operations"""
    file_paths = glob.glob(os.path.join(directory_path, "*.npy"))

    if len(file_paths) == 0:
        print(f"No .npy files found in directory: {directory_path}")
        return

    print(f"Found {len(file_paths)} .npy files in {directory_path}")

    # Auto-detect device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create optimized evaluator
    evaluator = OptimizedJointAnalysisEvaluator(
        file_paths=file_paths,
        device=device,
        use_multiprocessing=True,
        num_workers=min(8, len(file_paths))  # Adaptive number of workers
    )

    # Perform evaluation
    import time
    start_time = time.time()
    results = evaluator.evaluate_all_files_optimized()
    end_time = time.time()

    print(f"\nBatch evaluation completed in {end_time - start_time:.2f} seconds")

    # Print summary
    evaluator.print_summary(results)

    # Save results with directory name in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.basename(directory_path.rstrip('/'))
    output_file = f"optimized_joint_analysis_evaluation_{dir_name}_{timestamp}.json"

    success = evaluator.save_results(results, output_file)

    if success:
        print(f"\n✓ Optimized batch evaluation completed successfully!")
        print(f"✓ Results saved to: {output_file}")
        print(f"✓ Classification accuracy: {results['summary_statistics']['classification_accuracy']:.1f}%")
        print(f"✓ Total time: {end_time - start_time:.2f} seconds")
        print(f"✓ Average time per file: {(end_time - start_time) / len(file_paths):.3f} seconds")


if __name__ == "__main__":
    # Use this for specific files
    main_optimized()

    # Or use this for batch processing a directory
    # main_batch_directory_optimized("/path/to/your/npy/files/directory/")