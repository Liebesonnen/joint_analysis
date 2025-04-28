"""
Demo script for the joint analysis package.

This script demonstrates the basic usage of the joint analysis package, including:
1. Generating synthetic data
2. Running joint type estimation
3. Visualizing the results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys

# Add parent directory to path to import joint_analysis
sys.path.append(str(Path(__file__).parent.parent))

from joint_analysis.synthetic import SyntheticJointGenerator
from joint_analysis.core import compute_joint_info_all_types
from joint_analysis.data import load_numpy_sequence

# Optional imports for visualization
try:
    import polyscope as ps
    from joint_analysis.viz import PolyscopeVisualizer

    POLYSCOPE_AVAILABLE = True
except ImportError:
    POLYSCOPE_AVAILABLE = False
    print("Polyscope not available. Skipping 3D visualization.")


def generate_and_save_data():
    """Generate synthetic data and save it to disk."""
    print("Generating synthetic data...")

    # Set up output directory
    output_dir = "demo_data"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize data generator
    gen = SyntheticJointGenerator(output_dir=output_dir, num_points=500, noise_sigma=0.001)

    # Generate sequences for different joint types
    print("Generating prismatic joint sequence...")
    prismatic_seq = gen.generate_prismatic_door_sequence(n_frames=50)
    np.save(os.path.join(output_dir, "prismatic.npy"), prismatic_seq)

    print("Generating revolute joint sequence...")
    revolute_seq = gen.generate_revolute_door_sequence(n_frames=50)
    np.save(os.path.join(output_dir, "revolute.npy"), revolute_seq)

    print("Generating planar joint sequence...")
    planar_seq = gen.generate_planar_mouse_sequence(n_frames=50)
    np.save(os.path.join(output_dir, "planar.npy"), planar_seq)

    print("Generating ball joint sequence...")
    ball_seq = gen.generate_ball_joint_sequence(n_frames=50)
    np.save(os.path.join(output_dir, "ball.npy"), ball_seq)

    print("Generating screw joint sequence...")
    screw_seq = gen.generate_screw_joint_sequence(n_frames=50)
    np.save(os.path.join(output_dir, "screw.npy"), screw_seq)

    print(f"Data saved to {output_dir}")

    return {
        "prismatic": prismatic_seq,
        "revolute": revolute_seq,
        "planar": planar_seq,
        "ball": ball_seq,
        "screw": screw_seq
    }


def analyze_sequence(sequence, joint_type):
    """
    Analyze a point cloud sequence to estimate joint type.

    Args:
        sequence (ndarray): Point cloud sequence of shape (T, N, 3)
        joint_type (str): Ground truth joint type for display

    Returns:
        tuple: (param_dict, best_type, scores_info)
    """
    print(f"Analyzing {joint_type} sequence...")
    print(f"Sequence shape: {sequence.shape}")

    # Run joint type estimation
    param_dict, best_type, scores_info = compute_joint_info_all_types(
        sequence,
        neighbor_k=50,
        col_sigma=0.2, col_order=4.0,
        cop_sigma=0.2, cop_order=4.0,
        rad_sigma=0.2, rad_order=4.0,
        zp_sigma=0.2, zp_order=4.0,
        prob_sigma=0.2, prob_order=4.0,
        use_savgol=True,
        savgol_window=5,
        savgol_poly=2
    )

    # Print results
    print(f"Ground truth: {joint_type}")
    print(f"Estimated joint type: {best_type}")

    if scores_info is not None:
        # Print basic scores
        basic_scores = scores_info["basic_score_avg"]
        print("Basic Scores:")
        print(f"  Collinearity: {basic_scores['col_mean']:.3f}")
        print(f"  Coplanarity: {basic_scores['cop_mean']:.3f}")
        print(f"  Radius Consistency: {basic_scores['rad_mean']:.3f}")
        print(f"  Zero Pitch: {basic_scores['zp_mean']:.3f}")

        # Print joint probabilities
        joint_probs = scores_info["joint_probs"]
        print("Joint Probabilities:")
        for jt, prob in sorted(joint_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {jt}: {prob:.3f}")

    if best_type in param_dict:
        # Print joint parameters
        params = param_dict[best_type]
        print(f"{best_type.capitalize()} Joint Parameters:")

        if best_type == "planar":
            n_ = params["normal"]
            lim = params["motion_limit"]
            print(f"  Normal: ({n_[0]:.2f}, {n_[1]:.2f}, {n_[2]:.2f})")
            print(f"  Motion Limit: ({lim[0]:.2f}, {lim[1]:.2f})")

        elif best_type == "ball":
            c_ = params["center"]
            lim = params["motion_limit"]
            print(f"  Center: ({c_[0]:.2f}, {c_[1]:.2f}, {c_[2]:.2f})")
            print(f"  Motion Limit: Rx:{lim[0]:.2f}, Ry:{lim[1]:.2f}, Rz:{lim[2]:.2f}")

        elif best_type == "screw":
            a_ = params["axis"]
            o_ = params["origin"]
            p_ = params["pitch"]
            lim = params["motion_limit"]
            print(f"  Axis: ({a_[0]:.2f}, {a_[1]:.2f}, {a_[2]:.2f})")
            print(f"  Origin: ({o_[0]:.2f}, {o_[1]:.2f}, {o_[2]:.2f})")
            print(f"  Pitch: {p_:.3f}")
            print(f"  Motion Limit: ({lim[0]:.2f} rad, {lim[1]:.2f} rad)")

        elif best_type == "prismatic":
            a_ = params["axis"]
            o_ = params.get("origin", np.array([0., 0., 0.]))
            lim = params["motion_limit"]
            print(f"  Axis: ({a_[0]:.2f}, {a_[1]:.2f}, {a_[2]:.2f})")
            print(f"  Origin: ({o_[0]:.2f}, {o_[1]:.2f}, {o_[2]:.2f})")
            print(f"  Motion Limit: ({lim[0]:.2f}, {lim[1]:.2f})")

        elif best_type == "revolute":
            a_ = params["axis"]
            o_ = params["origin"]
            lim = params["motion_limit"]
            print(f"  Axis: ({a_[0]:.2f}, {a_[1]:.2f}, {a_[2]:.2f})")
            print(f"  Origin: ({o_[0]:.2f}, {o_[1]:.2f}, {o_[2]:.2f})")
            print(f"  Motion Limit: ({lim[0]:.2f} rad, {lim[1]:.2f} rad)")

    print()

    return param_dict, best_type, scores_info


def plot_scores(scores_info, title):
    """
    Plot basic scores and joint probabilities.

    Args:
        scores_info (dict): Scores information from joint estimation
        title (str): Plot title
    """
    if scores_info is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot basic scores
    basic_scores = scores_info["basic_score_avg"]
    score_names = ["Collinearity", "Coplanarity", "Radius Cons.", "Zero Pitch"]
    score_values = [
        basic_scores["col_mean"],
        basic_scores["cop_mean"],
        basic_scores["rad_mean"],
        basic_scores["zp_mean"]
    ]

    ax1.bar(score_names, score_values, color=['blue', 'green', 'red', 'purple'])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Score Value")
    ax1.set_title("Basic Scores")

    # Plot joint probabilities
    joint_probs = scores_info["joint_probs"]
    joint_names = list(joint_probs.keys())
    prob_values = list(joint_probs.values())

    # Sort by probability
    sorted_idx = np.argsort(prob_values)[::-1]
    joint_names = [joint_names[i] for i in sorted_idx]
    prob_values = [prob_values[i] for i in sorted_idx]

    ax2.bar(joint_names, prob_values, color=['blue', 'green', 'red', 'purple', 'orange'])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probability")
    ax2.set_title("Joint Type Probabilities")

    fig.suptitle(title)
    plt.tight_layout()

    return fig


def visualize_3d(sequence, joint_type, param_dict=None, best_type=None):
    if not POLYSCOPE_AVAILABLE:
        print("Polyscope not available. Skipping 3D visualization.")
        return

    if not ps.is_initialized():
        ps.init()

    ps.remove_all_structures()

    viz = PolyscopeVisualizer()
    viz.register_point_cloud(f"{joint_type}_sequence", sequence[0])  # 只画第0帧

    if param_dict is not None and best_type is not None and best_type in param_dict:
        viz.show_joint_visualization(best_type, param_dict[best_type])

    center = np.mean(sequence[0], axis=0)
    viz.setup_camera(center)

    ps.show()


def run_demo():
    """Run the full demo."""
    # Generate synthetic data
    data_dir = "demo_data"
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        sequences = generate_and_save_data()
    else:
        print(f"Loading data from {data_dir}...")
        sequences = {
            "prismatic": load_numpy_sequence(os.path.join(data_dir, "prismatic.npy")),
            "revolute": load_numpy_sequence(os.path.join(data_dir, "revolute.npy")),
            "planar": load_numpy_sequence(os.path.join(data_dir, "planar.npy")),
            "ball": load_numpy_sequence(os.path.join(data_dir, "ball.npy")),
            "screw": load_numpy_sequence(os.path.join(data_dir, "screw.npy"))
        }

    # Analyze each sequence
    results = {}
    for joint_type, sequence in sequences.items():
        param_dict, best_type, scores_info = analyze_sequence(sequence, joint_type)
        results[joint_type] = (param_dict, best_type, scores_info)

        # Plot and save scores
        if scores_info is not None:
            fig = plot_scores(scores_info, f"{joint_type.capitalize()} Joint Analysis")
            os.makedirs("demo_plots", exist_ok=True)
            fig.savefig(f"demo_plots/{joint_type}_scores.png")

    # Ask user if they want to see 3D visualization
    if POLYSCOPE_AVAILABLE:
        print("\nDo you want to see 3D visualizations? (y/n)")
        choice = input().strip().lower()

        if choice.startswith('y'):
            for joint_type, (param_dict, best_type, _) in results.items():
                print(f"\nVisualizing {joint_type} joint...")
                visualize_3d(sequences[joint_type], joint_type, param_dict, best_type)

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()