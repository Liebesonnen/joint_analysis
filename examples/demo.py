"""
完全兼容的演示脚本，可在各种Polyscope版本上运行
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import threading

import sys

# Add parent directory to path to import joint_analysis
sys.path.append(str(Path(__file__).parent.parent))

from joint_analysis.synthetic import SyntheticJointGenerator
from joint_analysis.core import compute_joint_info_all_types
from joint_analysis.data import load_numpy_sequence

# Optional imports for visualization
try:
    import polyscope as ps
    import polyscope.imgui as psim
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
    gen = SyntheticJointGenerator(output_dir=output_dir, num_points=500, noise_sigma=0.00)

    # Generate sequences for different joint types
    print("Generating prismatic joint sequence...")
    prismatic_seq = gen.generate_prismatic_door_sequence(n_frames=150)
    np.save(os.path.join(output_dir, "prismatic.npy"), prismatic_seq)

    print("Generating revolute joint sequence...")
    revolute_seq = gen.generate_revolute_door_sequence(n_frames=150)
    np.save(os.path.join(output_dir, "revolute.npy"), revolute_seq)

    print("Generating planar joint sequence...")
    planar_seq = gen.generate_planar_mouse_sequence(n_frames=150)
    np.save(os.path.join(output_dir, "planar.npy"), planar_seq)

    print("Generating ball joint sequence...")
    ball_seq = gen.generate_ball_joint_sequence(n_frames=150)
    np.save(os.path.join(output_dir, "ball.npy"), ball_seq)

    print("Generating screw joint sequence...")
    screw_seq = gen.generate_screw_joint_sequence(n_frames=150)
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
        neighbor_k=10,
        col_sigma=0.2, col_order=4.0,
        cop_sigma=0.2, cop_order=4.0,
        rad_sigma=0.2, rad_order=4.0,
        zp_sigma=0.2, zp_order=4.0,
        prob_sigma=0.2, prob_order=4.0,
        use_savgol=True,
        savgol_window=10,
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

    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    if scores_info is None:
        return None

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


class AnimatedVisualizer:
    """Class for animated visualization of point cloud sequences."""

    def __init__(self, sequences, results):
        """
        Initialize the visualizer.

        Args:
            sequences (dict): Dictionary mapping joint types to point cloud sequences
            results (dict): Dictionary mapping joint types to analysis results
        """
        self.sequences = sequences
        self.results = results
        self.joint_types = list(sequences.keys())

        # Initialize Polyscope if not already initialized
        if not ps.is_initialized():
            ps.init()

        # Set ground plane mode to none
        try:
            ps.set_ground_plane_mode("none")
        except Exception:
            # If this setting isn't available, continue without it
            pass

        self.viz = PolyscopeVisualizer()

        # State variables
        self.current_joint_idx = 0
        self.current_frame_idx = 0
        self.playing = False
        self.play_speed = 1
        self.show_joint_vis = True
        self.show_previous_frames = False
        self.num_previous_frames = 5
        self.trail_alpha = 0.5
        self.auto_advance = False
        self.loop_playback = True
        self.frame_delay = 0.05  # seconds between frames
        self.last_frame_time = time.time()
        self.show_ground_truth = True
        self.point_size = 0.01
        self.color_by_motion = False

        # Thread for continuous playback
        self.playback_thread = None
        self.stop_playback = False

        # Initialize point clouds
        self._init_point_clouds()
        self._update_visualization()

    def _init_point_clouds(self):
        """Initialize all point clouds."""
        # Register each joint type's first frame
        for joint_type in self.joint_types:
            sequence = self.sequences[joint_type]
            # Use the first frame initially
            self.viz.register_point_cloud(
                joint_type,
                sequence[0],
                enabled=(joint_type == self.joint_types[self.current_joint_idx])
            )

            # Set initial point size
            if ps.has_point_cloud(joint_type):
                try:
                    ps.get_point_cloud(joint_type).set_radius(self.point_size)
                except Exception:
                    pass

    def _update_visualization(self):
        """Update the current visualization with the current frame."""
        # Get current joint type and sequence
        joint_type = self.joint_types[self.current_joint_idx]
        sequence = self.sequences[joint_type]

        # Get current frame
        frame_idx = min(self.current_frame_idx, len(sequence) - 1)
        current_frame = sequence[frame_idx]

        # Update main point cloud
        self.viz.update_point_cloud(joint_type, current_frame)

        # Color points by motion if enabled
        if self.color_by_motion and frame_idx > 0:
            prev_frame = sequence[frame_idx - 1]
            dists = np.linalg.norm(current_frame - prev_frame, axis=1)
            max_dist = np.max(dists) if np.max(dists) > 0 else 1.0

            # Create color array based on motion magnitude (red = high motion, blue = low motion)
            colors = np.zeros((current_frame.shape[0], 3))
            norm_dists = dists / max_dist
            colors[:, 0] = norm_dists  # Red channel
            colors[:, 2] = 1.0 - norm_dists  # Blue channel

            # Apply colors to point cloud
            if ps.has_point_cloud(joint_type):
                # Use try/except to handle missing has_quantity method
                try:
                    # First try to check if the quantity exists
                    if hasattr(ps.get_point_cloud(joint_type), 'has_quantity'):
                        if ps.get_point_cloud(joint_type).has_quantity("motion_color"):
                            ps.get_point_cloud(joint_type).remove_quantity("motion_color")
                    # If that doesn't work, just try to add the quantity (it will replace any existing one)
                    ps.get_point_cloud(joint_type).add_color_quantity("motion_color", colors, enabled=True)
                except Exception as e:
                    # Fall back to just adding the quantity
                    try:
                        ps.get_point_cloud(joint_type).add_color_quantity("motion_color", colors, enabled=True)
                    except Exception:
                        # If all else fails, just ignore coloring
                        pass
        else:
            # Try to remove motion coloring if not enabled
            if ps.has_point_cloud(joint_type):
                try:
                    # First try to check if the quantity exists
                    if hasattr(ps.get_point_cloud(joint_type), 'has_quantity'):
                        if ps.get_point_cloud(joint_type).has_quantity("motion_color"):
                            ps.get_point_cloud(joint_type).remove_quantity("motion_color")
                except Exception:
                    # If that fails, we can't do much, so just continue
                    pass

        # Remove previous trail point clouds
        for i in range(1, 30):  # Arbitrary large number to ensure all trails are removed
            trail_name = f"{joint_type}_trail_{i}"
            if ps.has_point_cloud(trail_name):
                ps.remove_point_cloud(trail_name)

        # Add trail of previous frames if enabled
        if self.show_previous_frames and frame_idx > 0:
            for i in range(1, min(self.num_previous_frames + 1, frame_idx + 1)):
                trail_name = f"{joint_type}_trail_{i}"
                prev_frame = sequence[frame_idx - i]

                # Calculate alpha value based on frame distance
                alpha = self.trail_alpha * (self.num_previous_frames - i + 1) / self.num_previous_frames

                # Create color array with decreasing alpha
                colors = np.ones((prev_frame.shape[0], 3)) * 0.7  # Light gray
                colors[:, 0] = 0.9 - (i / self.num_previous_frames) * 0.5  # Gradually change color

                # Register trail point cloud
                trail_pc = ps.register_point_cloud(trail_name, prev_frame, enabled=True)
                try:
                    trail_pc.set_radius(self.point_size * 0.7)  # Smaller radius for trail points
                except Exception:
                    # If set_radius fails, try other ways or ignore
                    pass

                try:
                    trail_pc.add_color_quantity("trail_color", colors, enabled=True)
                except Exception:
                    # If coloring fails, continue without it
                    pass

                # Try to set transparency if available
                try:
                    trail_pc.set_transparency(1.0 - alpha)
                except Exception:
                    # If transparency setting fails, continue without it
                    pass

        # Show joint visualization if enabled
        if self.show_joint_vis:
            # First remove any existing joint visualization
            self.viz.remove_joint_visualization()

            # Get results for current joint type
            param_dict, best_type, _ = self.results[joint_type]

            # Show joint visualization
            if best_type in param_dict:
                self.viz.set_joint_estimation_result(best_type, param_dict[best_type])

                # Show ground truth visualization if enabled
                if self.show_ground_truth:
                    ground_truth_param = {
                        "axis": np.array([1.0, 0.0, 0.0]) if joint_type == "prismatic" else
                        np.array([0.0, 1.0, 0.0]) if joint_type == "revolute" else
                        np.array([0.0, 1.0, 0.0]) if joint_type == "screw" else
                        np.array([0.0, 0.0, 0.0]),
                        "origin": np.array([0.0, 0.0, 0.0]),
                        "normal": np.array([0.0, 1.0, 0.0]) if joint_type == "planar" else np.array([0.0, 0.0, 0.0]),
                        "center": np.array([0.0, 0.0, 0.0]) if joint_type == "ball" else np.array([0.0, 0.0, 0.0]),
                        "pitch": 0.5 if joint_type == "screw" else 0.0,
                        "motion_limit": (0.0, 0.0)
                    }

                    try:
                        self.viz.show_joint_visualization(joint_type, ground_truth_param, is_ground_truth=True)
                    except Exception:
                        # If showing ground truth fails, continue without it
                        pass
        else:
            # Remove joint visualization if disabled
            self.viz.remove_joint_visualization()

    def _switch_joint_type(self, new_idx):
        """Switch to a different joint type."""
        if new_idx == self.current_joint_idx:
            return

        # Disable current joint point cloud
        old_joint_type = self.joint_types[self.current_joint_idx]
        if ps.has_point_cloud(old_joint_type):
            ps.get_point_cloud(old_joint_type).set_enabled(False)

        # Remove trail point clouds for old joint type
        for i in range(1, 30):  # Arbitrary large number to ensure all trails are removed
            trail_name = f"{old_joint_type}_trail_{i}"
            if ps.has_point_cloud(trail_name):
                ps.remove_point_cloud(trail_name)

        # Update current joint index
        self.current_joint_idx = new_idx

        # Enable new joint point cloud
        new_joint_type = self.joint_types[self.current_joint_idx]
        if ps.has_point_cloud(new_joint_type):
            ps.get_point_cloud(new_joint_type).set_enabled(True)

        # Reset frame index
        self.current_frame_idx = 0

        # Remove joint visualization
        self.viz.remove_joint_visualization()

        # Update visualization
        self._update_visualization()

    def _advance_frame(self):
        """Advance to the next frame."""
        joint_type = self.joint_types[self.current_joint_idx]
        sequence = self.sequences[joint_type]

        # Increment frame index
        self.current_frame_idx += self.play_speed

        # Handle frame wrapping
        if self.current_frame_idx >= len(sequence):
            if self.auto_advance:
                # Move to next joint type
                new_idx = (self.current_joint_idx + 1) % len(self.joint_types)
                self._switch_joint_type(new_idx)
            elif self.loop_playback:
                # Loop back to beginning
                self.current_frame_idx = 0
            else:
                # Stop at end
                self.current_frame_idx = len(sequence) - 1
                self.playing = False

        # Update visualization
        self._update_visualization()

    def _reverse_frame(self):
        """Go to the previous frame."""
        # Decrement frame index
        self.current_frame_idx -= self.play_speed

        # Handle frame wrapping
        if self.current_frame_idx < 0:
            if self.auto_advance:
                # Move to previous joint type
                new_idx = (self.current_joint_idx - 1) % len(self.joint_types)
                self._switch_joint_type(new_idx)
                # Set to last frame
                joint_type = self.joint_types[self.current_joint_idx]
                sequence = self.sequences[joint_type]
                self.current_frame_idx = len(sequence) - 1
            elif self.loop_playback:
                # Loop to last frame
                joint_type = self.joint_types[self.current_joint_idx]
                sequence = self.sequences[joint_type]
                self.current_frame_idx = len(sequence) - 1
            else:
                # Stop at beginning
                self.current_frame_idx = 0
                self.playing = False

        # Update visualization
        self._update_visualization()

    def _export_current_frame(self):
        """Export the current frame as an image."""
        joint_type = self.joint_types[self.current_joint_idx]
        os.makedirs("demo_screenshots", exist_ok=True)
        filename = f"demo_screenshots/{joint_type}_frame_{self.current_frame_idx}.png"
        try:
            ps.screenshot(filename)
            print(f"Screenshot saved as {filename}")
        except Exception as e:
            print(f"Failed to save screenshot: {e}")

    def _playback_thread_func(self):
        """Thread function for continuous playback."""
        while not self.stop_playback and self.playing:
            time.sleep(self.frame_delay / self.play_speed)
            if self.playing:  # Check again in case it was changed
                self._advance_frame()

    def _start_playback_thread(self):
        """Start the playback thread."""
        self.stop_playback = False
        self.playback_thread = threading.Thread(target=self._playback_thread_func)
        self.playback_thread.daemon = True
        self.playback_thread.start()

    def _stop_playback_thread(self):
        """Stop the playback thread."""
        self.stop_playback = True
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(0.5)  # Wait up to 0.5 seconds for thread to finish
        self.playback_thread = None

    def _toggle_play(self):
        """Toggle playback state."""
        self.playing = not self.playing

        if self.playing:
            self._start_playback_thread()
        else:
            self._stop_playback_thread()

    # def polyscope_callback(self):
    #     """Callback function for polyscope UI, compatible with older versions."""
    #     # Joint type selector
    #     psim.TextUnformatted("Joint Type:")
    #     joint_selector_width = 300
    #
    #     # Create a horizontal layout for joint type buttons
    #     with psim.Indent():
    #         for i, joint_type in enumerate(self.joint_types):
    #             if i % 3 == 0 and i > 0:
    #                 # Start a new row after every 3 buttons
    #                 psim.NewLine()
    #
    #             # Color the button based on selection
    #             if self.current_joint_idx == i:
    #                 # Set button colors if available
    #                 try:
    #                     psim.PushStyleColor(psim.ImGuiCol_Button, (0.2, 0.7, 0.3, 0.7))
    #                     psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, (0.3, 0.8, 0.4, 1.0))
    #                     psim.PushStyleColor(psim.ImGuiCol_ButtonActive, (0.4, 0.9, 0.5, 1.0))
    #                 except Exception:
    #                     # If styling isn't available, continue without it
    #                     pass
    #
    #             # Create the button
    #             if psim.Button(joint_type.capitalize()):
    #                 self._switch_joint_type(i)
    #
    #             # Remove color styling if it was applied
    #             if self.current_joint_idx == i:
    #                 try:
    #                     psim.PopStyleColor(3)
    #                 except Exception:
    #                     # If pop style isn't available, continue
    #                     pass
    #
    #             # Add spacing except for the last button in a row
    #             if i % 3 != 2 and i < len(self.joint_types) - 1:
    #                 psim.SameLine()
    #
    #     psim.Separator()
    #
    #     # Current joint info
    #     joint_type = self.joint_types[self.current_joint_idx]
    #     sequence = self.sequences[joint_type]
    #     total_frames = len(sequence)
    #
    #     # Frame counter and slider control
    #     psim.TextUnformatted(f"Frame: {self.current_frame_idx + 1} / {total_frames}")
    #
    #     # Frame slider
    #     changed, new_frame = psim.SliderInt("##FrameSlider", self.current_frame_idx, 0, total_frames - 1)
    #     if changed:
    #         self.current_frame_idx = new_frame
    #         self._update_visualization()
    #
    #     # Playback controls
    #     if psim.Button("First Frame"):
    #         self.current_frame_idx = 0
    #         self._update_visualization()
    #
    #     psim.SameLine()
    #
    #     if psim.Button("<<"):
    #         self.current_frame_idx = max(0, self.current_frame_idx - 10)
    #         self._update_visualization()
    #
    #     psim.SameLine()
    #
    #     if psim.Button("<"):
    #         self._reverse_frame()
    #
    #     psim.SameLine()
    #
    #     # Play/Pause button with different text based on state
    #     play_label = "Pause" if self.playing else "Play"
    #     if psim.Button(play_label):
    #         self._toggle_play()
    #
    #     psim.SameLine()
    #
    #     if psim.Button(">"):
    #         self._advance_frame()
    #
    #     psim.SameLine()
    #
    #     if psim.Button(">>"):
    #         self.current_frame_idx = min(total_frames - 1, self.current_frame_idx + 10)
    #         self._update_visualization()
    #
    #     psim.SameLine()
    #
    #     if psim.Button("Last Frame"):
    #         self.current_frame_idx = total_frames - 1
    #         self._update_visualization()
    #
    #     # Screenshot button
    #     if psim.Button("Take Screenshot"):
    #         self._export_current_frame()
    #
    #     psim.Separator()
    #
    #     # Playback options section
    #     if psim.CollapsingHeader("Playback Options"):
    #         # Play speed slider
    #         changed, new_speed = psim.SliderInt("Play Speed", self.play_speed, 1, 10)
    #         if changed:
    #             self.play_speed = new_speed
    #
    #         # Frame delay slider (controls playback smoothness)
    #         changed, new_delay = psim.SliderFloat("Frame Delay (s)", self.frame_delay, 0.01, 0.5)
    #         if changed:
    #             self.frame_delay = new_delay
    #
    #         # Loop playback checkbox
    #         changed, new_loop = psim.Checkbox("Loop Playback", self.loop_playback)
    #         if changed:
    #             self.loop_playback = new_loop
    #
    #         # Auto advance to next joint checkbox
    #         changed, new_auto_advance = psim.Checkbox("Auto Advance to Next Joint", self.auto_advance)
    #         if changed:
    #             self.auto_advance = new_auto_advance
    #
    #     # Visualization options
    #     if psim.CollapsingHeader("Visualization Options"):
    #         # Point cloud display options
    #         changed, new_point_size = psim.SliderFloat("Point Size", self.point_size, 0.001, 0.05)
    #         if changed:
    #             self.point_size = new_point_size
    #             # Update point size for all point clouds
    #             for joint_type in self.joint_types:
    #                 if ps.has_point_cloud(joint_type):
    #                     try:
    #                         ps.get_point_cloud(joint_type).set_radius(self.point_size)
    #                     except Exception:
    #                         pass
    #
    #         # Joint visualization options
    #         changed, new_joint_vis = psim.Checkbox("Show Joint Visualization", self.show_joint_vis)
    #         if changed:
    #             self.show_joint_vis = new_joint_vis
    #             self._update_visualization()
    #
    #         # Trail visualization options
    #         changed, new_prev_frames = psim.Checkbox("Show Motion Trails", self.show_previous_frames)
    #         if changed:
    #             self.show_previous_frames = new_prev_frames
    #             self._update_visualization()
    #
    #         if self.show_previous_frames:
    #             changed, new_num_frames = psim.SliderInt("Number of Trail Frames", self.num_previous_frames, 1, 20)
    #             if changed:
    #                 self.num_previous_frames = new_num_frames
    #                 self._update_visualization()
    #
    #     # Analysis results
    #     if psim.CollapsingHeader("Analysis Results"):
    #         param_dict, best_type, scores_info = self.results[joint_type]
    #
    #         psim.TextUnformatted(f"Ground Truth Joint Type: {joint_type}")
    #         psim.TextUnformatted(f"Estimated Joint Type: {best_type}")
    #
    #         if scores_info is not None:
    #             if psim.TreeNode("Basic Scores"):
    #                 basic_scores = scores_info["basic_score_avg"]
    #                 psim.TextUnformatted(f"Collinearity: {basic_scores['col_mean']:.3f}")
    #                 psim.TextUnformatted(f"Coplanarity: {basic_scores['cop_mean']:.3f}")
    #                 psim.TextUnformatted(f"Radius Consistency: {basic_scores['rad_mean']:.3f}")
    #                 psim.TextUnformatted(f"Zero Pitch: {basic_scores['zp_mean']:.3f}")
    #                 psim.TreePop()
    #
    #             if psim.TreeNode("Joint Probabilities"):
    #                 joint_probs = scores_info["joint_probs"]
    #                 for jt, prob in sorted(joint_probs.items(), key=lambda x: x[1], reverse=True):
    #                     psim.TextUnformatted(f"{jt.capitalize()}: {prob:.3f}")
    #                 psim.TreePop()
    #
    #             if best_type in param_dict and psim.TreeNode("Joint Parameters"):
    #                 params = param_dict[best_type]
    #
    #                 if best_type == "planar":
    #                     n_ = params["normal"]
    #                     lim = params["motion_limit"]
    #                     psim.TextUnformatted(f"Normal: ({n_[0]:.3f}, {n_[1]:.3f}, {n_[2]:.3f})")
    #                     psim.TextUnformatted(f"Motion Limit: ({lim[0]:.3f}, {lim[1]:.3f})")
    #
    #                 elif best_type == "ball":
    #                     c_ = params["center"]
    #                     lim = params["motion_limit"]
    #                     psim.TextUnformatted(f"Center: ({c_[0]:.3f}, {c_[1]:.3f}, {c_[2]:.3f})")
    #                     psim.TextUnformatted(f"Motion Limit: Rx:{lim[0]:.3f}, Ry:{lim[1]:.3f}, Rz:{lim[2]:.3f}")
    #
    #                 elif best_type == "screw":
    #                     a_ = params["axis"]
    #                     o_ = params["origin"]
    #                     p_ = params["pitch"]
    #                     lim = params["motion_limit"]
    #                     psim.TextUnformatted(f"Axis: ({a_[0]:.3f}, {a_[1]:.3f}, {a_[2]:.3f})")
    #                     psim.TextUnformatted(f"Origin: ({o_[0]:.3f}, {o_[1]:.3f}, {o_[2]:.3f})")
    #                     psim.TextUnformatted(f"Pitch: {p_:.3f}")
    #                     psim.TextUnformatted(f"Motion Limit: ({lim[0]:.3f} rad, {lim[1]:.3f} rad)")
    #
    #                 elif best_type == "prismatic":
    #                     a_ = params["axis"]
    #                     o_ = params.get("origin", np.array([0., 0., 0.]))
    #                     lim = params["motion_limit"]
    #                     psim.TextUnformatted(f"Axis: ({a_[0]:.3f}, {a_[1]:.3f}, {a_[2]:.3f})")
    #                     psim.TextUnformatted(f"Origin: ({o_[0]:.3f}, {o_[1]:.3f}, {o_[2]:.3f})")
    #                     psim.TextUnformatted(f"Motion Limit: ({lim[0]:.3f}, {lim[1]:.3f})")
    #
    #                 elif best_type == "revolute":
    #                     a_ = params["axis"]
    #                     o_ = params["origin"]
    #                     lim = params["motion_limit"]
    #                     psim.TextUnformatted(f"Axis: ({a_[0]:.3f}, {a_[1]:.3f}, {a_[2]:.3f})")
    #                     psim.TextUnformatted(f"Origin: ({o_[0]:.3f}, {o_[1]:.3f}, {o_[2]:.3f})")
    #                     psim.TextUnformatted(f"Motion Limit: ({lim[0]:.3f} rad, {lim[1]:.3f} rad)")
    #
    #                 psim.TreePop()
    #
    #     # Help and shortcuts
    #     if psim.CollapsingHeader("Help & Shortcuts"):
    #         psim.TextUnformatted("Space: Play/Pause")
    #         psim.TextUnformatted("Left/Right Arrows: Previous/Next frame")
    #         psim.TextUnformatted("Up/Down Arrows: Next/Previous joint type")
    #         psim.TextUnformatted("Home/End: First/Last frame")
    #         psim.TextUnformatted("J: Toggle joint visualization")
    #         psim.TextUnformatted("T: Toggle trail visualization")
    #         psim.TextUnformatted("S: Take screenshot")
    #
    #     # Handle keyboard shortcuts
    #     try:
    #         # Try to detect key presses in different ways
    #         key_space = False
    #         key_right = False
    #         key_left = False
    #         key_up = False
    #         key_down = False
    #         key_home = False
    #         key_end = False
    #         key_j = False
    #         key_t = False
    #         key_s = False
    #
    #         # Method 1: Using ImGuiKey constants (newer versions)
    #         if hasattr(psim, 'ImGuiKey_Space'):
    #             try:
    #                 key_space = ps.is_key_pressed(psim.ImGuiKey_Space)
    #                 key_right = ps.is_key_pressed(psim.ImGuiKey_RightArrow)
    #                 key_left = ps.is_key_pressed(psim.ImGuiKey_LeftArrow)
    #                 key_up = ps.is_key_pressed(psim.ImGuiKey_UpArrow)
    #                 key_down = ps.is_key_pressed(psim.ImGuiKey_DownArrow)
    #                 key_home = ps.is_key_pressed(psim.ImGuiKey_Home)
    #                 key_end = ps.is_key_pressed(psim.ImGuiKey_End)
    #             except Exception:
    #                 pass
    #
    #         # Method 2: Using character keys (older versions)
    #         try:
    #             if not key_space:
    #                 key_space = ps.is_key_pressed(' ')
    #             if not key_right:
    #                 key_right = ps.is_key_pressed('right')
    #             if not key_left:
    #                 key_left = ps.is_key_pressed('left')
    #             if not key_up:
    #                 key_up = ps.is_key_pressed('up')
    #             if not key_down:
    #                 key_down = ps.is_key_pressed('down')
    #             if not key_home:
    #                 key_home = ps.is_key_pressed('home')
    #             if not key_end:
    #                 key_end = ps.is_key_pressed('end')
    #             key_j = ps.is_key_pressed('j')
    #             key_t = ps.is_key_pressed('t')
    #             key_s = ps.is_key_pressed('s')
    #         except Exception:
    #             pass
    #
    #         # Handle the detected keys
    #         if key_space:
    #             self._toggle_play()
    #
    #         if key_right:
    #             self._advance_frame()
    #
    #         if key_left:
    #             self._reverse_frame()
    #
    #         if key_up:
    #             new_idx = (self.current_joint_idx + 1) % len(self.joint_types)
    #             self._switch_joint_type(new_idx)
    #
    #         if key_down:
    #             new_idx = (self.current_joint_idx - 1) % len(self.joint_types)
    #             self._switch_joint_type(new_idx)
    #
    #         if key_home:
    #             self.current_frame_idx = 0
    #             self._update_visualization()
    #
    #         if key_end:
    #             joint_type = self.joint_types[self.current_joint_idx]
    #             sequence = self.sequences[joint_type]
    #             self.current_frame_idx = len(sequence) - 1
    #             self._update_visualization()
    #
    #         if key_j:
    #             self.show_joint_vis = not self.show_joint_vis
    #             self._update_visualization()
    #
    #         if key_t:
    #             self.show_previous_frames = not self.show_previous_frames
    #             self._update_visualization()
    #
    #         if key_s:
    #             self._export_current_frame()
    #     except Exception:
    #         # If keyboard handling fails, continue without it
    #         pass
    #
    #     # Advance frame if playing
    #     if self.playing:
    #         current_time = time.time()
    #         # Only advance frame if enough time has passed since last frame
    #         if current_time - self.last_frame_time > self.frame_delay / max(1, self.play_speed):
    #             self._advance_frame()
    #             self.last_frame_time = current_time
    def polyscope_callback(self):
        """最基础的Polyscope UI回调，兼容非常旧的版本"""
        # 关节类型选择器
        psim.TextUnformatted("Joint Type:")

        # 创建水平布局的关节类型按钮
        for i, joint_type in enumerate(self.joint_types):
            if psim.Button(joint_type.capitalize()):
                self._switch_joint_type(i)

            # 如果不是一行的最后一个按钮，添加水平间距
            if i < len(self.joint_types) - 1:
                psim.SameLine()

        psim.Separator()

        # 当前关节信息
        joint_type = self.joint_types[self.current_joint_idx]
        sequence = self.sequences[joint_type]
        total_frames = len(sequence)

        # 帧计数器和滑块控制
        psim.TextUnformatted(f"Frame: {self.current_frame_idx + 1} / {total_frames}")

        # 帧滑块
        changed, new_frame = psim.SliderInt("##FrameSlider", self.current_frame_idx, 0, total_frames - 1)
        if changed:
            self.current_frame_idx = new_frame
            self._update_visualization()

        # 播放控制
        if psim.Button("First"):
            self.current_frame_idx = 0
            self._update_visualization()

        psim.SameLine()

        if psim.Button("<<"):
            self.current_frame_idx = max(0, self.current_frame_idx - 10)
            self._update_visualization()

        psim.SameLine()

        if psim.Button("<"):
            self._reverse_frame()

        psim.SameLine()

        # 播放/暂停按钮
        play_label = "Pause" if self.playing else "Play"
        if psim.Button(play_label):
            self.playing = not self.playing

        psim.SameLine()

        if psim.Button(">"):
            self._advance_frame()

        psim.SameLine()

        if psim.Button(">>"):
            self.current_frame_idx = min(total_frames - 1, self.current_frame_idx + 10)
            self._update_visualization()

        psim.SameLine()

        if psim.Button("Last"):
            self.current_frame_idx = total_frames - 1
            self._update_visualization()

        # 截图按钮
        if psim.Button("Screenshot"):
            self._export_current_frame()

        psim.Separator()

        # 播放选项
        psim.TextUnformatted("Playback Options:")

        # 播放速度滑块
        changed, new_speed = psim.SliderInt("Play Speed", self.play_speed, 1, 10)
        if changed:
            self.play_speed = new_speed

        # 循环播放复选框
        changed, new_loop = psim.Checkbox("Loop Playback", self.loop_playback)
        if changed:
            self.loop_playback = new_loop

        # 显示关节可视化复选框
        changed, new_joint_vis = psim.Checkbox("Show Joint Visualization", self.show_joint_vis)
        if changed:
            self.show_joint_vis = new_joint_vis
            self._update_visualization()

        # 显示运动轨迹复选框
        changed, new_prev_frames = psim.Checkbox("Show Motion Trails", self.show_previous_frames)
        if changed:
            self.show_previous_frames = new_prev_frames
            self._update_visualization()

        if self.show_previous_frames:
            changed, new_num_frames = psim.SliderInt("Trail Frames", self.num_previous_frames, 1, 10)
            if changed:
                self.num_previous_frames = new_num_frames
                self._update_visualization()

        psim.Separator()

        # 分析结果
        psim.TextUnformatted("Analysis Results:")
        param_dict, best_type, scores_info = self.results[joint_type]

        psim.TextUnformatted(f"Ground Truth: {joint_type}")
        psim.TextUnformatted(f"Estimated: {best_type}")

        if scores_info is not None:
            # 基本分数
            psim.TextUnformatted("Basic Scores:")
            basic_scores = scores_info["basic_score_avg"]
            psim.TextUnformatted(f"  Collinearity: {basic_scores['col_mean']:.3f}")
            psim.TextUnformatted(f"  Coplanarity: {basic_scores['cop_mean']:.3f}")
            psim.TextUnformatted(f"  Radius Consistency: {basic_scores['rad_mean']:.3f}")
            psim.TextUnformatted(f"  Zero Pitch: {basic_scores['zp_mean']:.3f}")

            # 关节概率
            psim.TextUnformatted("Joint Probabilities:")
            joint_probs = scores_info["joint_probs"]
            for jt, prob in sorted(joint_probs.items(), key=lambda x: x[1], reverse=True):
                psim.TextUnformatted(f"  {jt.capitalize()}: {prob:.3f}")

        # 前进帧如果正在播放
        if self.playing:
            current_time = time.time()
            if current_time - self.last_frame_time > self.frame_delay / max(1, self.play_speed):
                self._advance_frame()
                self.last_frame_time = current_time

    def show(self):
        """Show the visualization and start the main loop."""
        # Set up callback
        ps.set_user_callback(self.polyscope_callback)

        # Show polyscope window
        ps.show()

        # Ensure playback thread is stopped when window is closed
        self._stop_playback_thread()


def run_animated_visualization(sequences, results):
    """
    Run animated visualization of point cloud sequences.

    Args:
        sequences (dict): Dictionary mapping joint types to point cloud sequences
        results (dict): Dictionary mapping joint types to analysis results
    """
    if not POLYSCOPE_AVAILABLE:
        print("Polyscope not available. Skipping 3D visualization.")
        return

    # Create animator
    animator = AnimatedVisualizer(sequences, results)

    # Show visualization
    animator.show()


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
            print("\nStarting animated visualization...")
            print("Use keyboard shortcuts for navigation:")
            print("  Space: Play/Pause")
            print("  Left/Right Arrows: Previous/Next frame")
            print("  Up/Down Arrows: Next/Previous joint type")
            print("  J: Toggle joint visualization")
            print("  T: Toggle trail visualization")
            run_animated_visualization(sequences, results)

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()