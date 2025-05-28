import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from io import BytesIO
import os
from datetime import datetime
from robot_utils.viz.polyscope import PolyscopeUtils, ps, psim, register_point_cloud, draw_frame_3d

# Import joint analysis project modules
from joint_analysis.joint_analysis.core.joint_estimation import compute_joint_info_all_types
from joint_analysis.joint_analysis.viz.polyscope_viz import PolyscopeVisualizer


class EnhancedViz:
    def __init__(self, file_paths=None):
        # Initialize default file paths list if none provided
        if file_paths is None:
            self.file_paths = ["./demo_data/s1_gasstove_part2_1110_1170.npy"]
        else:
            self.file_paths = file_paths

        # Create Polyscope utils instance
        self.pu = PolyscopeUtils()
        # Initialize current time frame index and change flag
        self.t, self.t_changed = 0, False
        # Initialize current point index and change flag
        self.idx_point, self.idx_point_changed = 0, False
        # Current selected dataset index
        self.current_dataset = 0
        self.noise_sigma = 0
        # Store multiple datasets dictionary
        self.datasets = {}
        # Load all files
        for i, file_path in enumerate(self.file_paths):
            # Give each dataset a name identifier
            dataset_name = f"dataset_{i}"
            # Load data
            data = np.load(file_path)
            # Create dataset name from filename for possible relative paths
            display_name = os.path.basename(file_path).split('.')[0]
            if self.noise_sigma > 0:
                data += np.random.randn(*data.shape) * self.noise_sigma
            # Store dataset information
            self.datasets[dataset_name] = {
                "path": file_path,
                "display_name": display_name,
                "data": data,
                "T": data.shape[0],  # Number of time frames
                "N": data.shape[1],  # Number of points per frame
                "visible": True  # Whether to display
            }

            # Apply Savitzky-Golay filter to smooth data
            self.datasets[dataset_name]["data_filter"] = savgol_filter(
                x=data, window_length=21, polyorder=2, deriv=0, axis=0, delta=0.1
            )

            # Use Savitzky-Golay filter to calculate velocity (first derivative)
            self.datasets[dataset_name]["dv_filter"] = savgol_filter(
                x=data, window_length=21, polyorder=2, deriv=1, axis=0, delta=0.1
            )

        # Use first dataset as current dataset
        dataset_keys = list(self.datasets.keys())
        if dataset_keys:
            self.current_dataset_key = dataset_keys[0]
            current_data = self.datasets[self.current_dataset_key]
            # Get dimensions of current dataset
            self.T = current_data["T"]
            self.N = current_data["N"]
            # Current active data
            self.d = current_data["data"]
            self.d_filter = current_data["data_filter"]
            self.dv_filter = current_data["dv_filter"]
        else:
            print("No datasets loaded.")
            return

        # Average time step
        self.dt_mean = 0.1
        # Number of neighbors for SVD computation
        self.num_neighbors = 50

        # Joint analysis parameters
        self.col_sigma = 0.2
        self.col_order = 4.0
        self.cop_sigma = 0.2
        self.cop_order = 4.0
        self.rad_sigma = 0.2
        self.rad_order = 4.0
        self.zp_sigma = 0.2
        self.zp_order = 4.0
        self.use_savgol = True
        self.savgol_window = 11
        self.savgol_poly = 2

        # Create folder for saving images
        self.output_dir = "visualization_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Calculate angular velocity
        for dataset_key in self.datasets:
            dataset = self.datasets[dataset_key]
            angular_velocity_raw, angular_velocity_filtered = self.calculate_angular_velocity(
                dataset["data_filter"], dataset["N"]
            )
            dataset["angular_velocity_raw"] = angular_velocity_raw
            dataset["angular_velocity_filtered"] = angular_velocity_filtered

            # Perform joint analysis
            joint_params, best_joint, info_dict = self.perform_joint_analysis(dataset["data_filter"])
            dataset["joint_params"] = joint_params
            dataset["best_joint"] = best_joint
            dataset["joint_info"] = info_dict

        # Current dataset's angular velocity
        self.angular_velocity_raw = self.datasets[self.current_dataset_key]["angular_velocity_raw"]
        self.angular_velocity_filtered = self.datasets[self.current_dataset_key]["angular_velocity_filtered"]

        # Current dataset's joint information
        self.current_joint_params = self.datasets[self.current_dataset_key]["joint_params"]
        self.current_best_joint = self.datasets[self.current_dataset_key]["best_joint"]
        self.current_joint_info = self.datasets[self.current_dataset_key]["joint_info"]

        # Draw origin coordinate frame
        draw_frame_3d(np.zeros(6), label="origin", scale=0.1)

        # Register initial point cloud for all datasets and display
        for dataset_key, dataset in self.datasets.items():
            if dataset["visible"]:
                register_point_cloud(
                    dataset["display_name"],
                    dataset["data_filter"][self.t],
                    radius=0.01,
                    enabled=True
                )

        # Reset view bounds to include all point clouds
        visible_point_clouds = [dataset["data_filter"][self.t] for dataset in self.datasets.values()
                                if dataset["visible"]]
        if visible_point_clouds:
            self.pu.reset_bbox_from_pcl_list(visible_point_clouds)

        # Visualize joint parameters for current dataset
        self.visualize_joint_parameters()

        # Set user interaction callback function
        ps.set_user_callback(self.callback)
        # Show Polyscope interface
        ps.show()

    def perform_joint_analysis(self, point_history):
        """Perform joint analysis on point history data"""
        return compute_joint_info_all_types(
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
        # Print joint analysis results to terminal with 16 decimal places
        print("\n" + "=" * 80)
        print(f"Joint Type: {best_joint}")
        print("=" * 80)

        # Print basic scores
        if info_dict and "basic_score_avg" in info_dict:
            basic_scores = info_dict["basic_score_avg"]
            print("\nBasic Scores:")
            print(f"Collinearity Score: {basic_scores.get('col_mean', 0.0):.16f}")
            print(f"Coplanarity Score: {basic_scores.get('cop_mean', 0.0):.16f}")
            print(f"Radius Consistency Score: {basic_scores.get('rad_mean', 0.0):.16f}")
            print(f"Zero Pitch Score: {basic_scores.get('zp_mean', 0.0):.16f}")

        # Print joint probabilities
        if info_dict and "joint_probs" in info_dict:
            joint_probs = info_dict["joint_probs"]
            print("\nJoint Probabilities:")
            for joint_type, prob in joint_probs.items():
                print(f"{joint_type.capitalize()}: {prob:.16f}")

        # Print joint parameters
        if best_joint in joint_params:
            params = joint_params[best_joint]
            print(f"\n{best_joint.capitalize()} Joint Parameters:")

            if best_joint == "planar":
                normal = params.get("normal", [0, 0, 0])
                motion_limit = params.get("motion_limit", (0, 0))
                print(f"Normal Vector: [{normal[0]:.16f}, {normal[1]:.16f}, {normal[2]:.16f}]")
                print(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f})")

            elif best_joint == "ball":
                center = params.get("center", [0, 0, 0])
                radius = params.get("radius", 0)
                motion_limit = params.get("motion_limit", (0, 0, 0))
                print(f"Center: [{center[0]:.16f}, {center[1]:.16f}, {center[2]:.16f}]")
                print(f"Radius: {radius:.16f}")
                print(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}, {motion_limit[2]:.16f}) rad")

            elif best_joint == "screw":
                axis = params.get("axis", [0, 0, 0])
                origin = params.get("origin", [0, 0, 0])
                pitch = params.get("pitch", 0)
                motion_limit = params.get("motion_limit", (0, 0))
                print(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]")
                print(f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]")
                print(f"Pitch: {pitch:.16f}")
                print(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) rad")

            elif best_joint == "prismatic":
                axis = params.get("axis", [0, 0, 0])
                origin = params.get("origin", [0, 0, 0])
                motion_limit = params.get("motion_limit", (0, 0))
                print(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]")
                print(f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]")
                print(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) m")

            elif best_joint == "revolute":
                axis = params.get("axis", [0, 0, 0])
                origin = params.get("origin", [0, 0, 0])
                motion_limit = params.get("motion_limit", (0, 0))
                print(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]")
                print(f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]")
                print(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) rad")
                print(f"Motion Range: {np.degrees(motion_limit[1] - motion_limit[0]):.16f}°")

        print("=" * 80 + "\n")

        return joint_params, best_joint, info_dict

    def visualize_joint_parameters(self):
        """Visualize estimated joint parameters in Polyscope"""
        # Remove any existing joint visualizations
        self.remove_joint_visualization()

        joint_type = self.current_best_joint
        joint_params = None

        if joint_type in self.current_joint_params:
            joint_params = self.current_joint_params[joint_type]
            self.show_joint_visualization(joint_type, joint_params)

    def remove_joint_visualization(self):
        """Remove all joint visualization elements."""
        # Remove curve networks
        for name in [
            "Planar Normal", "Ball Center", "Screw Axis", "Screw Axis Pitch",
            "Prismatic Axis", "Revolute Axis", "Revolute Origin", "Planar Axes"
        ]:
            if ps.has_curve_network(name):
                ps.remove_curve_network(name)

        # Remove point clouds used for visualization
        for name in ["BallCenterPC"]:
            if ps.has_point_cloud(name):
                ps.remove_point_cloud(name)

    def show_joint_visualization(self, joint_type, joint_params):
        """Show visualization for a specific joint type."""
        if joint_type == "planar":
            # Extract parameters
            n_np = joint_params.get("normal", np.array([0., 0., 1.]))

            # Visualize normal
            seg_nodes = np.array([[0, 0, 0], n_np])
            seg_edges = np.array([[0, 1]])
            name = "Planar Normal"
            planarnet = ps.register_curve_network(name, seg_nodes, seg_edges)
            planarnet.set_color((1.0, 0.0, 0.0))
            planarnet.set_radius(0.02)

        elif joint_type == "ball":
            # Extract parameters
            center_np = joint_params.get("center", np.array([0., 0., 0.]))

            # Visualize center
            name = "BallCenterPC"
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

            name = "Ball Center"
            axisviz = ps.register_curve_network(name, seg_nodes, seg_edges)
            axisviz.set_radius(0.02)
            axisviz.set_color((1., 0., 1.))

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

            name = "Screw Axis"
            scv = ps.register_curve_network(name, seg_nodes, seg_edges)
            scv.set_radius(0.02)
            scv.set_color((0., 0., 1.0))

            # Visualize pitch using an arrow
            pitch_arrow_start = origin_np + axis_np * 0.6

            # Find a perpendicular vector
            perp_vec = np.array([1, 0, 0])
            if np.abs(np.dot(axis_np, perp_vec)) > 0.9:
                perp_vec = np.array([0, 1, 0])

            perp_vec = perp_vec - np.dot(perp_vec, axis_np) * axis_np
            perp_vec = perp_vec / (np.linalg.norm(perp_vec) + 1e-9)

            pitch_arrow_end = pitch_arrow_start + 0.2 * pitch_ * perp_vec

            seg_nodes2 = np.array([pitch_arrow_start, pitch_arrow_end])
            seg_edges2 = np.array([[0, 1]])

            name = "Screw Axis Pitch"
            pitch_net = ps.register_curve_network(name, seg_nodes2, seg_edges2)
            pitch_net.set_color((1., 0., 0.))
            pitch_net.set_radius(0.02)

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

            name = "Prismatic Axis"
            pcv = ps.register_curve_network(name, seg_nodes, seg_edges)
            pcv.set_radius(0.01)
            pcv.set_color((0., 1., 1.))

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

            name = "Revolute Axis"
            rvnet = ps.register_curve_network(name, seg_nodes, seg_edges)
            rvnet.set_radius(0.01)
            rvnet.set_color((1., 1., 0.))

            # Visualize origin
            seg_nodes2 = np.array([origin_np, origin_np + 1e-5 * axis_np])
            seg_edges2 = np.array([[0, 1]])

            name = "Revolute Origin"
            origin_net = ps.register_curve_network(name, seg_nodes2, seg_edges2)
            origin_net.set_radius(0.015)
            origin_net.set_color((1., 0., 0.))

    def find_neighbors(self, points, num_neighbors):
        """Find neighbors for each point"""
        # Calculate distance matrix
        N = points.shape[0]
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            dist_matrix[i] = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))

        # Find closest points (excluding self)
        neighbors = np.zeros((N, num_neighbors), dtype=int)
        for i in range(N):
            indices = np.argsort(dist_matrix[i])[1:num_neighbors + 1]
            neighbors[i] = indices

        return neighbors

    def compute_rotation_matrix(self, src_points, dst_points):
        """Compute rotation matrix between two point sets using SVD"""
        # Center points
        src_center = np.mean(src_points, axis=0)
        dst_center = np.mean(dst_points, axis=0)

        src_centered = src_points - src_center
        dst_centered = dst_points - dst_center

        # Compute covariance matrix
        H = np.dot(src_centered.T, dst_centered)

        # SVD decomposition
        U, _, Vt = np.linalg.svd(H)

        # Construct rotation matrix
        R = np.dot(Vt.T, U.T)

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        return R, src_center, dst_center

    def rotation_matrix_to_angular_velocity(self, R, dt):
        """Extract angular velocity from rotation matrix"""
        # Ensure R is a valid rotation matrix
        U, _, Vt = np.linalg.svd(R)
        R = np.dot(U, Vt)

        # Compute rotation angle
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # If angle is too small, return zero vector
        if abs(theta) < 1e-6:
            return np.zeros(3)

        # Calculate rotation axis
        sin_theta = np.sin(theta)
        if abs(sin_theta) < 1e-6:
            return np.zeros(3)

        # Extract angular velocity vector from skew-symmetric part
        W = (R - R.T) / (2 * sin_theta)
        omega = np.zeros(3)
        omega[0] = W[2, 1]
        omega[1] = W[0, 2]
        omega[2] = W[1, 0]

        # Angular velocity = axis * angle / time
        omega = omega * theta / dt

        return omega

    def calculate_angular_velocity(self, d_filter, N):
        """Compute angular velocity using SVD on filtered point cloud data"""
        T = d_filter.shape[0]

        # Initialize arrays
        angular_velocity = np.zeros((T - 1, N, 3))

        # For each pair of consecutive frames
        for t in range(T - 1):
            # Current and next frame points
            current_points = d_filter[t]
            next_points = d_filter[t + 1]

            # Find neighbors for all points
            neighbors = self.find_neighbors(current_points, self.num_neighbors)

            # Compute angular velocity for each point
            for i in range(N):
                # Get current point and its neighbors
                src_neighborhood = current_points[neighbors[i]]
                dst_neighborhood = next_points[neighbors[i]]

                # Compute rotation matrix
                R, _, _ = self.compute_rotation_matrix(src_neighborhood, dst_neighborhood)

                # Extract angular velocity
                omega = self.rotation_matrix_to_angular_velocity(R, self.dt_mean)

                # Store result
                angular_velocity[t, i] = omega

        # Filter angular velocity
        angular_velocity_filtered = np.zeros_like(angular_velocity)

        if T - 1 >= 5:
            for i in range(N):
                for dim in range(3):
                    angular_velocity_filtered[:, i, dim] = savgol_filter(
                        angular_velocity[:, i, dim],
                        window_length=min(21, T - 1),  # Ensure window length doesn't exceed data length
                        polyorder=min(2, (T - 1) // 2)  # Ensure polynomial order is appropriate
                    )
        else:
            angular_velocity_filtered = angular_velocity.copy()

        return angular_velocity, angular_velocity_filtered

    def switch_dataset(self, new_dataset_key):
        """Switch current active dataset"""
        if new_dataset_key in self.datasets:
            self.current_dataset_key = new_dataset_key
            dataset = self.datasets[new_dataset_key]

            # Update current data and time/point limits
            self.d = dataset["data"]
            self.d_filter = dataset["data_filter"]
            self.dv_filter = dataset["dv_filter"]
            self.angular_velocity_raw = dataset["angular_velocity_raw"]
            self.angular_velocity_filtered = dataset["angular_velocity_filtered"]

            # Update joint information
            self.current_joint_params = dataset["joint_params"]
            self.current_best_joint = dataset["best_joint"]
            self.current_joint_info = dataset["joint_info"]

            # Update T and N values
            self.T = dataset["T"]
            self.N = dataset["N"]

            # Ensure current t and idx_point are within valid range
            self.t = min(self.t, self.T - 1)
            self.idx_point = min(self.idx_point, self.N - 1)

            # Update plot
            self.plot_image()

            # Update joint visualization
            self.visualize_joint_parameters()

            return True
        return False

    def toggle_visibility(self, dataset_key):
        """Toggle dataset visibility"""
        if dataset_key in self.datasets:
            self.datasets[dataset_key]["visible"] = not self.datasets[dataset_key]["visible"]
            return True
        return False

    def plot_image(self):
        """Plot trajectory of current point"""
        dataset = self.datasets[self.current_dataset_key]

        # Create time axis array (seconds)
        t = np.arange(self.T) * self.dt_mean
        t_angular = np.arange(self.T - 1) * self.dt_mean

        # Create figure with five subplots
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), dpi=100)

        # First subplot: position data
        # Plot original data
        ax1.plot(t, self.d[:, self.idx_point, 0], "--", c="red", label="Original X")
        ax1.plot(t, self.d[:, self.idx_point, 1], "--", c="green", label="Original Y")
        ax1.plot(t, self.d[:, self.idx_point, 2], "--", c="blue", label="Original Z")

        # Plot filtered data
        ax1.plot(t, self.d_filter[:, self.idx_point, 0], "-", c="darkred", label="Filtered X")
        ax1.plot(t, self.d_filter[:, self.idx_point, 1], "-", c="darkgreen", label="Filtered Y")
        ax1.plot(t, self.d_filter[:, self.idx_point, 2], "-", c="darkblue", label="Filtered Z")

        # Set first subplot title and labels
        ax1.set_title(f"Position Trajectory of Point #{self.idx_point} - {dataset['display_name']}")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.set_xlim(0, max(t))
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        # Second subplot: velocity data
        ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity")

        # Set second subplot title and labels
        ax2.set_title(f"Linear Velocity of Point #{self.idx_point} - {dataset['display_name']}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlim(0, max(t))
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        # Third subplot: Angular velocity X component
        ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red", label="Raw ωx")
        ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred", label="Filtered ωx")

        ax3.set_title(f"Angular Velocity X Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angular Velocity (rad/s)")
        ax3.set_xlim(0, max(t))
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right')

        # Fourth subplot: Angular velocity Y component
        ax4.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green", label="Raw ωy")
        ax4.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
                 label="Filtered ωy")

        ax4.set_title(f"Angular Velocity Y Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angular Velocity (rad/s)")
        ax4.set_xlim(0, max(t))
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')

        # Fifth subplot: Angular velocity Z component
        ax5.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 2], "--", c="blue", label="Raw ωz")
        ax5.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 2], "-", c="darkblue",
                 label="Filtered ωz")

        ax5.set_title(f"Angular Velocity Z Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Angular Velocity (rad/s)")
        ax5.set_xlim(0, max(t))
        ax5.grid(True, linestyle='--', alpha=0.7)
        ax5.legend(loc='upper right')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Create binary buffer in memory
        buf = BytesIO()
        # Save plot as PNG format to buffer
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        # Load image from buffer and convert to RGB format
        image = Image.open(buf).convert('RGB')
        # Convert image to NumPy array
        rgb_array = np.array(image)
        # Add image to Polyscope interface and enable display
        ps.add_color_image_quantity("plot", rgb_array / 255.0, enabled=True)
        # Close figure to release resources
        plt.close(fig)

    def save_current_plot(self):
        """Save current point's trajectory plot to file"""
        dataset = self.datasets[self.current_dataset_key]

        # Create time axis array (seconds)
        t = np.arange(self.T) * self.dt_mean
        t_angular = np.arange(self.T - 1) * self.dt_mean

        # Create figure with five subplots
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), dpi=100)

        # Plot position data
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

        # Plot velocity data
        ax2.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red", label="X Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 1], "-", c="green", label="Y Velocity")
        ax2.plot(t, self.dv_filter[:, self.idx_point, 2], "-", c="blue", label="Z Velocity")
        ax2.set_title(f"Linear Velocity of Point #{self.idx_point} - {dataset['display_name']}")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.set_xlim(0, max(t))
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')

        # Plot angular velocity X component
        ax3.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 0], "--", c="red", label="Raw ωx")
        ax3.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 0], "-", c="darkred", label="Filtered ωx")
        ax3.set_title(f"Angular Velocity X Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angular Velocity (rad/s)")
        ax3.set_xlim(0, max(t))
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(loc='upper right')

        # Plot angular velocity Y component
        ax4.plot(t_angular, self.angular_velocity_raw[:, self.idx_point, 1], "--", c="green", label="Raw ωy")
        ax4.plot(t_angular, self.angular_velocity_filtered[:, self.idx_point, 1], "-", c="darkgreen",
                 label="Filtered ωy")
        ax4.set_title(f"Angular Velocity Y Component of Point #{self.idx_point} - {dataset['display_name']}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angular Velocity (rad/s)")
        ax4.set_xlim(0, max(t))
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')

        # Plot angular velocity Z component
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

        # Create filename including timestamp and point index
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{dataset['display_name']}_point_{self.idx_point}_t{self.t}_{timestamp}.png"

        # Save to file
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"Plot saved to: {filename}")
        return filename

    def callback(self):
        """Polyscope UI callback function"""
        # Display title text
        psim.Text("Point Cloud Joint Analysis")

        # Display current dataset info
        psim.Text(f"Active Dataset: {self.datasets[self.current_dataset_key]['display_name']}")

        # Display joint type information
        psim.Text(f"Detected Joint Type: {self.current_best_joint}")

        # Create time frame slider, returns change status and new value
        self.t_changed, self.t = psim.SliderInt("Time Frame", self.t, 0, self.T - 1)
        # Create point index slider, returns change status and new value
        self.idx_point_changed, self.idx_point = psim.SliderInt("Point Index", self.idx_point, 0, self.N - 1)
        # Number of neighbors slider for SVD
        changed_neighbors, self.num_neighbors = psim.SliderInt("Neighbors for SVD", self.num_neighbors, 5, 30)

        # Add sliders for joint analysis parameters
        if psim.TreeNode("Joint Analysis Parameters"):
            changed_col_sigma, self.col_sigma = psim.SliderFloat("Collinearity Sigma", self.col_sigma, 0.05, 0.5)
            changed_col_order, self.col_order = psim.SliderFloat("Collinearity Order", self.col_order, 1.0, 10.0)
            changed_cop_sigma, self.cop_sigma = psim.SliderFloat("Coplanarity Sigma", self.cop_sigma, 0.05, 0.5)
            changed_cop_order, self.cop_order = psim.SliderFloat("Coplanarity Order", self.cop_order, 1.0, 10.0)
            changed_rad_sigma, self.rad_sigma = psim.SliderFloat("Radius Sigma", self.rad_sigma, 0.05, 0.5)
            changed_rad_order, self.rad_order = psim.SliderFloat("Radius Order", self.rad_order, 1.0, 10.0)
            changed_zp_sigma, self.zp_sigma = psim.SliderFloat("Zero Pitch Sigma", self.zp_sigma, 0.05, 0.5)
            changed_zp_order, self.zp_order = psim.SliderFloat("Zero Pitch Order", self.zp_order, 1.0, 10.0)

            changed_savgol_window, self.savgol_window = psim.SliderInt("SG Window", self.savgol_window, 3, 31, 2)
            changed_savgol_poly, self.savgol_poly = psim.SliderInt("SG Poly Order", self.savgol_poly, 1, 5)
            changed_savgol, self.use_savgol = psim.Checkbox("Use SG Filter", self.use_savgol)

            # Check if any parameter changed
            params_changed = (changed_col_sigma or changed_col_order or
                              changed_cop_sigma or changed_cop_order or
                              changed_rad_sigma or changed_rad_order or
                              changed_zp_sigma or changed_zp_order or
                              changed_savgol_window or changed_savgol_poly or
                              changed_savgol)

            if params_changed:
                if psim.Button("Apply Parameter Changes"):
                    # Rerun joint analysis with new parameters
                    joint_params, best_joint, info_dict = self.perform_joint_analysis(
                        self.datasets[self.current_dataset_key]["data_filter"])
                    self.datasets[self.current_dataset_key]["joint_params"] = joint_params
                    self.datasets[self.current_dataset_key]["best_joint"] = best_joint
                    self.datasets[self.current_dataset_key]["joint_info"] = info_dict

                    self.current_joint_params = joint_params
                    self.current_best_joint = best_joint
                    self.current_joint_info = info_dict

                    # Update visualization
                    self.visualize_joint_parameters()

            psim.TreePop()

        # If time frame changes, update point cloud display
        if self.t_changed:
            for dataset_key, dataset in self.datasets.items():
                if dataset["visible"] and self.t < dataset["T"]:
                    register_point_cloud(
                        dataset["display_name"],
                        dataset["data_filter"][min(self.t, dataset["T"] - 1)],
                        radius=0.01,
                        enabled=True
                    )

        # If point index changes, redraw trajectory plot
        if self.idx_point_changed:
            self.plot_image()

        # Dataset selection section
        if psim.TreeNode("Dataset Selection"):
            # Display dataset list, allow selection and toggle visibility
            for dataset_key, dataset in self.datasets.items():
                if psim.Button(f"Select {dataset['display_name']}"):
                    self.switch_dataset(dataset_key)

                psim.SameLine()
                vis_text = "Visible" if dataset["visible"] else "Hidden"
                if psim.Button(f"{vis_text}###{dataset_key}"):
                    self.toggle_visibility(dataset_key)
                    # Update point cloud display
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

        # Add save image button
        if psim.Button("Save Current Plot"):
            saved_path = self.save_current_plot()
            psim.Text(f"Saved to: {saved_path}")

        # Add recalculate button
        if changed_neighbors or psim.Button("Reanalyze Joint"):
            # Recalculate angular velocity
            for dataset_key, dataset in self.datasets.items():
                angular_velocity_raw, angular_velocity_filtered = self.calculate_angular_velocity(
                    dataset["data_filter"], dataset["N"]
                )
                dataset["angular_velocity_raw"] = angular_velocity_raw
                dataset["angular_velocity_filtered"] = angular_velocity_filtered

                # Perform joint analysis
                joint_params, best_joint, info_dict = self.perform_joint_analysis(dataset["data_filter"])
                dataset["joint_params"] = joint_params
                dataset["best_joint"] = best_joint
                dataset["joint_info"] = info_dict

            # Update current dataset's values
            self.angular_velocity_raw = self.datasets[self.current_dataset_key]["angular_velocity_raw"]
            self.angular_velocity_filtered = self.datasets[self.current_dataset_key]["angular_velocity_filtered"]
            self.current_joint_params = self.datasets[self.current_dataset_key]["joint_params"]
            self.current_best_joint = self.datasets[self.current_dataset_key]["best_joint"]
            self.current_joint_info = self.datasets[self.current_dataset_key]["joint_info"]

            # Update plot and visualization
            self.plot_image()
            self.visualize_joint_parameters()

            psim.Text("Joint analysis recalculated")

        # Display current data and joint information
        if psim.TreeNode("Data Information"):
            dataset = self.datasets[self.current_dataset_key]
            psim.Text(f"Current dataset: {dataset['display_name']}")
            psim.Text(f"Data shape: {dataset['data'].shape}")
            psim.Text(f"Number of time frames: {dataset['T']}")
            psim.Text(f"Number of points: {dataset['N']}")

            # Display current point's position, velocity and angular velocity
            if self.t < self.T:
                pos = self.d_filter[self.t, self.idx_point]
                psim.Text(f"Position (m): X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")

                if self.t < self.T - 1:
                    # Linear velocity
                    vel = self.dv_filter[self.t, self.idx_point]
                    vel_mag = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
                    psim.Text(f"Linear velocity (m/s): X={vel[0]:.3f}, Y={vel[1]:.3f}, Z={vel[2]:.3f}")
                    psim.Text(f"Linear velocity magnitude: {vel_mag:.3f} m/s")

                    # Angular velocity
                    ang_vel_idx = min(self.t, len(self.angular_velocity_filtered) - 1)
                    if ang_vel_idx >= 0 and self.idx_point < self.angular_velocity_filtered.shape[1]:
                        ang_vel = self.angular_velocity_filtered[ang_vel_idx, self.idx_point]
                        ang_vel_mag = np.sqrt(ang_vel[0] ** 2 + ang_vel[1] ** 2 + ang_vel[2] ** 2)
                        psim.Text(
                            f"Angular velocity (rad/s): ωx={ang_vel[0]:.3f}, ωy={ang_vel[1]:.3f}, ωz={ang_vel[2]:.3f}")
                        psim.Text(
                            f"Angular velocity magnitude: {ang_vel_mag:.3f} rad/s ({np.degrees(ang_vel_mag):.2f}°/s)")
                # Add printing of analysis results for current active dataset
                print("\n" + "=" * 80)
                print(f"Dataset: {self.datasets[self.current_dataset_key]['display_name']}")
                print(f"Joint Type: {self.current_best_joint}")
                print("=" * 80)

                # Print basic scores
                if self.current_joint_info and "basic_score_avg" in self.current_joint_info:
                    basic_scores = self.current_joint_info["basic_score_avg"]
                    print("\nBasic Scores:")
                    print(f"Collinearity Score: {basic_scores.get('col_mean', 0.0):.16f}")
                    print(f"Coplanarity Score: {basic_scores.get('cop_mean', 0.0):.16f}")
                    print(f"Radius Consistency Score: {basic_scores.get('rad_mean', 0.0):.16f}")
                    print(f"Zero Pitch Score: {basic_scores.get('zp_mean', 0.0):.16f}")

                # Print joint probabilities
                if self.current_joint_info and "joint_probs" in self.current_joint_info:
                    joint_probs = self.current_joint_info["joint_probs"]
                    print("\nJoint Probabilities:")
                    for joint_type, prob in joint_probs.items():
                        print(f"{joint_type.capitalize()}: {prob:.16f}")

                # Print joint parameters
                if self.current_best_joint in self.current_joint_params:
                    params = self.current_joint_params[self.current_best_joint]
                    print(f"\n{self.current_best_joint.capitalize()} Joint Parameters:")

                    if self.current_best_joint == "planar":
                        normal = params.get("normal", [0, 0, 0])
                        motion_limit = params.get("motion_limit", (0, 0))
                        print(f"Normal Vector: [{normal[0]:.16f}, {normal[1]:.16f}, {normal[2]:.16f}]")
                        print(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f})")

                    elif self.current_best_joint == "ball":
                        center = params.get("center", [0, 0, 0])
                        radius = params.get("radius", 0)
                        motion_limit = params.get("motion_limit", (0, 0, 0))
                        print(f"Center: [{center[0]:.16f}, {center[1]:.16f}, {center[2]:.16f}]")
                        print(f"Radius: {radius:.16f}")
                        print(
                            f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}, {motion_limit[2]:.16f}) rad")

                    elif self.current_best_joint == "screw":
                        axis = params.get("axis", [0, 0, 0])
                        origin = params.get("origin", [0, 0, 0])
                        pitch = params.get("pitch", 0)
                        motion_limit = params.get("motion_limit", (0, 0))
                        print(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]")
                        print(f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]")
                        print(f"Pitch: {pitch:.16f}")
                        print(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) rad")

                    elif self.current_best_joint == "prismatic":
                        axis = params.get("axis", [0, 0, 0])
                        origin = params.get("origin", [0, 0, 0])
                        motion_limit = params.get("motion_limit", (0, 0))
                        print(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]")
                        print(f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]")
                        print(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) m")

                    elif self.current_best_joint == "revolute":
                        axis = params.get("axis", [0, 0, 0])
                        origin = params.get("origin", [0, 0, 0])
                        motion_limit = params.get("motion_limit", (0, 0))
                        print(f"Axis: [{axis[0]:.16f}, {axis[1]:.16f}, {axis[2]:.16f}]")
                        print(f"Origin: [{origin[0]:.16f}, {origin[1]:.16f}, {origin[2]:.16f}]")
                        print(f"Motion Limit: ({motion_limit[0]:.16f}, {motion_limit[1]:.16f}) rad")
                        print(f"Motion Range: {np.degrees(motion_limit[1] - motion_limit[0]):.16f}°")

                print("=" * 80 + "\n")
            psim.TreePop()

        # Display joint information
        if psim.TreeNode("Joint Information"):
            # Basic scores
            if self.current_joint_info and "basic_score_avg" in self.current_joint_info:
                psim.Text("Basic Scores:")
                basic_scores = self.current_joint_info["basic_score_avg"]
                psim.Text(f"Collinearity: {basic_scores.get('col_mean', 0.0):.3f}")
                psim.Text(f"Coplanarity: {basic_scores.get('cop_mean', 0.0):.3f}")
                psim.Text(f"Radius Consistency: {basic_scores.get('rad_mean', 0.0):.3f}")
                psim.Text(f"Zero Pitch: {basic_scores.get('zp_mean', 0.0):.3f}")

                # Joint probabilities
                psim.Text("\nJoint Probabilities:")
                joint_probs = self.current_joint_info.get("joint_probs", {})
                for joint_type, prob in joint_probs.items():
                    psim.Text(f"{joint_type.capitalize()}: {prob:.3f}")

                # Joint parameters
                if self.current_best_joint in self.current_joint_params:
                    psim.Text(f"\n{self.current_best_joint.capitalize()} Joint Parameters:")
                    params = self.current_joint_params[self.current_best_joint]

                    if self.current_best_joint == "planar":
                        normal = params.get("normal", [0, 0, 0])
                        motion_limit = params.get("motion_limit", (0, 0))
                        psim.Text(f"Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
                        psim.Text(f"Motion Limit: ({motion_limit[0]:.3f}, {motion_limit[1]:.3f})")

                    elif self.current_best_joint == "ball":
                        center = params.get("center", [0, 0, 0])
                        radius = params.get("radius", 0)
                        motion_limit = params.get("motion_limit", (0, 0, 0))
                        psim.Text(f"Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
                        psim.Text(f"Radius: {radius:.3f}")
                        psim.Text(
                            f"Motion Limit: ({motion_limit[0]:.3f}, {motion_limit[1]:.3f}, {motion_limit[2]:.3f}) rad")

                    elif self.current_best_joint == "screw":
                        axis = params.get("axis", [0, 0, 0])
                        origin = params.get("origin", [0, 0, 0])
                        pitch = params.get("pitch", 0)
                        motion_limit = params.get("motion_limit", (0, 0))
                        psim.Text(f"Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
                        psim.Text(f"Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
                        psim.Text(f"Pitch: {pitch:.3f}")
                        psim.Text(f"Motion Limit: ({motion_limit[0]:.3f}, {motion_limit[1]:.3f}) rad")

                    elif self.current_best_joint == "prismatic":
                        axis = params.get("axis", [0, 0, 0])
                        origin = params.get("origin", [0, 0, 0])
                        motion_limit = params.get("motion_limit", (0, 0))
                        psim.Text(f"Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
                        psim.Text(f"Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
                        psim.Text(f"Motion Limit: ({motion_limit[0]:.3f}, {motion_limit[1]:.3f}) m")

                    elif self.current_best_joint == "revolute":
                        axis = params.get("axis", [0, 0, 0])
                        origin = params.get("origin", [0, 0, 0])
                        motion_limit = params.get("motion_limit", (0, 0))
                        psim.Text(f"Axis: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
                        psim.Text(f"Origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f}]")
                        psim.Text(f"Motion Limit: ({motion_limit[0]:.3f}, {motion_limit[1]:.3f}) rad")
                        psim.Text(f"Motion Range: {np.degrees(motion_limit[1] - motion_limit[0]):.2f}°")

            psim.TreePop()


# Program entry point
if __name__ == "__main__":
    # You can specify multiple data file paths here
    file_paths = [

#open refrigerator
        # "./demo_data/s1_refrigerator_part2_3180_3240.npy",
        # "./demo_data/s1_refrigerator_base_3180_3240.npy",
        # "./demo_data/s1_refrigerator_part1_3180_3240.npy"

# close refrigerator
        # "./demo_data/s1_refrigerator_part2_3360_3420.npy",
        # "./demo_data/s1_refrigerator_base_3360_3420.npy",
        # "./demo_data/s1_refrigerator_part1_3360_3420.npy"

# open and close drawer
#         "./demo_data/s2_drawer_part1_1770_1950.npy",
#         "./demo_data/s2_drawer_base_1770_1950.npy",
#         "./demo_data/s2_drawer_part2_1770_1950.npy"

# #open gasstove
#         "./demo_data/s1_gasstove_part2_1110_1170.npy",
#         "./demo_data/s1_gasstove_part1_1110_1170.npy",
#         "./demo_data/s1_gasstove_base_1110_1170.npy"

#close gasstove
        # "./demo_data/s1_gasstove_part2_2760_2850.npy",
        # "./demo_data/s1_gasstove_part1_2760_2850.npy",
        # "./demo_data/s1_gasstove_base_2760_2850.npy"

# open microwave
#         "./demo_data/s1_microwave_part1_1380_1470.npy",     #
#         "./demo_data/s1_microwave_base_1380_1470.npy",

# close microwave
#         "./demo_data/s1_microwave_part1_1740_1830.npy",
#         "./demo_data/s1_microwave_base_1740_1830.npy",

# open drawer
        "./demo_data/s2_drawer_part1_1770_1950.npy",
        "./demo_data/s2_drawer_base_1770_1950.npy",
        "./demo_data/s2_drawer_part2_1770_1950.npy",

# open washingmachine
        # "./demo_data/s2_washingmachine_part1_1140_1170.npy",
        # "./demo_data/s2_washingmachine_base_1140_1170.npy"

# close washingmachine
#         "./demo_data/s2_washingmachine_part1_1260_1290.npy",
#         "./demo_data/s2_washingmachine_base_1260_1290.npy"

# #chair
#         "./demo_data/s3_chair_base_2610_2760.npy"

# #chair
#         "./demo_data/s6_chair_base_90_270.npy"

# open laptop
        # "./demo_data/s3_laptop_part1_3000_3090.npy",
        # "./demo_data/s3_laptop_base_3000_3090.npy"

# open laptop
        # "./demo_data/s6_trashbin_part1_750_900.npy",
        # "./demo_data/s6_trashbin_base_750_900.npy"

# #cap
#         "./demo_data/screw.npy"

# #prismatic door
#         "./demo_data/prismatic.npy"
#
# #planar
#         "./demo_data/planar.npy"
#
# #ball
#         "./demo_data/ball.npy"
        
#revolute door
        # "./demo_data/revolute.npy"

    ]


    # Create EnhancedViz instance and execute visualization, passing multiple file paths
    viz = EnhancedViz(file_paths)