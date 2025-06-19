import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim


class NPYVisualizer:
    """Visualize NPY point cloud data using Polyscope"""

    def __init__(self):
        """Initialize the visualizer"""
        # Initialize Polyscope
        if not ps.is_initialized():
            ps.init()

        # Disable ground plane
        ps.set_ground_plane_mode("none")

        # Store point clouds and paths
        self.point_clouds = {}
        self.all_files = {}
        self.current_file = None
        self.current_frame = 0
        self.total_frames = 0
        self.auto_play = False
        self.play_speed = 1
        self.root_dir = "exported_pointclouds"

        # Find all NPY files
        self._find_all_npy_files()

    def _find_all_npy_files(self):
        """Find all NPY files in the exported_pointclouds directory"""
        self.all_files = {}

        if not os.path.exists(self.root_dir):
            print(f"Error: Directory '{self.root_dir}' does not exist")
            return

        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('.npy'):
                    file_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(file_path, self.root_dir)
                    self.all_files[rel_path] = file_path

    def load_npy_file(self, file_path):
        """Load NPY file"""
        try:
            data = np.load(file_path)

            # Check data shape
            if len(data.shape) != 3 or data.shape[2] != 3:
                print(f"Warning: File {file_path} has shape {data.shape}, expected (T, N, 3)")
                return None

            print(f"Loaded file: {os.path.basename(file_path)}")
            print(f"Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def register_point_cloud(self, frame, points, name="Point Cloud"):
        """Register point cloud in Polyscope"""
        # Clear all previous point clouds
        for cloud_name in list(self.point_clouds.keys()):
            ps.remove_point_cloud(cloud_name)
        self.point_clouds.clear()

        # Register new point cloud
        cloud_name = f"{name}_Frame{frame}"
        self.point_clouds[cloud_name] = ps.register_point_cloud(cloud_name, points)

        # Optional: Add color to point cloud
        color = np.ones((points.shape[0], 3)) * 0.7  # Default light gray
        self.point_clouds[cloud_name].add_color_quantity("Default Color", color, enabled=True)

        # Optional: Set point size
        self.point_clouds[cloud_name].set_radius(0.01)

        return cloud_name

    def show_sequence(self, file_path, frame_index=0):
        """Show specific frame in sequence"""
        data = self.load_npy_file(file_path)
        if data is None:
            return

        self.current_file = file_path
        self.total_frames = data.shape[0]
        self.current_frame = min(frame_index, self.total_frames - 1)

        file_name = os.path.basename(file_path)
        points = data[self.current_frame]
        self.register_point_cloud(self.current_frame, points, file_name)

    def update_frame(self, new_frame):
        """Update displayed frame"""
        if self.current_file is None:
            return

        data = self.load_npy_file(self.current_file)
        if data is None:
            return

        self.current_frame = min(max(0, new_frame), self.total_frames - 1)
        file_name = os.path.basename(self.current_file)
        points = data[self.current_frame]
        self.register_point_cloud(self.current_frame, points, file_name)

    def polyscope_callback(self):
        """Polyscope UI callback"""
        psim.TextUnformatted("NPY Point Cloud Visualization")
        psim.Separator()

        # File selection dropdown menu
        if psim.BeginCombo("Select NPY File", os.path.basename(self.current_file) if self.current_file else "Select File"):
            for rel_path, file_path in self.all_files.items():
                _, selected = psim.Selectable(rel_path, self.current_file == file_path)
                if selected and self.current_file != file_path:
                    self.show_sequence(file_path, 0)
            psim.EndCombo()

        psim.Separator()

        # Frame controls
        if self.current_file is not None:
            # Display current frame information
            psim.TextUnformatted(f"Total Frames: {self.total_frames}")

            # Frame slider
            changed, new_frame = psim.SliderInt("Frame", self.current_frame, 0, self.total_frames - 1)
            if changed:
                self.update_frame(new_frame)

            # Playback controls
            psim.TextUnformatted("Playback Controls:")
            if self.auto_play:
                if psim.Button("Pause"):
                    self.auto_play = False
            else:
                if psim.Button("Play"):
                    self.auto_play = True

            psim.SameLine()
            if psim.Button("Previous Frame"):
                self.update_frame(self.current_frame - 1)

            psim.SameLine()
            if psim.Button("Next Frame"):
                self.update_frame(self.current_frame + 1)

            # Playback speed
            changed, new_speed = psim.SliderInt("Playback Speed", self.play_speed, 1, 10)
            if changed:
                self.play_speed = new_speed

            # If auto playing, update frame
            if self.auto_play:
                new_frame = (self.current_frame + self.play_speed) % self.total_frames
                self.update_frame(new_frame)

        psim.Separator()

        # Point cloud settings
        if self.point_clouds:
            psim.TextUnformatted("Point Cloud Settings:")

            # Point size
            for cloud_name, cloud in self.point_clouds.items():
                radius = cloud.get_radius()
                changed, new_radius = psim.SliderFloat("Point Size", radius, 0.001, 0.05)
                if changed:
                    cloud.set_radius(new_radius)

        # Refresh file list button
        if psim.Button("Refresh File List"):
            self._find_all_npy_files()

    def run(self):
        """Run the visualizer"""
        ps.set_user_callback(self.polyscope_callback)
        ps.show()


def main():
    """Main function"""
    print("Initializing Polyscope visualizer...")
    visualizer = NPYVisualizer()

    # If files exist, show the first one
    if visualizer.all_files:
        first_file = list(visualizer.all_files.values())[0]
        visualizer.show_sequence(first_file)

    print("Starting Polyscope interface...")
    visualizer.run()

if __name__ == "__main__":
    main()