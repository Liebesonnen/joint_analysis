"""
Generate synthetic data for testing joint estimation algorithms.
"""

import numpy as np
import os

from ..core.geometry import (
    rotate_points, rotate_points_y, rotate_points_xyz,
    translate_points, apply_screw_motion,
    generate_sphere, generate_cylinder,
    generate_ball_joint_points, generate_hollow_cylinder
)


class SyntheticJointGenerator:
    """Class for generating synthetic joint data."""

    def __init__(self, output_dir="generated_data", num_points=500, noise_sigma=0.0):
        """
        Initialize the synthetic joint generator.

        Args:
            output_dir (str): Directory to save generated data
            num_points (int): Number of points to generate
            noise_sigma (float): Standard deviation of noise to add
        """
        self.output_dir = output_dir
        self.num_points = num_points
        self.noise_sigma = noise_sigma

        os.makedirs(output_dir, exist_ok=True)

        # Initialize object points
        self._init_objects()

    def _init_objects(self):
        """Initialize all the synthetic objects."""
        # Prismatic Door 1
        door_width, door_height, door_thickness = 2.0, 3.0, 0.2
        self.prismatic_door_points = np.random.rand(self.num_points, 3)
        self.prismatic_door_points[:, 0] = self.prismatic_door_points[:, 0] * door_width - 0.5 * door_width
        self.prismatic_door_points[:, 1] = self.prismatic_door_points[:, 1] * door_height
        self.prismatic_door_points[:, 2] = self.prismatic_door_points[:, 2] * door_thickness - 0.5 * door_thickness
        self.prismatic_door_axis = np.array([1., 0., 0.])

        # Prismatic Door 2
        self.prismatic_door_points_2 = self.prismatic_door_points.copy() + np.array([1.5, 0., 0.])
        self.prismatic_door_axis_2 = np.array([0., 1., 0.])

        # Prismatic Door 3
        self.prismatic_door_points_3 = self.prismatic_door_points.copy() + np.array([-1., 1., 0.])
        self.prismatic_door_axis_3 = np.array([0., 0., 1.])

        # Revolute Door 1
        self.revolute_door_points = self.prismatic_door_points.copy()
        self.revolute_door_origin = np.array([1.0, 1.5, 0.0])
        self.revolute_door_axis = np.array([0.0, 1.0, 0.0])

        # Revolute Door 2
        self.revolute_door_points_2 = self.revolute_door_points.copy() + np.array([0., 0., -1.])
        self.revolute_door_origin_2 = np.array([0.5, 2.0, -1.0])
        self.revolute_door_axis_2 = np.array([1.0, 0.0, 0.0])

        # Revolute Door 3
        self.revolute_door_points_3 = self.revolute_door_points.copy() + np.array([0., -0.5, 1.0])
        self.revolute_door_origin_3 = np.array([2.0, 1.0, 1.0])
        self.revolute_door_axis_3 = np.array([1.0, 1.0, 0.])
        self.revolute_door_axis_3 = self.revolute_door_axis_3 / np.linalg.norm(self.revolute_door_axis_3)

        # Planar Mouse 1
        mouse_length, mouse_width, mouse_height = 1.0, 0.6, 0.3
        self.planar_mouse_points = np.zeros((self.num_points, 3))
        self.planar_mouse_points[:, 0] = np.random.rand(self.num_points) * mouse_length - 0.5 * mouse_length
        self.planar_mouse_points[:, 2] = np.random.rand(self.num_points) * mouse_width - 0.5 * mouse_width
        self.planar_mouse_points[:, 1] = np.random.rand(self.num_points) * mouse_height
        self.planar_mouse_normal = np.array([0., 1., 0.])

        # Planar Mouse 2
        self.planar_mouse_points_2 = self.planar_mouse_points.copy() + np.array([1., 0., 1.])
        self.planar_mouse_normal_2 = np.array([0., 1., 0.])

        # Planar Mouse 3
        self.planar_mouse_points_3 = self.planar_mouse_points.copy() + np.array([-1., 0., 1.])
        self.planar_mouse_normal_3 = np.array([0., 1., 0.])

        # Ball Joint 1
        sphere_radius = 0.3
        rod_length = sphere_radius * 10.0
        rod_radius = 0.05
        self.ball_joint_points = generate_ball_joint_points(
            np.array([0., 0., 0.]), sphere_radius, rod_length, rod_radius, 250, 250
        )
        self.ball_joint_center = np.array([0., 0., 0.])

        # Ball Joint 2
        self.ball_joint_points_2 = generate_ball_joint_points(
            np.array([1., 0., 0.]), sphere_radius, rod_length, rod_radius, 250, 250
        )
        self.ball_joint_center_2 = np.array([1., 0., 0.])

        # Ball Joint 3
        self.ball_joint_points_3 = generate_ball_joint_points(
            np.array([1., 1., 0.]), sphere_radius, rod_length, rod_radius, 250, 250
        )
        self.ball_joint_center_3 = np.array([1., 1., 0.])

        # Screw Joint 1
        self.screw_joint_points = generate_hollow_cylinder(
            radius=0.4, height=0.2, thickness=0.05,
            num_points=500, cap_position="top", cap_points_ratio=0.2
        )
        self.screw_joint_axis = np.array([0., 1., 0.])
        self.screw_joint_origin = np.array([0., 0., 0.])
        self.screw_joint_pitch = 0.5

        # Screw Joint 2
        self.screw_joint_points_2 = self.screw_joint_points.copy() + np.array([1., 0., 0.])
        self.screw_joint_axis_2 = np.array([1., 0., 0.])
        self.screw_joint_origin_2 = np.array([1., 0., 0.])
        self.screw_joint_pitch_2 = 0.8

        # Screw Joint 3
        self.screw_joint_points_3 = self.screw_joint_points.copy() + np.array([-1., 0., 1.])
        self.screw_joint_axis_3 = np.array([1., 1., 0.])
        self.screw_joint_axis_3 = self.screw_joint_axis_3 / np.linalg.norm(self.screw_joint_axis_3)
        self.screw_joint_origin_3 = np.array([-1., 0., 1.])
        self.screw_joint_pitch_3 = 0.6

    def generate_prismatic_door_sequence(self, n_frames=50, max_displacement=5.0):
        """
        Generate a sequence of a prismatic door motion.

        Args:
            n_frames (int): Number of frames to generate
            max_displacement (float): Maximum displacement

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            pos = (i / (n_frames - 1)) * max_displacement
            new_points = self.prismatic_door_points.copy()
            new_points = translate_points(new_points, pos, self.prismatic_door_axis)

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_prismatic_door_2_sequence(self, n_frames=50, max_displacement=4.0):
        """
        Generate a sequence of a second prismatic door motion.

        Args:
            n_frames (int): Number of frames to generate
            max_displacement (float): Maximum displacement

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            pos = (i / (n_frames - 1)) * max_displacement
            new_points = self.prismatic_door_points_2.copy()
            new_points = translate_points(new_points, pos, self.prismatic_door_axis_2)

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_prismatic_door_3_sequence(self, n_frames=50, max_displacement=3.0):
        """
        Generate a sequence of a third prismatic door motion.

        Args:
            n_frames (int): Number of frames to generate
            max_displacement (float): Maximum displacement

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            pos = (i / (n_frames - 1)) * max_displacement
            new_points = self.prismatic_door_points_3.copy()
            new_points = translate_points(new_points, pos, self.prismatic_door_axis_3)

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_revolute_door_sequence(self, n_frames=50, angle_min=-45.0, angle_max=45.0):
        """
        Generate a sequence of a revolute door motion.

        Args:
            n_frames (int): Number of frames to generate
            angle_min (float): Minimum angle in degrees
            angle_max (float): Maximum angle in degrees

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            angle = np.radians(angle_min + (angle_max - angle_min) * t)
            new_points = self.revolute_door_points.copy()
            new_points = rotate_points(new_points, angle, self.revolute_door_axis, self.revolute_door_origin)

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_revolute_door_2_sequence(self, n_frames=50, angle_min=-30.0, angle_max=60.0):
        """
        Generate a sequence of a second revolute door motion.

        Args:
            n_frames (int): Number of frames to generate
            angle_min (float): Minimum angle in degrees
            angle_max (float): Maximum angle in degrees

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            angle = np.radians(angle_min + (angle_max - angle_min) * t)
            new_points = self.revolute_door_points_2.copy()
            new_points = rotate_points(new_points, angle, self.revolute_door_axis_2, self.revolute_door_origin_2)

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_revolute_door_3_sequence(self, n_frames=50, angle_min=0.0, angle_max=90.0):
        """
        Generate a sequence of a third revolute door motion.

        Args:
            n_frames (int): Number of frames to generate
            angle_min (float): Minimum angle in degrees
            angle_max (float): Maximum angle in degrees

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            angle = np.radians(angle_min + (angle_max - angle_min) * t)
            new_points = self.revolute_door_points_3.copy()
            new_points = rotate_points(new_points, angle, self.revolute_door_axis_3, self.revolute_door_origin_3)

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_planar_mouse_sequence(self, n_frames=50):
        """
        Generate a sequence of a planar mouse motion.

        Args:
            n_frames (int): Number of frames to generate

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            new_points = self.planar_mouse_points.copy()

            if i < n_frames // 2:
                # First half: translation
                alpha = i / (n_frames // 2 - 1)
                tx = -1.0 + alpha * (0.0 - (-1.0))
                tz = 1.0 + alpha * (0.0 - 1.0)
                ry = 0.0
                new_points += np.array([tx, 0., tz])
                new_points = rotate_points_y(new_points, ry, [0., 0., 0.])
            else:
                # Second half: rotation and translation
                alpha = (i - n_frames // 2) / (n_frames - n_frames // 2 - 1)
                ry = np.radians(40.0) * alpha
                tx = alpha * 1.0
                tz = alpha * (-1.0)
                new_points += np.array([tx, 0., tz])
                new_points = rotate_points_y(new_points, ry, [0., 0., 0.])

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_planar_mouse_2_sequence(self, n_frames=50):
        """
        Generate a sequence of a second planar mouse motion.

        Args:
            n_frames (int): Number of frames to generate

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            if i < n_frames // 2:
                # First half: translation
                alpha = i / (n_frames // 2 - 1)
                dy = alpha * 1.0
                dz = alpha * 1.0
                new_points = self.planar_mouse_points_2.copy()
                new_points += np.array([0., dy, dz])
            else:
                # Second half: rotation and translation
                alpha = (i - n_frames // 2) / (n_frames - n_frames // 2 - 1)
                mp = self.planar_mouse_points_2.copy()
                mp += np.array([0., 1.0, 1.0])
                rx = np.radians(30.0) * alpha
                new_points = rotate_points(mp, rx,
                                           np.array([1.0, 0., 0.]),
                                           np.array([0., 1.0, 1.0]))

                dy = alpha * 1.0
                dz = alpha * 0.5
                new_points += np.array([0., dy, dz])

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_planar_mouse_3_sequence(self, n_frames=50):
        """
        Generate a sequence of a third planar mouse motion.

        Args:
            n_frames (int): Number of frames to generate

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            if i < n_frames // 2:
                # First half: translation
                alpha = i / (n_frames // 2 - 1)
                dx = alpha * 1.0
                dz = alpha * 0.5
                new_points = self.planar_mouse_points_3.copy()
                new_points += np.array([dx, 0., dz])
            else:
                # Second half: rotation and translation
                alpha = (i - n_frames // 2) / (n_frames - n_frames // 2 - 1)
                mp = self.planar_mouse_points_3.copy()
                mp += np.array([1.0, 0., 0.5])
                ry = np.radians(30.0) * alpha
                new_points = rotate_points(mp, ry,
                                           np.array([0., 1., 0.]),
                                           np.array([1.0, 0., 0.5]))

                dx = alpha * 1.0
                dz = alpha * 0.5
                new_points += np.array([dx, 0., dz])

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_ball_joint_sequence(self, n_frames=50):
        """
        Generate a sequence of a ball joint motion.

        Args:
            n_frames (int): Number of frames to generate

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)

            if i < n_frames // 3:
                # First third: X rotation
                alpha = i / (n_frames // 3 - 1)
                ax = np.radians(60.0) * alpha
                ay = 0.0
                az = 0.0
            elif i < 2 * n_frames // 3:
                # Second third: Y rotation
                alpha = (i - n_frames // 3) / (n_frames // 3 - 1)
                ax = np.radians(60.0)
                ay = np.radians(40.0) * alpha
                az = 0.0
            else:
                # Last third: Z rotation
                alpha = (i - 2 * n_frames // 3) / (n_frames - 2 * n_frames // 3 - 1)
                ax = np.radians(60.0)
                ay = np.radians(40.0)
                az = np.radians(70.0) * alpha

            new_points = self.ball_joint_points.copy()
            new_points = rotate_points_xyz(new_points, ax, ay, az, self.ball_joint_center)

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_ball_joint_2_sequence(self, n_frames=50):
        """
        Generate a sequence of a second ball joint motion.

        Args:
            n_frames (int): Number of frames to generate

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)

            if i < n_frames // 2:
                # First half: X and Y rotation
                alpha = i / (n_frames // 2 - 1)
                rx = np.radians(50.0) * alpha
                ry = np.radians(10.0) * alpha
                rz = 0.0
            else:
                # Second half: Z rotation
                alpha = (i - n_frames // 2) / (n_frames - n_frames // 2 - 1)
                rx = np.radians(50.0)
                ry = np.radians(10.0)
                rz = np.radians(45.0) * alpha

            new_points = self.ball_joint_points_2.copy()
            new_points = rotate_points_xyz(new_points, rx, ry, rz, self.ball_joint_center_2)

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_ball_joint_3_sequence(self, n_frames=50):
        """
        Generate a sequence of a third ball joint motion.

        Args:
            n_frames (int): Number of frames to generate

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)

            if i < n_frames // 3:
                # First third: X rotation
                alpha = i / (n_frames // 3 - 1)
                ax = np.radians(30.0) * alpha
                ay = 0.0
                az = 0.0
            elif i < 2 * n_frames // 3:
                # Second third: Y rotation
                alpha = (i - n_frames // 3) / (n_frames // 3 - 1)
                ax = np.radians(30.0)
                ay = np.radians(50.0) * alpha
                az = 0.0
            else:
                # Last third: Z rotation
                alpha = (i - 2 * n_frames // 3) / (n_frames - 2 * n_frames // 3 - 1)
                ax = np.radians(30.0)
                ay = np.radians(50.0)
                az = np.radians(80.0) * alpha

            new_points = self.ball_joint_points_3.copy()
            new_points = rotate_points_xyz(new_points, ax, ay, az, self.ball_joint_center_3)

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_screw_joint_sequence(self, n_frames=50):
        """
        Generate a sequence of a screw joint motion.

        Args:
            n_frames (int): Number of frames to generate

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            angle = 2 * np.pi * t

            new_points = self.screw_joint_points.copy()
            new_points = apply_screw_motion(
                new_points, angle, self.screw_joint_axis,
                self.screw_joint_origin, self.screw_joint_pitch
            )

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_screw_joint_2_sequence(self, n_frames=50):
        """
        Generate a sequence of a second screw joint motion.

        Args:
            n_frames (int): Number of frames to generate

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            angle = 2 * np.pi * t

            new_points = self.screw_joint_points_2.copy()
            new_points = apply_screw_motion(
                new_points, angle, self.screw_joint_axis_2,
                self.screw_joint_origin_2, self.screw_joint_pitch_2
            )

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_screw_joint_3_sequence(self, n_frames=50):
        """
        Generate a sequence of a third screw joint motion.

        Args:
            n_frames (int): Number of frames to generate

        Returns:
            ndarray: Point sequence of shape (n_frames, num_points, 3)
        """
        frames = []
        for i in range(n_frames):
            t = i / (n_frames - 1)
            angle = 2 * np.pi * t

            new_points = self.screw_joint_points_3.copy()
            new_points = apply_screw_motion(
                new_points, angle, self.screw_joint_axis_3,
                self.screw_joint_origin_3, self.screw_joint_pitch_3
            )

            # Add noise if specified
            if self.noise_sigma > 0:
                new_points += np.random.normal(0, self.noise_sigma, new_points.shape)

            frames.append(new_points)

        return np.stack(frames, axis=0)

    def generate_all_sequences(self, n_frames=50):
        """
        Generate all sequences and save them to files.

        Args:
            n_frames (int): Number of frames for each sequence

        Returns:
            dict: Dictionary mapping joint types to file paths
        """
        result = {}

        # Prismatic joints
        prismatic_1_seq = self.generate_prismatic_door_sequence(n_frames)
        prismatic_2_seq = self.generate_prismatic_door_2_sequence(n_frames)
        prismatic_3_seq = self.generate_prismatic_door_3_sequence(n_frames)

        # Revolute joints
        revolute_1_seq = self.generate_revolute_door_sequence(n_frames)
        revolute_2_seq = self.generate_revolute_door_2_sequence(n_frames)
        revolute_3_seq = self.generate_revolute_door_3_sequence(n_frames)

        # Planar joints
        planar_1_seq = self.generate_planar_mouse_sequence(n_frames)
        planar_2_seq = self.generate_planar_mouse_2_sequence(n_frames)
        planar_3_seq = self.generate_planar_mouse_3_sequence(n_frames)

        # Ball joints
        ball_1_seq = self.generate_ball_joint_sequence(n_frames)
        ball_2_seq = self.generate_ball_joint_2_sequence(n_frames)
        ball_3_seq = self.generate_ball_joint_3_sequence(n_frames)

        # Screw joints
        screw_1_seq = self.generate_screw_joint_sequence(n_frames)
        screw_2_seq = self.generate_screw_joint_2_sequence(n_frames)
        screw_3_seq = self.generate_screw_joint_3_sequence(n_frames)

        # Save all sequences
        os.makedirs(os.path.join(self.output_dir, "prismatic"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "revolute"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "planar"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "ball"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "screw"), exist_ok=True)

        # Prismatic
        file_path = os.path.join(self.output_dir, "prismatic", "prismatic_1.npy")
        np.save(file_path, prismatic_1_seq)
        result["prismatic_1"] = file_path

        file_path = os.path.join(self.output_dir, "prismatic", "prismatic_2.npy")
        np.save(file_path, prismatic_2_seq)
        result["prismatic_2"] = file_path

        file_path = os.path.join(self.output_dir, "prismatic", "prismatic_3.npy")
        np.save(file_path, prismatic_3_seq)
        result["prismatic_3"] = file_path

        # Revolute
        file_path = os.path.join(self.output_dir, "revolute", "revolute_1.npy")
        np.save(file_path, revolute_1_seq)
        result["revolute_1"] = file_path

        file_path = os.path.join(self.output_dir, "revolute", "revolute_2.npy")
        np.save(file_path, revolute_2_seq)
        result["revolute_2"] = file_path

        file_path = os.path.join(self.output_dir, "revolute", "revolute_3.npy")
        np.save(file_path, revolute_3_seq)
        result["revolute_3"] = file_path

        # Planar
        file_path = os.path.join(self.output_dir, "planar", "planar_1.npy")
        np.save(file_path, planar_1_seq)
        result["planar_1"] = file_path

        file_path = os.path.join(self.output_dir, "planar", "planar_2.npy")
        np.save(file_path, planar_2_seq)
        result["planar_2"] = file_path

        file_path = os.path.join(self.output_dir, "planar", "planar_3.npy")
        np.save(file_path, planar_3_seq)
        result["planar_3"] = file_path

        # Ball
        file_path = os.path.join(self.output_dir, "ball", "ball_1.npy")
        np.save(file_path, ball_1_seq)
        result["ball_1"] = file_path

        file_path = os.path.join(self.output_dir, "ball", "ball_2.npy")
        np.save(file_path, ball_2_seq)
        result["ball_2"] = file_path

        file_path = os.path.join(self.output_dir, "ball", "ball_3.npy")
        np.save(file_path, ball_3_seq)
        result["ball_3"] = file_path

        # Screw
        file_path = os.path.join(self.output_dir, "screw", "screw_1.npy")
        np.save(file_path, screw_1_seq)
        result["screw_1"] = file_path

        file_path = os.path.join(self.output_dir, "screw", "screw_2.npy")
        np.save(file_path, screw_2_seq)
        result["screw_2"] = file_path

        file_path = os.path.join(self.output_dir, "screw", "screw_3.npy")
        np.save(file_path, screw_3_seq)
        result["screw_3"] = file_path

        return result