"""
Geometric and motion utilities for joint analysis.
"""

import numpy as np


def point_line_distance(p, line_origin, line_dir):
    """
    Compute the distance from a 3D point p to a line defined by (line_origin, line_dir).
    line_dir should be a unit vector.

    Args:
        p (ndarray): Point coordinates (3,)
        line_origin (ndarray): Line origin point (3,)
        line_dir (ndarray): Line direction (unit vector) (3,)

    Returns:
        float: Distance from point to line
    """
    v = p - line_origin
    cross_ = np.cross(v, line_dir)
    return np.linalg.norm(cross_)


def translate_points(points, displacement, axis):
    """
    Translate points by a given displacement along a specified axis.

    Args:
        points (ndarray): Points to translate (N, 3)
        displacement (float): Displacement magnitude
        axis (ndarray): Displacement direction (3,)

    Returns:
        ndarray: Translated points (N, 3)
    """
    return points + displacement * axis


def rotate_points(points, angle, axis, origin):
    """
    Rotate points around a given origin by an angle around a specified axis (which need not be unit).

    Args:
        points (ndarray): Points to rotate (N, 3)
        angle (float): Rotation angle in radians
        axis (ndarray): Rotation axis (3,)
        origin (ndarray): Origin of rotation (3,)

    Returns:
        ndarray: Rotated points (N, 3)
    """
    # Normalize axis
    axis = axis / np.linalg.norm(axis)

    # Center points at origin
    points = points - origin

    # Compute rotation matrix (Rodrigues' rotation formula)
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    R = np.array([
        [t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
        [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0]],
        [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c]
    ])

    # Apply rotation and translate back
    rotated_points = points @ R.T
    rotated_points += origin

    return rotated_points


def rotate_points_y(points, angle, center):
    """
    Rotate points around the y-axis.

    Args:
        points (ndarray): Points to rotate (N, 3)
        angle (float): Rotation angle in radians
        center (ndarray): Center of rotation (3,)

    Returns:
        ndarray: Rotated points (N, 3)
    """
    points = points - center
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([
        [c, 0., s],
        [0, 1., 0.],
        [-s, 0., c]
    ])
    rotated_points = points @ R.T
    rotated_points += center
    return rotated_points


def rotate_points_xyz(points, angle_x, angle_y, angle_z, center):
    """
    Rotate points around X, then Y, then Z axes in sequence.

    Args:
        points (ndarray): Points to rotate (N, 3)
        angle_x (float): Rotation angle around X-axis in radians
        angle_y (float): Rotation angle around Y-axis in radians
        angle_z (float): Rotation angle around Z-axis in radians
        center (ndarray): Center of rotation (3,)

    Returns:
        ndarray: Rotated points (N, 3)
    """
    points = points - center

    # X-axis rotation
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])

    # Y-axis rotation
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])

    # Z-axis rotation
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])

    # Apply rotations sequentially
    rotated = points @ Rx.T
    rotated = rotated @ Ry.T
    rotated = rotated @ Rz.T

    # Translate back
    rotated += center

    return rotated


def apply_screw_motion(points, angle, axis, origin, pitch):
    """
    Apply a screw motion: rotate around an axis by angle, then translate along the axis proportionally.

    Args:
        points (ndarray): Points to transform (N, 3)
        angle (float): Rotation angle in radians
        axis (ndarray): Screw axis direction (3,)
        origin (ndarray): Origin point on the screw axis (3,)
        pitch (float): Pitch of the screw motion (translation per 2π rotation)

    Returns:
        ndarray: Transformed points (N, 3)
    """
    # Normalize axis
    axis = axis / np.linalg.norm(axis)

    # Center points at origin
    points = points - origin

    # Compute rotation matrix
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    R = np.array([
        [t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
        [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c, t * axis[1] * axis[2] - s * axis[0]],
        [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c]
    ])

    # Apply rotation
    rotated_points = points @ R.T

    # Apply translation along the axis (proportional to angle)
    translation = (angle / (2 * np.pi)) * pitch * axis
    transformed_points = rotated_points + translation

    # Translate back
    transformed_points += origin

    return transformed_points


def generate_sphere(center, radius, num_points):
    """
    Randomly sample points inside a sphere.

    Args:
        center (ndarray): Center of the sphere (3,)
        radius (float): Radius of the sphere
        num_points (int): Number of points to generate

    Returns:
        ndarray: Generated points (num_points, 3)
    """
    phi = np.random.rand(num_points) * 2 * np.pi
    costheta = 2 * np.random.rand(num_points) - 1
    theta = np.arccos(costheta)
    r = radius * (np.random.rand(num_points) ** (1 / 3))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return center + np.vstack([x, y, z]).T


def generate_cylinder(radius, height, num_points=500):
    """
    Randomly sample points in a cylinder.

    Args:
        radius (float): Radius of the cylinder
        height (float): Height of the cylinder
        num_points (int): Number of points to generate

    Returns:
        ndarray: Generated points (num_points, 3)
    """
    zs = np.random.rand(num_points) * height - height / 2
    phi = np.random.rand(num_points) * 2 * np.pi
    rs = radius * np.sqrt(np.random.rand(num_points))
    xs = rs * np.cos(phi)
    ys = zs
    zs = rs * np.sin(phi)
    return np.vstack([xs, ys, zs]).T


def generate_ball_joint_points(center, sphere_radius, rod_length, rod_radius,
                               num_points_sphere=250, num_points_rod=250):
    """
    Generate a ball joint: a sphere + a cylinder-like rod.

    Args:
        center (ndarray): Center of the ball joint (3,)
        sphere_radius (float): Radius of the sphere
        rod_length (float): Length of the rod
        rod_radius (float): Radius of the rod
        num_points_sphere (int): Number of points for the sphere
        num_points_rod (int): Number of points for the rod

    Returns:
        ndarray: Generated points (num_points_sphere + num_points_rod, 3)
    """
    sphere_pts = generate_sphere(center, sphere_radius, num_points_sphere)
    rod_pts = generate_cylinder(rod_radius, rod_length, num_points_rod)
    rod_pts[:, 1] += center[1]  # Align rod with the sphere
    return np.concatenate([sphere_pts, rod_pts], axis=0)


def generate_hollow_cylinder(radius, height, thickness,
                             num_points=500, cap_position="top", cap_points_ratio=0.2):
    """
    Generate a hollow cylinder with one end capped.

    Args:
        radius (float): Outer radius of the cylinder
        height (float): Height of the cylinder
        thickness (float): Wall thickness of the cylinder
        num_points (int): Total number of points to generate
        cap_position (str): Position of the cap ("top" or "bottom")
        cap_points_ratio (float): Ratio of points to use for the cap

    Returns:
        ndarray: Generated points (num_points, 3)
    """
    num_wall_points = int(num_points * (1 - cap_points_ratio))
    num_cap_points = num_points - num_wall_points

    # Generate wall points
    rr_wall = radius - np.random.rand(num_wall_points) * thickness
    theta_wall = np.random.rand(num_wall_points) * 2 * np.pi
    z_wall = np.random.rand(num_wall_points) * height
    x_wall = rr_wall * np.cos(theta_wall)
    zs = rr_wall * np.sin(theta_wall)
    y_wall = z_wall
    z_wall = zs

    # Generate cap points
    rr_cap = np.random.rand(num_cap_points) * radius
    theta_cap = np.random.rand(num_cap_points) * 2 * np.pi
    z_cap = np.full_like(rr_cap, height if cap_position == "top" else 0.0)
    x_cap = rr_cap * np.cos(theta_cap)
    y_cap = z_cap
    z_cap = rr_cap * np.sin(theta_cap)

    # Combine all points
    x = np.hstack([x_wall, x_cap])
    y = np.hstack([y_wall, y_cap])
    z = np.hstack([z_wall, z_cap])

    return np.vstack([x, y, z]).T