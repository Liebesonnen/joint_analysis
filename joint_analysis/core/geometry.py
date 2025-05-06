"""
Geometric and motion utilities for joint analysis.
"""

import numpy as np


def point_line_distance(p, line_origin, line_dir):
    """Compute the distance from a 3D point p to a line defined by (line_origin, line_dir).
       line_dir should be a unit vector."""
    v = p - line_origin
    cross_ = np.cross(v, line_dir)
    return np.linalg.norm(cross_)

def translate_points(points, displacement, axis):
    """Translate points by a given displacement along a specified axis."""
    return points + displacement * axis

def rotate_points(points, angle, axis, origin):
    """Rotate points around a given origin by an angle around a specified axis (which need not be unit)."""
    axis = axis / np.linalg.norm(axis)
    points = points - origin
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    R = np.array([
        [t * axis[0] * axis[0] + c,         t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
        [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c,         t * axis[1] * axis[2] - s * axis[0]],
        [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c       ]
    ])
    rotated_points = points @ R.T
    rotated_points += origin
    return rotated_points

def rotate_points_y(points, angle, center):
    """Rotate points around the y-axis."""
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
    """Rotate points around X, then Y, then Z axes in sequence."""
    points = points - center
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z),  np.cos(angle_z), 0],
        [0,                0,               1]
    ])
    rotated = points @ Rx.T
    rotated = rotated @ Ry.T
    rotated = rotated @ Rz.T
    rotated += center
    return rotated

def apply_screw_motion(points, angle, axis, origin, pitch):
    """Apply a screw motion: rotate around an axis by angle, then translate along the axis proportionally."""
    axis = axis / np.linalg.norm(axis)
    points = points - origin
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    R = np.array([
        [t * axis[0] * axis[0] + c,         t * axis[0] * axis[1] - s * axis[2], t * axis[0] * axis[2] + s * axis[1]],
        [t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c,         t * axis[1] * axis[2] - s * axis[0]],
        [t * axis[0] * axis[2] - s * axis[1], t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c       ]
    ])
    rotated_points = points @ R.T
    translation = (angle / (2 * np.pi)) * pitch * axis
    transformed_points = rotated_points + translation
    transformed_points += origin
    return transformed_points

def generate_sphere(center, radius, num_points):
    """Randomly sample points inside a sphere."""
    phi = np.random.rand(num_points) * 2 * np.pi
    costheta = 2 * np.random.rand(num_points) - 1
    theta = np.arccos(costheta)
    r = radius * (np.random.rand(num_points) ** (1 / 3))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return center + np.vstack([x, y, z]).T

def generate_cylinder(radius, height, num_points=500):
    """Randomly sample points in a cylinder."""
    zs = np.random.rand(num_points) * height - height / 2
    phi = np.random.rand(num_points) * 2 * np.pi
    rs = radius * np.sqrt(np.random.rand(num_points))
    xs = rs * np.cos(phi)
    ys = zs
    zs = rs * np.sin(phi)
    return np.vstack([xs, ys, zs]).T

def generate_ball_joint_points(center, sphere_radius, rod_length, rod_radius,
                               num_points_sphere=250, num_points_rod=250):
    """Generate a ball joint: a sphere + a cylinder-like rod."""
    sphere_pts = generate_sphere(center, sphere_radius, num_points_sphere)
    rod_pts = generate_cylinder(rod_radius, rod_length, num_points_rod)
    rod_pts[:, 1] += center[1]
    return np.concatenate([sphere_pts, rod_pts], axis=0)

def generate_hollow_cylinder(radius, height, thickness,
                             num_points=500, cap_position="top", cap_points_ratio=0.2):
    """Generate a hollow cylinder with one end capped."""
    num_wall_points = int(num_points * (1 - cap_points_ratio))
    num_cap_points = num_points - num_wall_points

    rr_wall = radius - np.random.rand(num_wall_points) * thickness
    theta_wall = np.random.rand(num_wall_points) * 2 * np.pi
    z_wall = np.random.rand(num_wall_points) * height
    x_wall = rr_wall * np.cos(theta_wall)
    zs = rr_wall * np.sin(theta_wall)
    y_wall = z_wall
    z_wall = zs

    rr_cap = np.random.rand(num_cap_points) * radius
    theta_cap = np.random.rand(num_cap_points) * 2 * np.pi
    z_cap = np.full_like(rr_cap, height if cap_position == "top" else 0.0)
    x_cap = rr_cap * np.cos(theta_cap)
    y_cap = z_cap
    z_cap = rr_cap * np.sin(theta_cap)

    x = np.hstack([x_wall, x_cap])
    y = np.hstack([y_wall, y_cap])
    z = np.hstack([z_wall, z_cap])
    return np.vstack([x, y, z]).T