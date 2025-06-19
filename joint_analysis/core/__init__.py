"""Core functionality for joint analysis."""

from .geometry import (
    point_line_distance, translate_points, rotate_points,
    rotate_points_y, rotate_points_xyz, apply_screw_motion,
    generate_sphere, generate_cylinder, generate_ball_joint_points,
    generate_hollow_cylinder
)
from .scoring import (
    super_gaussian, normalize_vector_torch, compute_basic_scores,
    compute_joint_probability, compute_motion_salience_batch,
    compute_position_average_3d
)
from .joint_estimation import (
    compute_joint_info_all_types, compute_planar_info, compute_ball_info,
    compute_screw_info, compute_prismatic_info, compute_revolute_info,
    calculate_velocity_and_angular_velocity_for_all_frames
)

__all__ = [
    # Geometry
    'point_line_distance', 'translate_points', 'rotate_points',
    'rotate_points_y', 'rotate_points_xyz', 'apply_screw_motion',
    'generate_sphere', 'generate_cylinder', 'generate_ball_joint_points',
    'generate_hollow_cylinder',

    # Scoring
    'super_gaussian', 'normalize_vector_torch', 'compute_basic_scores',
    'compute_joint_probability', 'compute_motion_salience_batch',
    'compute_position_average_3d',

    # Joint estimation
    'compute_joint_info_all_types', 'compute_planar_info', 'compute_ball_info',
    'compute_screw_info', 'compute_prismatic_info', 'compute_revolute_info',
    'calculate_velocity_and_angular_velocity_for_all_frames'
]