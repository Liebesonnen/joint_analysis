"""
Joint Analysis package for estimating and analyzing joint types from point cloud data.
"""

from .core import (
    compute_joint_info_all_types,
    compute_basic_scores,
    compute_joint_probability_new,
    compute_motion_salience_batch,
    point_line_distance,
    translate_points,
    rotate_points,
    rotate_points_y,
    rotate_points_xyz,
    apply_screw_motion,
    generate_sphere,
    generate_cylinder,
    generate_ball_joint_points,
    generate_hollow_cylinder
)

from .viz import PolyscopeVisualizer, JointAnalysisGUI, PlotSaver
from .synthetic import SyntheticJointGenerator
from .data import (
    load_numpy_sequence,
    load_pytorch_data,
    load_all_sequences_from_directory,
    subsample_sequence,
    load_real_datasets
)

from .main import JointAnalysisApp, run_application

__version__ = "0.1.0"

__all__ = [
    # Core functionality
    'compute_joint_info_all_types',
    'compute_basic_scores',
    'compute_joint_probability_new',
    'compute_motion_salience_batch',

    # Geometry utilities
    'point_line_distance',
    'translate_points',
    'rotate_points',
    'rotate_points_y',
    'rotate_points_xyz',
    'apply_screw_motion',
    'generate_sphere',
    'generate_cylinder',
    'generate_ball_joint_points',
    'generate_hollow_cylinder',

    # Visualization
    'PolyscopeVisualizer',
    'JointAnalysisGUI',
    'PlotSaver',

    # Data generation and loading
    'SyntheticJointGenerator',
    'load_numpy_sequence',
    'load_pytorch_data',
    'load_all_sequences_from_directory',
    'subsample_sequence',
    'load_real_datasets',

    # Application
    'JointAnalysisApp',
    'run_application'
]