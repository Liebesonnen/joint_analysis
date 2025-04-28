# Joint Analysis

This package provides tools for analyzing and estimating the joint types from point cloud data. It can identify different joint types including:

- Prismatic joints
- Revolute joints
- Planar joints
- Ball joints
- Screw joints

## Installation

```bash
pip install -e .
```

## Features

- Extended Kalman Filter for robust motion estimation
- Point cloud processing and joint type classification
- Visualization with Polyscope and DearPyGUI
- Support for both synthetic and real data
- Multiple joint type estimation methods

## Usage

```python
from joint_analysis.main import run_application

# Run the full interactive application
run_application()

# Or import specific components
from joint_analysis.core.joint_estimation import compute_joint_info_all_types
from joint_analysis.core.geometry import rotate_points, point_line_distance

# For advanced usage
```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- PyTorch
- Polyscope
- DearPyGUI

## Examples

See the `examples` directory for demo scripts and usage examples.