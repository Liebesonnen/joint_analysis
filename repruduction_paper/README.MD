from joint_analysis.build.lib.evaluation.unwrap import velocity

# Reproductions

This repository contains reproductions of three influential papers in the field of articulated object analysis, joint estimation, and motion filtering. Each reproduction focuses on different aspects of understanding and tracking articulated objects from sensor data.

## 1. Savitzky-Golay Filter for Angular Velocity Estimation on SO(3)

### Overview
This reproduction implements Savitzky-Golay filtering techniques specifically designed for rotation matrices and angular velocity estimation on the Special Orthogonal group SO(3).

### Key Components

**Files:**
- `so3_functions.py`: Core mathematical functions for SO(3) operations
- `savitzky_golay_so3.py`: Main implementation and demonstration

**Implemented Features:**
- **SO(3) Mathematical Operations:**
  - `hat()`: Converts 3D vector to skew-symmetric matrix
  - `vee()`: Extracts vector from skew-symmetric matrix  
  - `expSO3()`: Exponential map from so(3) to SO(3)
  - `dexpSO3()`: Right trivialized tangent differential
  - `DdexpSO3()`: Directional derivative of dexp

- **Savitzky-Golay Filtering:**
  - `sgolayfiltSO3()`: Main filtering function for rotation matrix sequences
  - Polynomial fitting in the Lie algebra
  - Simultaneous estimation of rotation, angular velocity, and angular acceleration

- **Comparison Methods:**
  - Finite differencing for baseline comparison
  - Error analysis between analytical solutions and estimates
  - Noise robustness evaluation

**Key Innovations:**
- Extends classical Savitzky-Golay filtering to the non-Euclidean SO(3) manifold
- Preserves geometric structure of rotation matrices
- Provides smooth estimates with reduced noise sensitivity compared to finite differencing

### Usage
```python
# Generate synthetic rotation data with noise
cd sg_filter_for_angular-velocity
python savizky_golay_so3.py
```

## 2. Sturm et al.（2012）
### Overview
This reproduction implements a framework for detecting and modeling articulated objects, with focus on joint parameter estimation and tracking.

### Key Components

**Files:**
- `reproduction_sturm.py`: Main implementation with Polyscope visualization
- `door.txt`, `drawer.txt`: Real sensor data examples
- `tracking-1.log`, `tracking-2.log`: Tracking trajectory logs

**Implemented Joint Models:**
- **Rigid Joints:** Fixed connections between objects
- **Prismatic Joints:** Linear sliding motion along an axis
- **Revolute Joints:** Rotational motion around an axis  
- **Gaussian Process Models:** Data-driven joint modeling

**Core Algorithms:**
- **RANSAC-based Joint Detection:** Robust estimation of joint parameters
- **Model Selection:** BIC-based selection among joint hypotheses
- **Parameter Estimation:** 
  - Axis estimation for prismatic/revolute joints
  - Origin point estimation for revolute joints
  - Motion limit detection

**Evaluation Metrics:**
- Position error between estimated and ground truth joint axes
- Angular error between estimated and true joint orientations
- Model likelihood and BIC scores

### Features
- **Real-time Visualization:** Interactive 3D visualization with Polyscope
- **Multiple Object Types:** Doors, drawers, and other articulated objects
- **Ground Truth Comparison:** Built-in synthetic data with known parameters
- **Noise Robustness:** Configurable noise levels for evaluation

### Usage
```python
cd ..
cd stuerm
python reproduction_sturm.py
```

## 3. Martin-Martin et al.（2022)

### Overview
This reproduction implements a multi-level recursive estimation framework for real-time joint classification.


### Key Components

**Files:**
- `reproduction_martin.py`: Complete framework implementation with visualization

**Three-Level Architecture:**

1. **Feature Motion Estimation (Level 1):**
   - Tracks 3D features across point cloud sequences
   - Estimates feature-level motion using rigid body velocities
   - Handles feature-to-rigid-body assignments

2. **Rigid Body Motion Estimation (Level 2):**
   - Estimates pose and velocity for detected rigid bodies
   - RANSAC-based rigid body detection
   - Integration of kinematic model predictions

3. **Kinematic Model Estimation (Level 3):**
   - Joint type classification (rigid, prismatic, revolute, disconnected)
   - Parameter estimation for each joint type
   - Model probability computation and selection

**Advanced Features:**
- **SE(3) and Twist Representations:** Proper handling of 6D motion
- **Joint Filters:** Probabilistic filters for each joint type
- **Bidirectional Information Flow:** Top-down predictions and bottom-up updates
- **Real-time Processing:** Designed for online operation

**Supported Joint Types:**
- **Rigid:** Fixed connections with motion constraints
- **Prismatic:** Linear motion with axis and limit estimation
- **Revolute:** Rotational motion with axis, origin, and limit estimation  
- **Disconnected:** Independent motion detection

### Usage
```python
cd ..
cd martin_martin
python reproduction_martin.py
```
```