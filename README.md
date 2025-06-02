# Joint Analysis

## 🚨 **CRITICAL INFORMATION FOR CODE HANDOVER**

### **📁 Key Directories Overview**

#### **1. `evaluation/` - Primary Experimental Results**
Contains experimental code and results for **Parahome Dataset** analysis:
- **Our Method**: Joint analysis implementation and results
- **Sturm et al. (2012)， Martin-Martin et al. (2022)**: Baseline comparison implementation
- **Dataset Location**: Parahome dataset is located in the `/common/datasets/users/gao/kvil/art_kvil/ma_rui/`
- **📖 Details**: See `evaluation/README.md` for complete experimental setup and reproduction instructions

#### **2. `reproduction_paper/` - Paper Reproductions**
- **Sturm et al. (2012)**: Joint estimation methodology
- **Martin-Martin et al. (2022)**: Joint estimation methodology  
- **Jongenee et al. (2022)**: Estimation of angular velocity
- **📖 Details**: See `reproduction_paper/README.md` for individual paper reproduction guides

#### **3. `joint_analysis/` - Core Implementation**
Our main joint analysis framework and tools

---

## 🎯 **Quick Start for Handover**

### **Step 1: Environment Setup**
```bash
# Clone and setup
git clone git@git.h2t.iar.kit.edu:student-projects/ma-rui-chen/joint_analysis.git
cd joint_analysis

# Install dependencies
pip install -e .
```

### **Step 2: Access Key Experiments**
```bash
# Navigate to evaluation for main results
cd evaluation/
# Follow evaluation/README.md for specific experiments

# Navigate to reproductions for baseline comparisons  
cd reproduction_paper/
# Follow reproduction_paper/README.md for paper reproductions
```

### **Step 3: Dataset Access**
- **Parahome Dataset**: Located in shared `/common/datasets/users/gao/kvil/art_kvil/ma_rui/` directory
- Ensure you have access to the shared dataset location
- Dataset structure and usage detailed in `evaluation/README.md`

---

## 📂 **Core Framework: `joint_analysis/`**

### **Main Components**

#### **Application Entry Points**
```bash
# Primary application
python run_application.py [--output-dir DIR] [--save-plots] [--no-gui]

# Direct module execution
python -m joint_analysis.main
```

#### **Key Modules Structure**
```
joint_analysis/
├── main.py                    # Main application entry
├── core/                      # Core algorithms
│   ├── joint_estimation.py    # Joint type estimation
│   ├── scoring.py             # Scoring functions
│   └── geometry.py            # Geometric utilities
├── viz/                       # Visualization components
│   ├── polyscope_viz.py      # 3D visualization
│   └── gui.py                 # Analysis GUI
├── synthetic/                 # Synthetic data generation
├── data/                      # Data loading utilities
└── __init__.py               # Package interface
```

### **Supported Joint Types**
- **Prismatic**: Linear sliding motion
- **Revolute**: Rotational motion around fixed axis
- **Planar**: Motion constrained to a plane
- **Ball**: Multi-axis rotational motion
- **Screw**: Combined rotation and translation

### **Output Formats**
All joint types return standardized parameter dictionaries:
```python
# Example: Revolute joint
{
    "axis": [x, y, z],           # Rotation axis
    "origin": [x, y, z],         # Rotation center  
    "motion_limit": (min, max)   # Angle range (radians)
}
```


### Noise & Analysis Parameters

```
Neighbor K: Number of neighbor points (default: 10)
  - Number of neighbor points for local motion estimation
  - Increase for stability but decrease precision

Noise Sigma: Noise standard deviation (default: 0.000)
  - Gaussian noise intensity added to synthetic data
  - Used for testing algorithm robustness

```

### Basic Score Parameters

```
col_sigma/col_order: Collinearity score parameters
cop_sigma/cop_order: Coplanarity score parameters
rad_sigma/rad_order: Radius consistency score parameters
zp_sigma/zp_order: Zero pitch score parameters
prob_sigma/prob_order: Probability calculation parameters

```

### Joint-Specific Parameters

```
prismatic_sigma/prismatic_order: Prismatic joint parameters
planar_sigma/planar_order: Planar joint parameters
revolute_sigma/revolute_order: Revolute joint parameters
screw_sigma/screw_order: Screw joint parameters
ball_sigma/ball_order: Ball joint parameters

```

### Filtering Parameters

```
Use SG Filter: Whether to use Savitzky-Golay filtering
SG Window: SG filter window size (default: 10)
SG PolyOrder: SG filter polynomial order (default: 2)

```

---

## 🔧 **For New Developers**

### **Priority Reading Order**
1. **`evaluation/README.md`** - Understand experimental setup and results
2. **`reproduction_paper/README.md`** - Understand baseline implementations  
3. **This document** - Overall framework understanding
4. **`joint_analysis/core/`** - Core algorithm implementation

### **Development Workflow**
1. **Reproduce existing results on Parahome** using `evaluation/` scripts
2. **Run baseline comparisons** using `reproduction_paper/` implementations
3. **Modify core algorithms** in `joint_analysis/core/`
4. **Test with synthetic data** using `joint_analysis/synthetic/`
---
## 📞 **Support and Contact**

### **Original Author**
- **Rui Cheng** - cr1744733299@gmail.com

### **Documentation References**
- **Detailed Experiments**: `evaluation/README.md`
- **Paper Reproductions**: `reproduction_paper/README.md`  
- **Core API Documentation**: See docstrings in `joint_analysis/core/`

### **🙏 Acknowledgments**
- Special thanks to **Jianfeng** for guidance and supervision throughout this project

---