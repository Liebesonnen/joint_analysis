from robot_utils.viz.polyscope import PolyscopeUtils, ps, psim, register_point_cloud, draw_frame_3d
import open3d as o3d
import numpy as np
import polyscope as ps


file_base = "/common/homes/all/uksqc_chen/projects/control/ParaHome/data/scan/refrigerator/simplified/base.obj"
file_part1 = "/common/homes/all/uksqc_chen/projects/control/ParaHome/data/scan/refrigerator/simplified/part1.obj"
file_part2 = "/common/homes/all/uksqc_chen/projects/control/ParaHome/data/scan/refrigerator/simplified/part2.obj"


pu = PolyscopeUtils()
draw_frame_3d(np.zeros(6), scale=1, radius=0.01)

# Read the OBJ file using Open3D
mesh = o3d.io.read_triangle_mesh(file_base)  # Replace with your file path
mesh_part1 = o3d.io.read_triangle_mesh(file_part1)  # Replace with your file path
mesh_part2 = o3d.io.read_triangle_mesh(file_part2)  # Replace with your file path

joint_info = {
    "part1": {
      "axis": [
        0.529561429306864,
        0.053152541384484144,
        0.8466046892941487
      ],
      "pivot": [
        0.585,
        0.228,
        0.2958
      ]
    },
    "part2": {
      "axis": [
        0.529561429306864,
        0.053152541384484144,
        0.8466046892941487
      ],
      "pivot": [
        0.585,
        0.228,
        0.2958
      ]
    }
  }

# Convert Open3D mesh to numpy arrays for Polyscope
def viz_part(mesh, name):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    ps.register_surface_mesh(name, vertices, faces, smooth_shade=True)

viz_part(mesh, "mesh")
viz_part(mesh_part1, "mesh_part1")
viz_part(mesh_part2, "mesh_part2")

def viz_axis(joint_dict, joint_name):
    axis = np.array(joint_dict.get("axis"))
    anchor = np.array(joint_dict.get("pivot"))
    points = np.linspace(-1, 1, 100).reshape(-1, 1) * axis[None] + anchor
    register_point_cloud(
        f"{joint_name}_axis", points, radius=0.01
    )

viz_axis(joint_info["part1"], "part1")
viz_axis(joint_info["part2"], "part2")

# Register the mesh with Polyscope

# Show the visualization
ps.show()