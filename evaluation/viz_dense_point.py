# import numpy as np
# import polyscope as ps
# import polyscope.imgui as psim
# from itertools import cycle
#
# # ---------- 1. 读入多份数据 ----------
# paths = [
#     "/common/homes/all/uksqc_chen/projects/control/joint_analysis/examples/demo_data/s1_refrigerator_part2_3180_3240.npy",
#     "/common/homes/all/uksqc_chen/projects/control/joint_analysis/examples/demo_data/s1_refrigerator_part1_3180_3240.npy",
#     "/common/homes/all/uksqc_chen/projects/control/joint_analysis/examples/demo_data/s1_refrigerator_base_3180_3240.npy",
# ]
# trajs = [np.load(p) for p in paths]                      # list[(F, N, 3)]
# assert all(t.ndim == 3 and t.shape[2] == 3 for t in trajs)
#
# F_list = [t.shape[0] for t in trajs]     # 每套数据的帧数
# N_list = [t.shape[1] for t in trajs]     # 每套数据的点数
#
# # ---------- 2. 交互状态 ----------
# dataset_idx = 0       # 当前激活的数据集
# frame_idx   = 0       # 当前帧
# track_idx   = 0       # 当前跟踪点
#
# # ---------- 3. Polyscope 初始化 ----------
# ps.init()
# ps.set_up_dir("z_up")
#
# point_rad = 0.004     # 普通点半径
# track_rad = 0.012     # 高亮点半径
# color_cycle = cycle([
#     (1, 0.1, 0.1), (0.1, 0.7, 0.2), (0.2, 0.4, 1),
#     (1, 0.6, 0.1), (0.8, 0.1, 0.8)
# ])
#
# cloud_all  = []       # 每份数据：整云（灰）
# cloud_sel  = []       # 每份数据：当前帧的单点（彩色）
# curve      = []       # 每份数据：该点的整条轨迹（彩色）
#
# for idx, (traj, col) in enumerate(zip(trajs, color_cycle)):
#     # 3-1  整云（灰）
#     pc = ps.register_point_cloud(f"all {idx}", traj[0], enabled=False)
#     pc.set_radius(point_rad, relative=True)
#     pc.set_color((0.6, 0.6, 0.6))
#     cloud_all.append(pc)
#
#     # 3-2  当前帧的高亮点
#     sel = ps.register_point_cloud(f"sel {idx}", traj[0, 0:1], enabled=False)
#     sel.set_radius(track_rad, relative=True)
#     sel.set_color(col)
#     cloud_sel.append(sel)
#
#     # 3-3  整条轨迹折线
#     edges = np.column_stack([np.arange(traj.shape[0] - 1),
#                              np.arange(1, traj.shape[0])])
#     cv = ps.register_curve_network(f"traj {idx}", traj[:, 0, :], edges,
#                                    enabled=False)
#     cv.set_radius(point_rad * 0.6, relative=True)
#     cv.set_color(col)
#     curve.append(cv)
#
# # ---------- 4. 辅助函数 ----------
# def activate_dataset(i: int):
#     """只激活第 i 套数据，并同步 frame/track 索引到合法范围"""
#     global frame_idx, track_idx
#     for k in range(len(trajs)):
#         is_on = (k == i)
#         cloud_all[k].set_enabled(is_on)
#         cloud_sel[k].set_enabled(is_on)
#         curve[k].set_enabled(is_on)
#
#     # 防越界
#     frame_idx = min(frame_idx, F_list[i] - 1)
#     track_idx = min(track_idx, N_list[i] - 1)
#
#     # 用当前索引刷新显示
#     cloud_all[i].update_point_positions(trajs[i][frame_idx])
#     cloud_sel[i].update_point_positions(
#         trajs[i][frame_idx, track_idx:track_idx + 1])
#     curve[i].update_node_positions(trajs[i][:, track_idx, :])
#
# activate_dataset(0)   # 默认打开第 0 套
#
# # ---------- 5. ImGui 回调 ----------
# def ui_callback():
#     global dataset_idx, frame_idx, track_idx
#
#     # 数据集选择
#     changed, ds_tmp = psim.SliderInt("Dataset", dataset_idx,
#                                      0, len(trajs) - 1)
#     if changed:
#         dataset_idx = ds_tmp
#         activate_dataset(dataset_idx)
#
#     F = F_list[dataset_idx]
#     N = N_list[dataset_idx]
#
#     # 帧滑杆
#     changed, f_tmp = psim.SliderInt("Frame", frame_idx, 0, F - 1)
#     if changed:
#         frame_idx = f_tmp
#         cloud_all[dataset_idx].update_point_positions(
#             trajs[dataset_idx][frame_idx])
#         cloud_sel[dataset_idx].update_point_positions(
#             trajs[dataset_idx][frame_idx, track_idx:track_idx + 1])
#
#     # 点滑杆
#     changed, p_tmp = psim.SliderInt("Point", track_idx, 0, N - 1)
#     if changed:
#         track_idx = p_tmp
#         curve[dataset_idx].update_node_positions(
#             trajs[dataset_idx][:, track_idx, :])
#         cloud_sel[dataset_idx].update_point_positions(
#             trajs[dataset_idx][frame_idx, track_idx:track_idx + 1])
#
#     psim.TextUnformatted(
#         f"Dataset {dataset_idx}: frame {frame_idx}/{F - 1}, "
#         f"point {track_idx}/{N - 1}"
#     )
#
# ps.set_user_callback(ui_callback)
# ps.show()

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from itertools import cycle
from scipy.signal import savgol_filter

# ---------- 1. 读入多份数据 ----------
paths = [
    # "/common/homes/all/uksqc_chen/projects/control/joint_analysis/examples/demo_data/s2_drawer_part1_1770_1950.npy",
    # "/common/homes/all/uksqc_chen/projects/control/joint_analysis/examples/demo_data/s2_drawer_part2_1770_1950.npy",
    # "/common/homes/all/uksqc_chen/projects/control/joint_analysis/examples/demo_data/s2_drawer_base_1770_1950.npy",
    "/common/homes/all/uksqc_chen/projects/control/joint_analysis/examples/demo_data/s1_refrigerator_part2_3180_3240.npy",
    "/common/homes/all/uksqc_chen/projects/control/joint_analysis/examples/demo_data/s1_refrigerator_part1_3180_3240.npy",
    "/common/homes/all/uksqc_chen/projects/control/joint_analysis/examples/demo_data/s1_refrigerator_base_3180_3240.npy"
]
trajs = [np.load(p) for p in paths]  # list[(F, N, 3)]
assert all(t.ndim == 3 and t.shape[2] == 3 for t in trajs)

F_list = [t.shape[0] for t in trajs]  # 每套数据的帧数
N_list = [t.shape[1] for t in trajs]  # 每套数据的点数

# ---------- 2. 速度计算参数 ----------
dt_mean = 0.1  # 平均时间步长
num_neighbors = 30  # SVD计算的邻居数量
window_length = min(21, min(F_list))  # SG滤波器窗口长度
if window_length % 2 == 0:  # 确保是奇数
    window_length -= 1
window_length = max(5, window_length)  # 至少5个点

# ---------- 3. 计算滤波后的数据和速度 ----------
trajs_filtered = []
linear_velocities = []
angular_velocities_raw = []
angular_velocities_filtered = []


def find_neighbors(points, num_neighbors):
    """Find neighbors for each point"""
    N = points.shape[0]
    num_neighbors = min(num_neighbors, N - 1)  # 防止邻居数超过点数
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        dist_matrix[i] = np.sqrt(np.sum((points - points[i]) ** 2, axis=1))

    neighbors = np.zeros((N, num_neighbors), dtype=int)
    for i in range(N):
        indices = np.argsort(dist_matrix[i])[1:num_neighbors + 1]
        neighbors[i] = indices

    return neighbors


def compute_rotation_matrix(src_points, dst_points):
    """Compute rotation matrix between two point sets using SVD"""
    src_center = np.mean(src_points, axis=0)
    dst_center = np.mean(dst_points, axis=0)

    src_centered = src_points - src_center
    dst_centered = dst_points - dst_center

    H = np.dot(src_centered.T, dst_centered)
    U, _, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R


def rotation_matrix_to_angular_velocity(R, dt):
    """Extract angular velocity from rotation matrix"""
    # Ensure R is a valid rotation matrix
    U, _, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)

    # Compute rotation angle
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # If angle is too small, return zero vector
    if abs(theta) < 1e-6:
        return np.zeros(3)

    # Calculate rotation axis
    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-6:
        return np.zeros(3)

    # Extract angular velocity vector from skew-symmetric part
    W = (R - R.T) / (2 * sin_theta)
    omega = np.zeros(3)
    omega[0] = W[2, 1]
    omega[1] = W[0, 2]
    omega[2] = W[1, 0]

    # Angular velocity = axis * angle / time
    omega = omega * theta / dt

    return omega


def calculate_angular_velocity(traj_filtered, dt_mean):
    """Compute angular velocity using SVD on filtered point cloud data"""
    F, N, _ = traj_filtered.shape

    # Initialize arrays
    angular_velocity = np.zeros((F - 1, N, 3))

    # For each pair of consecutive frames
    for t in range(F - 1):
        # Current and next frame points
        current_points = traj_filtered[t]
        next_points = traj_filtered[t + 1]

        # Find neighbors for all points
        neighbors = find_neighbors(current_points, num_neighbors)

        # Compute angular velocity for each point
        for i in range(N):
            # Get current point and its neighbors
            src_neighborhood = current_points[neighbors[i]]
            dst_neighborhood = next_points[neighbors[i]]

            # Compute rotation matrix
            R = compute_rotation_matrix(src_neighborhood, dst_neighborhood)

            # Extract angular velocity
            omega = rotation_matrix_to_angular_velocity(R, dt_mean)

            # Store result
            angular_velocity[t, i] = omega

    # Filter angular velocity
    angular_velocity_filtered = np.zeros_like(angular_velocity)

    if F - 1 >= window_length:
        for i in range(N):
            for dim in range(3):
                angular_velocity_filtered[:, i, dim] = savgol_filter(
                    angular_velocity[:, i, dim],
                    window_length=min(window_length, F - 1),
                    polyorder=2
                )
    else:
        angular_velocity_filtered = angular_velocity.copy()

    return angular_velocity, angular_velocity_filtered


# 处理每个数据集
for traj in trajs:
    F, N, _ = traj.shape

    # 应用 Savitzky-Golay 滤波器平滑数据
    if F >= window_length:
        traj_filtered = savgol_filter(
            x=traj, window_length=window_length, polyorder=2, deriv=0, axis=0, delta=dt_mean
        )
        # 计算线速度 (一阶导数)
        linear_vel = savgol_filter(
            x=traj, window_length=window_length, polyorder=2, deriv=1, axis=0, delta=dt_mean
        )
    else:
        # 数据太短，不进行滤波
        traj_filtered = traj.copy()
        # 简单差分计算速度
        linear_vel = np.zeros_like(traj)
        linear_vel[1:] = (traj[1:] - traj[:-1]) / dt_mean

    trajs_filtered.append(traj_filtered)
    linear_velocities.append(linear_vel)

    # 计算角速度
    angular_vel_raw, angular_vel_filtered = calculate_angular_velocity(traj_filtered, dt_mean)
    angular_velocities_raw.append(angular_vel_raw)
    angular_velocities_filtered.append(angular_vel_filtered)

# ---------- 4. 交互状态 ----------
dataset_idx = 0  # 当前激活的数据集
frame_idx = 0  # 当前帧
track_idx = 0  # 当前跟踪点
show_linear_vel = True  # 是否显示线速度
show_angular_vel = True  # 是否显示角速度
use_filtered = True  # 使用滤波后的角速度
linear_vel_scale = 10.0  # 线速度向量缩放因子
angular_vel_scale = 20.0  # 角速度向量缩放因子

# ---------- 5. Polyscope 初始化 ----------
ps.init()
ps.set_up_dir("z_up")

point_rad = 0.004  # 普通点半径
track_rad = 0.012  # 高亮点半径
color_cycle = cycle([
    (1, 0.1, 0.1), (0.1, 0.7, 0.2), (0.2, 0.4, 1),
    (1, 0.6, 0.1), (0.8, 0.1, 0.8)
])

cloud_all = []  # 每份数据：整云（灰）
cloud_sel = []  # 每份数据：当前帧的单点（彩色）
curve = []  # 每份数据：该点的整条轨迹（彩色）
linear_vel_point = []  # 选定点的线速度向量
angular_vel_point = []  # 选定点的角速度向量

for idx, (traj, col) in enumerate(zip(trajs, color_cycle)):
    # 5-1  整云（灰）
    pc = ps.register_point_cloud(f"all {idx}", traj[0], enabled=False)
    pc.set_radius(point_rad, relative=True)
    pc.set_color((0.6, 0.6, 0.6))
    cloud_all.append(pc)

    # 5-2  当前帧的高亮点
    sel = ps.register_point_cloud(f"sel {idx}", traj[0, 0:1], enabled=False)
    sel.set_radius(track_rad, relative=True)
    sel.set_color(col)
    cloud_sel.append(sel)

    # 5-3  整条轨迹折线
    edges = np.column_stack([np.arange(traj.shape[0] - 1),
                             np.arange(1, traj.shape[0])])
    cv = ps.register_curve_network(f"traj {idx}", traj[:, 0, :], edges,
                                   enabled=False)
    cv.set_radius(point_rad * 0.6, relative=True)
    cv.set_color(col)
    curve.append(cv)

    # 5-4  选定点的线速度向量（只有一个点）
    lin_vel_pt = ps.register_point_cloud(f"linear_vel_point {idx}", traj[0, 0:1], enabled=False)
    lin_vel_pt.add_vector_quantity("velocity", np.zeros((1, 3)),
                                   vectortype='ambient', enabled=True, radius=0.01,
                                   color=(0.2, 0.8, 0.2))  # 绿色表示线速度
    linear_vel_point.append(lin_vel_pt)

    # 5-5  选定点的角速度向量（只有一个点）
    ang_vel_pt = ps.register_point_cloud(f"angular_vel_point {idx}", traj[0, 0:1], enabled=False)
    ang_vel_pt.add_vector_quantity("angular_velocity", np.zeros((1, 3)),
                                   vectortype='ambient', enabled=True, radius=0.01,
                                   color=(0.8, 0.2, 0.2))  # 红色表示角速度
    angular_vel_point.append(ang_vel_pt)


# ---------- 6. 辅助函数 ----------
def activate_dataset(i: int):
    """只激活第 i 套数据，并同步 frame/track 索引到合法范围"""
    global frame_idx, track_idx
    for k in range(len(trajs)):
        is_on = (k == i)
        cloud_all[k].set_enabled(is_on)
        cloud_sel[k].set_enabled(is_on)
        curve[k].set_enabled(is_on)
        linear_vel_point[k].set_enabled(is_on and show_linear_vel)
        angular_vel_point[k].set_enabled(is_on and show_angular_vel)

    # 防越界
    frame_idx = min(frame_idx, F_list[i] - 1)
    track_idx = min(track_idx, N_list[i] - 1)

    # 用当前索引刷新显示
    update_visualization(i)


def update_visualization(i: int):
    """更新第 i 套数据的可视化"""
    # 更新点云位置
    cloud_all[i].update_point_positions(trajs_filtered[i][frame_idx])
    cloud_sel[i].update_point_positions(
        trajs_filtered[i][frame_idx, track_idx:track_idx + 1])
    curve[i].update_node_positions(trajs_filtered[i][:, track_idx, :])

    # 更新选定点的线速度向量
    if show_linear_vel:
        # 更新位置到选定点
        point_pos = trajs_filtered[i][frame_idx, track_idx:track_idx + 1]
        linear_vel_point[i].update_point_positions(point_pos)
        # 获取该点的速度
        vel_data = linear_velocities[i][frame_idx, track_idx:track_idx + 1]
        # 应用缩放因子
        linear_vel_point[i].add_vector_quantity("velocity", vel_data * linear_vel_scale,
                                                vectortype='ambient', enabled=True, radius=0.01,
                                                color=(0.2, 0.8, 0.2))

    # 更新选定点的角速度向量
    if show_angular_vel and frame_idx < F_list[i] - 1:
        # 更新位置到选定点
        point_pos = trajs_filtered[i][frame_idx, track_idx:track_idx + 1]
        angular_vel_point[i].update_point_positions(point_pos)
        # 获取该点的角速度
        if use_filtered:
            ang_vel_data = angular_velocities_filtered[i][frame_idx, track_idx:track_idx + 1]
        else:
            ang_vel_data = angular_velocities_raw[i][frame_idx, track_idx:track_idx + 1]
        # 应用缩放因子
        angular_vel_point[i].add_vector_quantity("angular_velocity", ang_vel_data * angular_vel_scale,
                                                 vectortype='ambient', enabled=True, radius=0.01,
                                                 color=(0.8, 0.2, 0.2))


activate_dataset(0)  # 默认打开第 0 套


# ---------- 7. ImGui 回调 ----------
def ui_callback():
    global dataset_idx, frame_idx, track_idx, show_linear_vel, show_angular_vel, use_filtered
    global linear_vel_scale, angular_vel_scale

    psim.TextUnformatted("=== Point Cloud Motion Analysis ===")

    # 数据集选择
    changed, ds_tmp = psim.SliderInt("Dataset", dataset_idx,
                                     0, len(trajs) - 1)
    if changed:
        dataset_idx = ds_tmp
        activate_dataset(dataset_idx)

    F = F_list[dataset_idx]
    N = N_list[dataset_idx]

    # 帧滑杆
    changed, f_tmp = psim.SliderInt("Frame", frame_idx, 0, F - 1)
    if changed:
        frame_idx = f_tmp
        update_visualization(dataset_idx)

    # 点滑杆
    changed, p_tmp = psim.SliderInt("Point", track_idx, 0, N - 1)
    if changed:
        track_idx = p_tmp
        update_visualization(dataset_idx)

    # 速度显示选项
    psim.Separator()
    psim.TextUnformatted("Velocity Visualization:")

    changed, show_linear_vel = psim.Checkbox("Show Linear Velocity (Green)", show_linear_vel)
    if changed:
        linear_vel_point[dataset_idx].set_enabled(show_linear_vel)

    # 线速度缩放滑块
    if show_linear_vel:
        changed, linear_vel_scale = psim.SliderFloat("Linear Velocity Scale",
                                                     linear_vel_scale, 0.1, 50.0)
        if changed:
            update_visualization(dataset_idx)

    changed, show_angular_vel = psim.Checkbox("Show Angular Velocity (Red)", show_angular_vel)
    if changed:
        angular_vel_point[dataset_idx].set_enabled(show_angular_vel)

    # 角速度缩放滑块
    if show_angular_vel:
        changed, angular_vel_scale = psim.SliderFloat("Angular Velocity Scale",
                                                      angular_vel_scale, 0.1, 100.0)
        if changed:
            update_visualization(dataset_idx)

    changed, use_filtered = psim.Checkbox("Use Filtered Angular Velocity", use_filtered)
    if changed:
        update_visualization(dataset_idx)

    # 显示当前状态信息
    psim.Separator()
    psim.TextUnformatted(
        f"Dataset {dataset_idx}: frame {frame_idx}/{F - 1}, "
        f"point {track_idx}/{N - 1}"
    )

    # 显示当前点的速度信息
    if psim.TreeNode("Current Point Velocity"):
        pos = trajs_filtered[dataset_idx][frame_idx, track_idx]
        psim.Text(f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        lin_vel = linear_velocities[dataset_idx][frame_idx, track_idx]
        lin_vel_mag = np.linalg.norm(lin_vel)
        psim.Text(f"Linear Velocity: [{lin_vel[0]:.3f}, {lin_vel[1]:.3f}, {lin_vel[2]:.3f}]")
        psim.Text(f"Linear Velocity Magnitude: {lin_vel_mag:.3f} m/s")

        if frame_idx < F - 1:
            if use_filtered:
                ang_vel = angular_velocities_filtered[dataset_idx][frame_idx, track_idx]
            else:
                ang_vel = angular_velocities_raw[dataset_idx][frame_idx, track_idx]
            ang_vel_mag = np.linalg.norm(ang_vel)
            psim.Text(f"Angular Velocity: [{ang_vel[0]:.3f}, {ang_vel[1]:.3f}, {ang_vel[2]:.3f}]")
            psim.Text(f"Angular Velocity Magnitude: {ang_vel_mag:.3f} rad/s ({np.degrees(ang_vel_mag):.1f}°/s)")

        psim.TreePop()

    # 显示计算参数
    if psim.TreeNode("Computation Parameters"):
        psim.Text(f"Time step: {dt_mean} s")
        psim.Text(f"Number of neighbors for SVD: {num_neighbors}")
        psim.Text(f"SG filter window length: {window_length}")
        psim.TreePop()


ps.set_user_callback(ui_callback)
ps.show()