import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim


class NPYVisualizer:
    """使用Polyscope可视化NPY点云数据"""

    def __init__(self):
        """初始化可视化器"""
        # 初始化Polyscope
        if not ps.is_initialized():
            ps.init()

        # 禁用地面平面
        ps.set_ground_plane_mode("none")

        # 存储点云和路径
        self.point_clouds = {}
        self.all_files = {}
        self.current_file = None
        self.current_frame = 0
        self.total_frames = 0
        self.auto_play = False
        self.play_speed = 1
        self.root_dir = "exported_pointclouds"

        # 查找所有NPY文件
        self._find_all_npy_files()

    def _find_all_npy_files(self):
        """查找exported_pointclouds目录中的所有NPY文件"""
        self.all_files = {}

        if not os.path.exists(self.root_dir):
            print(f"错误: 目录 '{self.root_dir}' 不存在")
            return

        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('.npy'):
                    file_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(file_path, self.root_dir)
                    self.all_files[rel_path] = file_path

    def load_npy_file(self, file_path):
        """加载NPY文件"""
        try:
            data = np.load(file_path)

            # 检查数据形状
            if len(data.shape) != 3 or data.shape[2] != 3:
                print(f"警告: 文件 {file_path} 的形状 {data.shape} 不是预期的 (T, N, 3) 形状")
                return None

            print(f"加载文件: {os.path.basename(file_path)}")
            print(f"形状: {data.shape}")
            return data
        except Exception as e:
            print(f"加载 {file_path} 时出错: {e}")
            return None

    def register_point_cloud(self, frame, points, name="点云"):
        """在Polyscope中注册点云"""
        # 清除所有之前的点云
        for cloud_name in list(self.point_clouds.keys()):
            ps.remove_point_cloud(cloud_name)
        self.point_clouds.clear()

        # 注册新的点云
        cloud_name = f"{name}_帧{frame}"
        self.point_clouds[cloud_name] = ps.register_point_cloud(cloud_name, points)

        # 可选: 为点云添加颜色
        color = np.ones((points.shape[0], 3)) * 0.7  # 默认浅灰色
        self.point_clouds[cloud_name].add_color_quantity("默认颜色", color, enabled=True)

        # 可选: 设置点大小
        self.point_clouds[cloud_name].set_radius(0.01)

        return cloud_name

    def show_sequence(self, file_path, frame_index=0):
        """显示序列中的特定帧"""
        data = self.load_npy_file(file_path)
        if data is None:
            return

        self.current_file = file_path
        self.total_frames = data.shape[0]
        self.current_frame = min(frame_index, self.total_frames - 1)

        file_name = os.path.basename(file_path)
        points = data[self.current_frame]
        self.register_point_cloud(self.current_frame, points, file_name)

    def update_frame(self, new_frame):
        """更新显示的帧"""
        if self.current_file is None:
            return

        data = self.load_npy_file(self.current_file)
        if data is None:
            return

        self.current_frame = min(max(0, new_frame), self.total_frames - 1)
        file_name = os.path.basename(self.current_file)
        points = data[self.current_frame]
        self.register_point_cloud(self.current_frame, points, file_name)

    def polyscope_callback(self):
        """Polyscope UI回调"""
        psim.TextUnformatted("NPY文件点云可视化")
        psim.Separator()

        # 文件选择下拉菜单
        if psim.BeginCombo("选择NPY文件", os.path.basename(self.current_file) if self.current_file else "选择文件"):
            for rel_path, file_path in self.all_files.items():
                _, selected = psim.Selectable(rel_path, self.current_file == file_path)
                if selected and self.current_file != file_path:
                    self.show_sequence(file_path, 0)
            psim.EndCombo()

        psim.Separator()

        # 帧控制
        if self.current_file is not None:
            # 显示当前帧信息
            psim.TextUnformatted(f"总帧数: {self.total_frames}")

            # 帧滑块
            changed, new_frame = psim.SliderInt("帧", self.current_frame, 0, self.total_frames - 1)
            if changed:
                self.update_frame(new_frame)

            # 播放控制
            psim.TextUnformatted("播放控制:")
            if self.auto_play:
                if psim.Button("暂停"):
                    self.auto_play = False
            else:
                if psim.Button("播放"):
                    self.auto_play = True

            psim.SameLine()
            if psim.Button("前一帧"):
                self.update_frame(self.current_frame - 1)

            psim.SameLine()
            if psim.Button("后一帧"):
                self.update_frame(self.current_frame + 1)

            # 播放速度
            changed, new_speed = psim.SliderInt("播放速度", self.play_speed, 1, 10)
            if changed:
                self.play_speed = new_speed

            # 如果自动播放，更新帧
            if self.auto_play:
                new_frame = (self.current_frame + self.play_speed) % self.total_frames
                self.update_frame(new_frame)

        psim.Separator()

        # 点云设置
        if self.point_clouds:
            psim.TextUnformatted("点云设置:")

            # 点大小
            for cloud_name, cloud in self.point_clouds.items():
                radius = cloud.get_radius()
                changed, new_radius = psim.SliderFloat("点大小", radius, 0.001, 0.05)
                if changed:
                    cloud.set_radius(new_radius)

        # 刷新文件列表按钮
        if psim.Button("刷新文件列表"):
            self._find_all_npy_files()

    def run(self):
        """运行可视化器"""
        ps.set_user_callback(self.polyscope_callback)
        ps.show()


def main():
    """主函数"""
    print("正在初始化Polyscope可视化器...")
    visualizer = NPYVisualizer()

    # 如果有文件，显示第一个
    if visualizer.all_files:
        first_file = list(visualizer.all_files.values())[0]
        visualizer.show_sequence(first_file)

    print("启动Polyscope界面...")
    visualizer.run()


if __name__ == "__main__":
    main()