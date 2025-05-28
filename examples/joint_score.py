import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from io import BytesIO
from robot_utils.viz.polyscope import PolyscopeUtils, ps, psim, register_point_cloud, draw_frame_3d


class Viz:
    def __init__(self):
        self.pu = PolyscopeUtils()
        self.t, self.t_changed = 0, False
        self.idx_point, self.idx_point_changed = 0, False
        self.d = np.load("/media/gao/dataset/kvil/ma_rui/synthetic_dataset/prismaric_joint/Prismatic_Door_0.npy")
        # self.d += np.random.randn(*self.d.shape) * 0.01
        ic(self.d.shape)
        self.T, self.N, _ = self.d.shape

        dt_mean = 0.1
        self.d_filter = savgol_filter(
            x=self.d, window_length=5, polyorder=2, deriv=0, axis=0, delta=dt_mean
        )
        self.dv_filter = savgol_filter(
            x=self.d, window_length=6, polyorder=2, deriv=1, axis=0, delta=dt_mean
        )

        draw_frame_3d(np.zeros(6), label="origin", scale=0.1)
        register_point_cloud(
            "door", self.d[self.t], radius=0.01, enabled=True
        )
        self.pu.reset_bbox_from_pcl_list([self.d[self.t]])
        ps.set_user_callback(self.callback)
        ps.show()

    def plot_image(self):
        t = np.arange(self.T)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        plt.plot(t, self.d[:, self.idx_point, 0], "--", c="red")
        plt.plot(t, self.d[:, self.idx_point, 1], "--", c="green")
        plt.plot(t, self.d[:, self.idx_point, 2], "--", c="blue")
        plt.plot(t, self.d_filter[:, self.idx_point, 0], "-", c="red")
        plt.plot(t, self.d_filter[:, self.idx_point, 1], "-", c="green")
        plt.plot(t, self.d_filter[:, self.idx_point, 2], "-", c="blue")

        plt.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="red")
        plt.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="green")
        plt.plot(t, self.dv_filter[:, self.idx_point, 0], "-", c="blue")

        ax.set_xlim(0, 55)
        ax.set_ylim(-1, 5)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close(fig)

        # Load buffer into PIL Image, then convert to NumPy array
        image = Image.open(buf).convert('RGB')
        rgb_array = np.array(image)

        ps.add_color_image_quantity("plot", rgb_array/255.0, enabled=True)

    def callback(self):
        psim.Text("Viz")
        self.t_changed, self.t = psim.SliderInt("time", self.t, 0, self.T-1)
        self.idx_point_changed, self.idx_point = psim.SliderInt("idx_point", self.idx_point, 0, self.N-1)

        if self.t_changed:
            register_point_cloud(
                "door", self.d_filter[self.t], radius=0.01, enabled=True
            )
        if self.idx_point_changed:
            self.plot_image()


if __name__ == "__main__":
    viz = Viz()