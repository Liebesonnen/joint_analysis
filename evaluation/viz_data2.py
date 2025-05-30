import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from io import BytesIO
import os
from datetime import datetime
from math import atan2, asin

# Polyscope utilities
from robot_utils.viz.polyscope import (
    PolyscopeUtils,
    ps,
    psim,
    register_point_cloud,
    draw_frame_3d,
)


class Viz:
    """Visualisation and kinematic analysis of a moving point‑cloud with
    Euler‑angle smoothing/derivatives extracted via Savitzky–Golay filtering
    (helper taken from the user‑supplied snippet)."""

    # ------------------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------------------
    def __init__(self):
        # Polyscope helpers ------------------------------------------------
        self.pu = PolyscopeUtils()

        # Indices controlled through the UI --------------------------------
        self.t, self.t_changed = 0, False  # time‑step index
        self.idx_point, self.idx_point_changed = 0, False  # point index

        # ------------------------------------------------------------------
        #  DATA LOADING
        # ------------------------------------------------------------------
        # Shape: (T, N, 3)
        self.d = np.load("./demo_data/revolute.npy")
        # Add light Gaussian noise so it is not perfectly rigid (for demo)
        self.d += np.random.randn(*self.d.shape) * 1e-2
        self.T, self.N, _ = self.d.shape

        # ------------------------------------------------------------------
        #  CONSTANTS & BUFFERS
        # ------------------------------------------------------------------
        self.dt_mean = 0.1  # s   – average sampling period (assumed uniform)
        self.num_neighbors = 50  # neighbourhood size for local SVD
        self.output_dir = "visualization_output"
        os.makedirs(self.output_dir, exist_ok=True)

        # ------------------------------------------------------------------
        #  PRE‑SMOOTH THE POSITION TRAJECTORY (unchanged)
        # ------------------------------------------------------------------
        self.d_filter = savgol_filter(
            x=self.d,
            window_length=21,
            polyorder=2,
            deriv=0,
            axis=0,
            delta=self.dt_mean,
        )
        self.dv_filter = savgol_filter(
            x=self.d,
            window_length=21,
            polyorder=2,
            deriv=1,
            axis=0,
            delta=self.dt_mean,
        )

        # ------------------------------------------------------------------
        #  ANGULAR KINEMATICS FROM EULER ANGLES
        # ------------------------------------------------------------------
        (
            self.euler_angles_raw,
            self.euler_angles_filtered,
            self.euler_rates,
            self.angular_velocity_filtered,
        ) = self.calculate_euler_kinematics()

        # ------------------------------------------------------------------
        #  POLYSCOPE VISUAL SET‑UP
        # ------------------------------------------------------------------
        draw_frame_3d(np.zeros(6), label="origin", scale=0.1)
        register_point_cloud("door", self.d_filter[self.t], radius=0.01, enabled=True)
        self.pu.reset_bbox_from_pcl_list([self.d_filter[self.t]])

        ps.set_user_callback(self.callback)
        self.plot_image()  # initial plot drawn once at start
        ps.show()

    # ------------------------------------------------------------------
    #  LOW‑LEVEL HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def find_neighbors(points, k):
        """Return indices of the *k* nearest spatial neighbours for every point"""
        n = len(points)
        dists = np.linalg.norm(points[None, :, :] - points[:, None, :], axis=-1)
        return np.argsort(dists, axis=1)[:, 1 : k + 1]  # (n, k)

    @staticmethod
    def compute_rotation_matrix(src, dst):
        """Rigid alignment *dst* ← *src* via SVD; returns rotation matrix R"""
        src_c = src - src.mean(0)
        dst_c = dst - dst.mean(0)
        u, _, vt = np.linalg.svd(src_c.T @ dst_c)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:  # reflection handling
            vt[-1] *= -1
            r = vt.T @ u.T
        return r

    @staticmethod
    def rot_to_euler_zyx(r):
        """Convert 3×3 rotation matrix to Z‑Y‑X Euler angles (roll‑pitch‑yaw)."""
        r20 = np.clip(-r[2, 0], -1.0, 1.0)
        pitch = asin(r20)
        cp = np.cos(pitch)
        if abs(cp) < 1e-6:  # gimbal lock
            roll = 0.0
            yaw = atan2(-r[0, 1], r[1, 1])
        else:
            roll = atan2(r[2, 1], r[2, 2])
            yaw = atan2(r[1, 0], r[0, 0])
        return np.array([roll, pitch, yaw])

    # ------------------------------------------------------------------
    #  USER‑SUPPLIED HELPER (SG‑based smoothing & differentiation)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_smoothed_euler_velocity(
        euler_angles: np.ndarray,
        window_length: int = 11,
        polyorder: int = 2,
        dt: float = 0.1,
    ):
        """Smooth Euler‑angle sequence and compute its first derivative.

        Args:
            euler_angles: (T, 3) array [roll, pitch, yaw].
            window_length: Odd window length for SG filter.
            polyorder: Polynomial order.
            dt: sampling period (s).
        Returns:
            smoothed angles (T,3), velocities d(angle)/dt (T,3)
        """
        # Ensure odd window length >= polyorder+2
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, polyorder + 3)

        ea_unwrapped = np.unwrap(euler_angles, axis=0)
        smooth = savgol_filter(
            ea_unwrapped,
            window_length,
            polyorder,
            axis=0,
            mode="interp",
        )
        vel = savgol_filter(
            ea_unwrapped,
            window_length,
            polyorder,
            axis=0,
            deriv=1,
            delta=dt,
            mode="interp",
        )
        return smooth, vel

    # ------------------------------------------------------------------
    #  EULER‑BASED ANGULAR KINEMATICS
    # ------------------------------------------------------------------
    def calculate_euler_kinematics(self):
        """Compute raw Euler angles, SG‑smoothed angles, SG‑based rates, and
        treat those rates as instantaneous angular velocity ω."""
        T, N, _ = self.d_filter.shape
        # Prepare containers
        eul_raw = np.zeros((T, N, 3))
        eul_filt = np.zeros_like(eul_raw)
        eul_rate = np.zeros_like(eul_raw)
        ang_vel = np.zeros_like(eul_raw)

        # Prepare neighbourhood indices once (based on first frame)
        neigh_idx = self.find_neighbors(self.d_filter[0], self.num_neighbors)  # (N,k)

        # Process each point independently (parallel‑friendly, but serial here)
        for i in range(N):
            nbrs0 = self.d_filter[0][neigh_idx[i]]

            # 1) Build rotation matrices against frame0 → Euler angles sequence
            for t in range(T):
                nbrs_t = self.d_filter[t][neigh_idx[i]]
                r = self.compute_rotation_matrix(nbrs0, nbrs_t)
                eul_raw[t, i] = self.rot_to_euler_zyx(r)

            # 2) Smooth & differentiate along time using supplied helper
            window_len = 21 if T >= 21 else (T // 2) * 2 + 1
            smooth, vel = self.compute_smoothed_euler_velocity(
                eul_raw[:, i],
                window_length=window_len,
                polyorder=2,
                dt=self.dt_mean,
            )
            eul_filt[:, i] = smooth
            eul_rate[:, i] = vel
            ang_vel[:, i] = vel  # in small‑angle assumption ω ≈ angle‑rates

        return eul_raw, eul_filt, eul_rate, ang_vel

    # -----------------------------------------------------------

    # ------------------------------------------------------------------
    #  PLOTTING
    # ------------------------------------------------------------------
    def plot_image(self):
        """Draw the 4‑panel summary plot and push it into Polyscope."""
        t = np.arange(self.T) * self.dt_mean
        t_ω = t  # same length as Euler‑rate arrays (we tolerate last sample)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), dpi=100)

        # --------------------------------------------------------------
        # 1) Position ---------------------------------------------------
        ax1.plot(t, self.d[:, self.idx_point, 0], "--", c="red", label="X raw")
        ax1.plot(t, self.d[:, self.idx_point, 1], "--", c="green", label="Y raw")
        ax1.plot(t, self.d[:, self.idx_point, 2], "--", c="blue", label="Z raw")
        ax1.plot(t, self.d_filter[:, self.idx_point, 0], "-", c="darkred", label="X filt")
        ax1.plot(t, self.d_filter[:, self.idx_point, 1], "-", c="darkgreen", label="Y filt")
        ax1.plot(t, self.d_filter[:, self.idx_point, 2], "-", c="darkblue", label="Z filt")
        ax1.set_xlim(0, 20)
        ax1.set_ylim(-1, 10)
        ax1.set_ylabel("position [m]")
        ax1.set_title(f"Position of point #{self.idx_point}")
        ax1.grid(True, ls="--", alpha=0.5)
        ax1.legend(loc="upper right")

        # --------------------------------------------------------------
        # 2) Linear velocity -------------------------------------------
        vel = self.dv_filter[:, self.idx_point]
        vel_mag = np.linalg.norm(vel, axis=1)
        ax2.plot(t, vel[:, 0], "-", c="red", label="Vx")
        ax2.plot(t, vel[:, 1], "-", c="green", label="Vy")
        ax2.plot(t, vel[:, 2], "-", c="blue", label="Vz")
        ax2.plot(t, vel_mag, "-", c="black", label="|v|")
        ax2.set_xlim(0, 20)
        ax2.set_ylim(-1, 1)
        ax2.set_ylabel("velocity [m/s]")
        ax2.set_title("Linear velocity")
        ax2.grid(True, ls="--", alpha=0.5)
        ax2.legend(loc="upper right")

        # --------------------------------------------------------------
        # 3) Smoothed Euler angles -------------------------------------
        eul = self.euler_angles_filtered[:, self.idx_point]  # (T,3)
        labels = ["roll φ", "pitch θ", "yaw ψ"]
        colors = ["red", "green", "blue"]
        for k in range(3):
            ax3.plot(t, eul[:, k], "-", c=colors[k], label=labels[k] + " (rad)")
        ax3_deg = ax3.twinx()
        for k in range(3):
            ax3_deg.plot(t, np.degrees(eul[:, k]), "--", c=colors[k])
        ax3.set_xlim(0, 20)
        ax3.set_ylabel("angle [rad]")
        ax3_deg.set_ylabel("angle [deg]")
        ax3.set_title("Smoothed Euler angles (ZYX)")
        ax3.grid(True, ls="--", alpha=0.5)
        ax3.legend(loc="upper right")

        # --------------------------------------------------------------
        # 4) Angular velocity (Euler‑rates) -----------------------------
        ω = self.angular_velocity_filtered[:, self.idx_point]
        ω_mag = np.linalg.norm(ω, axis=1)
        for k in range(3):
            ax4.plot(t_ω, ω[:, k], "-", c=colors[k], label=f"ω{['x','y','z'][k]}")
        ax4.plot(t_ω, ω_mag, "-", c="black", label="|ω|")
        ax4.set_xlim(0, 20)
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_ylabel("angular velocity [rad/s]")
        ax4.set_title("Angular velocity from Euler‑rates")
        ax4.grid(True, ls="--", alpha=0.5)
        ax4.legend(loc="upper right")

        plt.tight_layout()

        # Push to Polyscope as image -----------------------------------
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        img = np.array(Image.open(buf).convert("RGB")) / 255.0
        ps.add_color_image_quantity("plot", img, enabled=True)
        plt.close(fig)

    # ------------------------------------------------------------------
    #  SAVE CURRENT PLOT (unchanged apart from new arrays) --------------
    # ------------------------------------------------------------------
    def save_current_plot(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{self.output_dir}/point_{self.idx_point}_t{self.t}_{ts}.png"
        self.plot_image()  # redraw with any new selections
        ps.save_screenshot(path)  # polyscope screenshot helper
        print(f"Plot saved to {path}")
        return path

    # ------------------------------------------------------------------
    #  POLYSCOPE CALLBACK / UI -----------------------------------------
    # ------------------------------------------------------------------
    def callback(self):
        psim.Text("Point Cloud Motion Analysis")
        self.t_changed, self.t = psim.SliderInt("Time Frame", self.t, 0, self.T - 1)
        self.idx_point_changed, self.idx_point = psim.SliderInt(
            "Point Index", self.idx_point, 0, self.N - 1
        )
        changed_nb, self.num_neighbors = psim.SliderInt(
            "Neighbors for SVD", self.num_neighbors, 5, 50
        )

        if self.t_changed:
            register_point_cloud("door", self.d_filter[self.t], radius=0.01, enabled=True)
        if self.idx_point_changed:
            self.plot_image()

        if psim.Button("Save Current Plot"):
            path = self.save_current_plot()
            psim.Text(f"Saved to: {path}")

        if changed_nb or psim.Button("Recalculate Euler Kinematics"):
            (
                self.euler_angles_raw,
                self.euler_angles_filtered,
                self.euler_rates,
                self.angular_velocity_filtered,
            ) = self.calculate_euler_kinematics()
            self.plot_image()
            psim.Text("Recalculated with new neighbourhood size.")

        # --------------------------------------------------------------
        #  Debug / info pane -------------------------------------------
        if psim.TreeNode("Data Information"):
            psim.Text(f"Data shape: {self.d.shape}")
            psim.Text(f"Frames: {self.T}, points: {self.N}")
            psim.Text(f"Δt = {self.dt_mean} s")
            pos = self.d_filter[self.t, self.idx_point]
            psim.Text(f"Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            vel = self.dv_filter[self.t, self.idx_point]
            psim.Text(f"Velocity: vx={vel[0]:.3f}, vy={vel[1]:.3f}, vz={vel[2]:.3f}")
            eul = self.euler_angles_filtered[self.t, self.idx_point]
            psim.Text(f"Euler (rad): roll={eul[0]:.3f}, pitch={eul[1]:.3f}, yaw={eul[2]:.3f}")
            ω = self.angular_velocity_filtered[self.t, self.idx_point]
            psim.Text(f"ω: wx={ω[0]:.3f}, wy={ω[1]:.3f}, wz={ω[2]:.3f}")
            psim.TreePop()


# ----------------------------------------------------------------------
#  MAIN ENTRY POINT -----------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    Viz()
