import numpy as np
from scipy.signal import savgol_filter


def compute_smoothed_euler_velocity(euler_angles, window_length=11, polyorder=2, dt=0.1):
    """
    Computes smoothed angular velocities from a trajectory of Euler angles using the SG filter.

    Args:
        euler_angles (np.ndarray): Array of shape (N, 3), each column is roll, pitch, yaw (in radians).
        window_length (int): Window size for the SG filter (must be odd and >= polyorder+2).
        polyorder (int): Polynomial order for the SG filter.
        dt (float): Time step between samples (in seconds).

    Returns:
        np.ndarray: Angular velocity array of shape (N, 3).
    """
    euler_angles = np.unwrap(euler_angles, axis=0)  # unwrap along time axis to avoid discontinuities
    smoothed_angle = savgol_filter(euler_angles, window_length, polyorder, axis=0, mode='interp')
    velocity = savgol_filter(euler_angles, window_length, polyorder, axis=0, deriv=1, delta=dt, mode='interp')
    return smoothed_angle, velocity


# Generate a noisy Euler angle trajectory with discontinuities
N = 100
t = np.linspace(0, 10, N)
euler_angles = np.stack([
    np.sin(t),                     # roll
    np.sin(2 * t),                 # pitch
    np.unwrap(np.angle(np.exp(1j * t)))  # yaw with wrapping
], axis=1)

# Add artificial jumps to simulate wrap-around at pi/-pi
euler_angles[:, 2] = (euler_angles[:, 2] + np.pi) % (2 * np.pi) - np.pi  # simulate wrap

# Compute angular velocities
smoothed_angle, velocity = compute_smoothed_euler_velocity(euler_angles, window_length=11, polyorder=2, dt=t[1] - t[0])

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
ax = axs[0]
ax.plot(t, euler_angles[:, 0], "--", label='raw_roll')
ax.plot(t, euler_angles[:, 1], "--", label='raw_pitch')
ax.plot(t, euler_angles[:, 2], "--", label='raw_yaw')
ax.plot(t, smoothed_angle[:, 0], label='roll')
ax.plot(t, smoothed_angle[:, 1], label='pitch')
ax.plot(t, smoothed_angle[:, 2], label='yaw')

ax = axs[1]
ax.plot(t, velocity[:, 0], label='v_roll')
ax.plot(t, velocity[:, 1], label='v_pitch')
ax.plot(t, velocity[:, 2], label='v_yaw')
plt.legend()
plt.show()

