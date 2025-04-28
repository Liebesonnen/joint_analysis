"""
Extended Kalman Filter for 3D state estimation in joint analysis.
"""

import numpy as np


class ExtendedKalmanFilter3D:
    """
    Extended Kalman Filter for 3D position, velocity and angular velocity estimation.

    State vector (9 dimensions):
        [px, py, pz, vx, vy, vz, wx, wy, wz]

    where:
        p* - position components
        v* - velocity components
        w* - angular velocity components
    """

    def __init__(self, dt, Q, R, x0, P0):
        """
        Initialize the Extended Kalman Filter.

        Args:
            dt (float): Time step size
            Q (ndarray): Process noise covariance matrix (9x9)
            R (ndarray): Measurement noise covariance matrix (9x9)
            x0 (ndarray): Initial state vector (9,)
            P0 (ndarray): Initial state covariance matrix (9x9)
        """
        self.dt = dt
        self.Q = Q.copy()  # 9x9
        self.R = R.copy()  # 9x9
        self.x = x0.copy()  # (9,)
        self.P = P0.copy()  # 9x9
        self.n = 9

    def predict(self):
        """
        Prediction step of the Kalman filter.
        """
        px, py, pz, vx, vy, vz, wx, wy, wz = self.x
        dt = self.dt

        # 1) State prediction
        px_new = px + vx * dt
        py_new = py + vy * dt
        pz_new = pz + vz * dt
        vx_new = vx
        vy_new = vy
        vz_new = vz
        wx_new = wx
        wy_new = wy
        wz_new = wz
        self.x = np.array([px_new, py_new, pz_new, vx_new, vy_new, vz_new, wx_new, wy_new, wz_new])

        # 2) Jacobian F (9x9)
        F = np.eye(self.n)
        F[0, 3] = dt  # ∂px/∂vx
        F[1, 4] = dt  # ∂py/∂vy
        F[2, 5] = dt  # ∂pz/∂vz

        # 3) Covariance prediction
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        Update step of the Kalman filter.

        Args:
            z (ndarray): Measurement vector [px_meas, py_meas, pz_meas, vx_meas, vy_meas, vz_meas, wx_meas, wy_meas, wz_meas]

        Returns:
            ndarray: Updated state vector
        """
        # z = [px_meas, py_meas, pz_meas, vx_meas, vy_meas, vz_meas, wx_meas, wy_meas, wz_meas]
        z_pred = self.x
        y_res = z - z_pred

        # Observation matrix (in this case, identity since we measure the full state)
        H = np.eye(self.n)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y_res

        # Covariance update
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P

        return self.x