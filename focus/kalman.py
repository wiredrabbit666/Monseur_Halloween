import numpy as np


class ConstantVelocityKF:
    """1-D constant-velocity Kalman filter with a variable time step.

    State is [value, rate]. accel_std is how fast the rate is allowed to
    change (units/s^2); meas_std is the measurement noise (units).
    """

    def __init__(self, x0, meas_std, accel_std, rate_std0=1.0):
        self.x = np.array([x0, 0.0])
        self.P = np.diag([meas_std**2, rate_std0**2])
        self.r = meas_std**2
        self.q = accel_std**2

    @property
    def value(self):
        return float(self.x[0])

    @property
    def rate(self):
        return float(self.x[1])

    def predict(self, dt):
        if dt <= 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([0.5 * dt * dt, dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q * np.outer(G, G)

    def innovation(self, z):
        return z - self.x[0]

    def update(self, z, meas_std=None):
        r = self.r if meas_std is None else meas_std**2
        S = self.P[0, 0] + r
        K = self.P[:, 0] / S
        self.x = self.x + K * (z - self.x[0])
        self.P = self.P - np.outer(K, self.P[0, :])

    def shift(self, dx):
        """Offset the value without touching the rate (used after recalibration)."""
        self.x[0] += dx
