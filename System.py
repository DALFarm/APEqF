# === System.py (APEqF version) ===
# State and input definitions for APEqF (SE23 navigation + se3 biases + lever arm + magnetometer extrinsic)

import numpy as np
from dataclasses import dataclass
from pylie import SE23, SO3, SE3

# gravity in homogeneous SE23 matrices (5x5); g enters acceleration column
G = np.zeros((5, 5))
G[2, 3:4] = -9.81

@dataclass
class State:
    T: SE23 = SE23.identity()          # (R, v, p)
    b: np.ndarray = np.zeros((6, 1))   # [b_w; b_a] (se3 vector)
    t: np.ndarray = np.zeros((3, 1))   # GNSS lever arm in IMU/body
    S: SO3 = SO3.identity()            # magnetometer extrinsic rotation

    @staticmethod
    def random():
        return State(SE23.exp(np.random.randn(9,1)),
                     np.random.randn(6,1),
                     np.random.randn(3,1),
                     SO3.exp(np.random.randn(3,1)))

    def vec(self) -> np.ndarray:
        return np.vstack((
            self.T.R().as_euler().reshape(3,1),
            self.T.x().as_vector(),
            self.T.w().as_vector(),
            self.b,
            self.t,
            self.S.R().as_euler().reshape(3,1)
        ))

# data is expected as (R, p, v, bw, ba, RM, t)
def stateFromData(d) -> "State":
    result = State()
    result.T = SE23(d[0], d[2], d[1])  # (R, v, p)
    result.b = np.vstack((d[3], d[4]))
    result.S = SO3(d[5])
    result.t = d[6]
    return result

@dataclass
class InputSpace:
    # IMU
    w: np.ndarray = np.zeros((3, 3))   # so3 wedge
    a: np.ndarray = np.zeros((3, 1))
    # process for calibration
    tau: np.ndarray = np.zeros((6, 1)) # se3 bias random walk
    mu: np.ndarray = np.zeros((3, 1))  # lever-arm drift
    wM: np.ndarray = np.zeros((3, 1))  # SO3 extrinsic drift (axis-angle)

    @staticmethod
    def random():
        U = InputSpace()
        U.w = SO3.wedge(np.random.randn(3, 1))
        U.a = np.random.randn(3, 1)
        U.tau = np.zeros((6,1))
        U.mu = np.zeros((3,1))
        U.wM = np.random.randn(3,1)
        return U

    def as_vector(self) -> np.ndarray:
        v = np.zeros((21,1))
        v[0:3, 0:1] = SO3.vee(self.w)
        v[3:6, 0:1] = self.a
        v[6:9, 0:1] = np.zeros((3,1))  # μ enters as virtual (0) in nav lift
        v[9:15, 0:1] = self.tau
        v[15:18, 0:1] = self.mu
        v[18:21, 0:1] = self.wM
        return v

    def as_W_mat(self) -> np.ndarray:
        W = np.zeros((5, 5))
        W[0:3, 0:3] = self.w
        W[0:3, 3:4] = self.a
        W[0:3, 4:5] = np.zeros((3,1))  # μ does not enter propagation directly
        return W

    def as_W_vec(self) -> np.ndarray:
        v = np.zeros((9,1))
        w_vee = SO3.vee(self.w)
        if w_vee.ndim == 1:
            w_vee = w_vee.reshape(-1, 1)
        v[0:3] = w_vee
        v[3:6] = self.a
        v[6:9] = np.zeros((3,1))  # μ part zeroed in propagation
        return v

# Xi_0
xi_0 = State()

# Measurement functions (GNSS position incl. lever arm; optional bias)
def measurePos(xi: State) -> np.ndarray:
    # p + R * t  (GNSS antenna position)
    return xi.T.w().as_vector() + xi.T.R().as_matrix() @ xi.t

def measurePosAndBias(xi: State) -> np.ndarray:
    # Return [p_GNSS; b_mu] (here b_mu not used; keep zeros placeholder)
    b_mu = np.zeros((3,1))
    return np.vstack((measurePos(xi), b_mu))



# Magnetometer measurement (magnetic north in {G} observed in magnetometer frame {M})
# y_m = S^T * R^T * m_G
def measureMag(xi: State, m_G: 'np.ndarray') -> 'np.ndarray':
    if m_G is None:
        raise ValueError("m_G (magnetic field in {G}) must be provided")
    return xi.S.inv().as_matrix() @ xi.T.R().inv().as_matrix() @ m_G


# GNSS velocity measurement (antenna point) in world frame {G}
# y_v = v + R * (omega_body × t)
# omega_body is derived from IMU input (u.w) optionally minus gyro bias b_w
def measureVel(xi: State, u: 'InputSpace', subtract_bias: bool = True) -> 'np.ndarray':
    from pylie import SO3
    R = xi.T.R().as_matrix()
    v = xi.T.x().as_vector()
    t = xi.t
    omega = SO3.vee(u.w).reshape(3,1)
    if subtract_bias:
        omega = omega - xi.b[0:3, 0:1]
    omega_cross_t = SO3.wedge(omega) @ t
    return v + R @ omega_cross_t
# get input space from vector form
def input_from_vector(vec) -> "InputSpace":
    if not isinstance(vec, np.ndarray):
        raise TypeError
    if vec.shape not in [(18,1), (21,1), (6,1)]:
        raise ValueError
    U = InputSpace()
    U.w = SO3.wedge(vec[0:3, 0:1])
    U.a = vec[3:6, 0:1]
    if vec.shape[0] >= 9:
        # assume μ=0 in propagation
        pass
    if vec.shape[0] >= 15:
        U.tau = vec[9:15, 0:1]
    if vec.shape[0] == 21:
        U.mu = vec[15:18, 0:1]
        U.wM = vec[18:21, 0:1]
    return U
