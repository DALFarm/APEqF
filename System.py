# === System.py (APEqF version - Pure Equivariant Implementation) ===
# State and input definitions for APEqF with only equivariant measurement models
# Backward compatibility functions removed for clarity

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

# =============================================================================
# EQUIVARIANT MEASUREMENT MODELS (논문 형식 - Pure Implementation)
# =============================================================================

def measurePos_equivariant(xi: State, y_raw: np.ndarray) -> np.ndarray:
    """
    Equivariant position measurement following paper Eq. (3)
    hp(ξ) = G R_I^T (G π - (G p_I + G R_I I t)) ∈ Np
    
    Args:
        xi: Current state estimate
        y_raw: Raw GNSS position measurement G π
    
    Returns:
        Equivariant output (right-invariant error form)
    """
    R_G_I = xi.T.R().as_matrix()  # G R_I
    p_G_I = xi.T.w().as_vector()  # G p_I 
    t_I = xi.t  # I t
    
    # Ensure y_raw is column vector
    y_raw = np.asarray(y_raw).reshape(3, 1) if y_raw.ndim == 1 else y_raw
    
    # GNSS antenna position: G π = G p_I + G R_I I t
    predicted_gnss_pos = p_G_I.reshape(3, 1) + R_G_I @ t_I.reshape(3, 1)
    
    # Equivariant form: G R_I^T (G π - predicted)
    return R_G_I.T @ (y_raw - predicted_gnss_pos)

def measureVel_equivariant(xi: State, u: 'InputSpace', y_raw: np.ndarray, 
                          subtract_bias: bool = True) -> np.ndarray:
    """
    Equivariant velocity measurement following paper Eq. (4)
    hv(ξ) = G R_I^T (G ν - (G v_I + G R_I I ω∧ I t)) ∈ Nv
    
    Args:
        xi: Current state estimate
        u: Input (for ω)
        y_raw: Raw GNSS velocity measurement G ν
        subtract_bias: Whether to subtract gyro bias
    
    Returns:
        Equivariant output (right-invariant error form)
    """
    R_G_I = xi.T.R().as_matrix()  # G R_I
    v_G_I = xi.T.x().as_vector()  # G v_I
    t_I = xi.t  # I t
    
    # Ensure y_raw is column vector
    y_raw = np.asarray(y_raw).reshape(3, 1) if y_raw.ndim == 1 else y_raw
    
    # Angular velocity (optionally bias-corrected)
    omega = SO3.vee(u.w).reshape(3,1)
    if subtract_bias:
        omega = omega - xi.b[0:3, 0:1]
    
    # GNSS velocity with lever arm effect: G ν = G v_I + G R_I (I ω ∧ I t)
    omega_cross_t = SO3.wedge(omega) @ t_I.reshape(3, 1)
    predicted_gnss_vel = v_G_I.reshape(3, 1) + R_G_I @ omega_cross_t
    
    # Equivariant form: G R_I^T (G ν - predicted)
    return R_G_I.T @ (y_raw - predicted_gnss_vel)

def measureMag_equivariant(xi: State, m_G: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
    """
    Equivariant magnetometer measurement following paper Eq. (2)
    Modified for right-invariant error: uses the innovation approach
    
    Args:
        xi: Current state estimate  
        m_G: Magnetic field vector in global frame
        y_raw: Raw magnetometer measurement in magnetometer frame
        
    Returns:
        Equivariant output (magnetometer frame)
    """
    # Ensure vectors are column format
    m_G = np.asarray(m_G).reshape(3, 1) if m_G.ndim == 1 else m_G
    y_raw = np.asarray(y_raw).reshape(3, 1) if y_raw.ndim == 1 else y_raw
    
    # Predicted measurement: M m = I R_M^T G R_I^T G m
    R_G_I = xi.T.R().as_matrix()  # G R_I
    R_I_M = xi.S.as_matrix()      # I R_M
    
    predicted_mag = R_I_M.T @ R_G_I.T @ m_G
    
    # Innovation in magnetometer frame (equivariant)
    return y_raw - predicted_mag

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
