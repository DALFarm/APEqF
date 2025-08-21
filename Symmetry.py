# === Symmetry.py (APEqF version) ===
# Implements the symmetry group (SE23 × se(3) × R^3 × SO(3)) used for APEqF:
#   - C ∈ SE23 acts on navigation state (R, v, p)
#   - γ ∈ se(3) models bias translation (bw, ba)
#   - δ ∈ R^3 is the GNSS lever arm
#   - E ∈ SO(3) is the magnetometer extrinsic rotation
#
# Derived following the structure of the TG-EqF reference implementation (the "*T" files),
# then extended to include calibration states per APEqF.

import numpy as np
from dataclasses import dataclass
from pylie import SE23, SO3, SE3  # NOTE: pylie provides SE3/SE23/SO3 with wedge/vee

# -------------------------
# Utilities
# -------------------------
def numericalDifferential(f, x) -> np.ndarray:
    if isinstance(x, float):
        x = np.reshape([x], (1, 1))
    h = 1e-6
    fx = f(x)
    n = fx.shape[0]
    m = x.shape[0]
    Df = np.zeros((n, m))
    for j in range(m):
        ej = np.zeros((m, 1))
        ej[j, 0] = 1.0
        Df[:, j:j+1] = (f(x + h * ej) - f(x - h * ej)) / (2 * h)
    return Df

def blockDiag(A : np.ndarray, B : np.ndarray) -> np.ndarray:
    return np.block([[A, np.zeros((A.shape[0], B.shape[1]))],
                     [np.zeros((B.shape[0], A.shape[1])), B]])

# -------------------------
# Group helpers
# -------------------------
def Gamma(C: SE23) -> SO3:
    """Γ: SE23 → SO3, extract rotation."""
    return C.R()

def chi(C: SE23) -> SE3:
    """χ: SE23 → SE3, map (R,v,p) ↦ (R,v)."""
    R = C.R()
    v = C.x()  # first translational slot in SE23 is velocity
    # pylie SE3 constructor from rotation and translation (v)
    return SE3.from_R_t(R, v.as_vector())

# Coordinate selection
exponential_coords = False
fake_exponential = False

# -------------------------
# Import system types
# -------------------------
from System import State, InputSpace, xi_0

# -------------------------
# Left Jacobians (SO3/SE3/SE23/R3)
# -------------------------
def J1(so3vec: np.ndarray) -> np.ndarray:
    if not isinstance(so3vec, np.ndarray):
        raise TypeError
    if so3vec.shape != (3, 1):
        raise ValueError("so3vec must be (3,1)")
    angle = float(np.linalg.norm(so3vec))
    if np.isclose(angle, 0.0):
        return np.eye(3) + 0.5 * SO3.wedge(so3vec)
    axis = so3vec / angle
    s = np.sin(angle) / angle
    c = (1 - np.cos(angle)) / angle
    return s * np.eye(3) + (1 - s) * (axis @ axis.T) + c * SO3.wedge(axis)

def Q1(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError
    if arr.shape != (6, 1):
        raise ValueError("arr must be (6,1)")
    phi = arr[0:3, 0:1]
    rho = arr[3:6, 0:1]
    rx = SO3.wedge(rho)
    px = SO3.wedge(phi)
    ph = float(np.linalg.norm(phi))
    ph2 = ph * ph
    ph3 = ph2 * ph
    ph4 = ph3 * ph
    ph5 = ph4 * ph
    cph = np.cos(ph)
    sph = np.sin(ph)
    m1 = 0.5
    m2 = (ph - sph) / ph3 if ph3 != 0 else 1/6.0
    m3 = (0.5 * ph2 + cph - 1.0) / ph4 if ph4 != 0 else 0.0
    m4 = (ph - 1.5 * sph + 0.5 * ph * cph) / ph5 if ph5 != 0 else 0.0
    t1 = rx
    t2 = px @ rx + rx @ px + px @ rx @ px
    t3 = px @ px @ rx + rx @ px @ px - 3.0 * px @ rx @ px
    t4 = px @ rx @ px @ px + px @ px @ rx @ px
    return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4

def SO3LeftJacobian(so3vec: np.ndarray) -> np.ndarray:
    return J1(so3vec)

def SE3LeftJacobian(se3vec: np.ndarray) -> np.ndarray:
    if not isinstance(se3vec, np.ndarray):
        raise TypeError
    if se3vec.shape != (6, 1):
        raise ValueError
    phi = se3vec[0:3, 0:1]
    if np.isclose(np.linalg.norm(phi), 0.0):
        return np.eye(6) + 0.5 * SE3.adjoint(se3vec)
    SO3_JL = J1(phi)
    J = np.zeros((6, 6))
    J[0:3, 0:3] = SO3_JL
    J[3:6, 3:6] = SO3_JL
    J[3:6, 0:3] = Q1(se3vec)
    return J

def SE23LeftJacobian(se23vec: np.ndarray) -> np.ndarray:
    if not isinstance(se23vec, np.ndarray):
        raise TypeError
    if se23vec.shape != (9, 1):
        raise ValueError
    phi = se23vec[0:3, 0:1]
    rho = se23vec[3:6, 0:1]
    psi = se23vec[6:9, 0:1]
    if np.isclose(np.linalg.norm(phi), 0.0):
        return np.eye(9) + 0.5 * SE23.adjoint(se23vec)
    SO3_JL = J1(phi)
    J = np.zeros((9, 9))
    J[0:3, 0:3] = SO3_JL
    J[3:6, 3:6] = SO3_JL
    J[6:9, 6:9] = SO3_JL
    J[3:6, 0:3] = Q1(np.vstack((phi, rho)))
    J[6:9, 0:3] = Q1(np.vstack((phi, psi)))
    return J

def R3LeftJacobian(so3vec: np.ndarray) -> np.ndarray:
    """Left Jacobian for the R^3 calibration part affected by rotation only."""
    return J1(so3vec)

# From (R,v,p) to (0,0,v)
def f_10(mat: np.ndarray) -> np.ndarray:
    if mat.shape != (5, 5):
        raise ValueError
    f = np.zeros((5, 5))
    f[0:3, 4:5] = mat[0:3, 3:4]
    return f


def embed_se3_wedge_to_se23(gamma4: np.ndarray) -> np.ndarray:
    """Embed a 4x4 se(3) wedge into the top-left block of a 5x5 SE23 homogeneous matrix."""
    if gamma4.shape != (4,4):
        raise ValueError("gamma must be a 4x4 se(3) wedge")
    G5 = np.zeros((5,5))
    G5[0:3,0:3] = gamma4[0:3,0:3]
    G5[0:3,3:4] = gamma4[0:3,3:4]
    return G5


# adjoint matrix for the whole group tangent (used for curvature corr.)
def grp_adj(l: np.ndarray) -> np.ndarray:
    if l.shape != (21, 1):
        raise ValueError
    ad = np.zeros((21, 21))
    # SE23 on SE23
    ad[0:9, 0:9] = SE23.adjoint(l[0:9, :])
    # se3 on se3
    ad[9:15, 9:15] = SE3.adjoint(l[0:6, :])  # rotation/vel part only
    # R3 on R3 (through SO3) - use wedge matrix
    try:
        ad[15:18, 15:18] = SO3.wedge(l[0:3, :])
    except:
        # Fallback: identity
        ad[15:18, 15:18] = np.eye(3)
    # SO3 on SO3 - use wedge matrix
    try:
        ad[18:21, 18:21] = SO3.wedge(l[18:21, :])
    except:
        # Fallback: identity
        ad[18:21, 18:21] = np.eye(3)
    return ad

# -------------------------
# Symmetry group definition
# -------------------------
@dataclass
class SymGroup:
    C: SE23 = SE23.identity()
    gamma: np.ndarray = np.zeros((4, 4))   # se(3) wedge (4x4)
    delta: np.ndarray = np.zeros((3, 1))   # R^3 vector (lever arm)
    E: SO3 = SO3.identity()                # SO(3) extrinsic rotation

    def __mul__(self, other) -> 'SymGroup':
        assert isinstance(other, SymGroup)
        C = self.C * other.C
        g5 = self.C.as_matrix() @ embed_se3_wedge_to_se23(other.gamma) @ self.C.inv().as_matrix()
        g  = self.gamma + g5[0:4,0:4]
        d = self.delta + Gamma(self.C).as_matrix() @ other.delta
        E = self.E * other.E
        return SymGroup(C, g, d, E)

    @staticmethod
    def random() -> 'SymGroup':
        return SymGroup(SE23.exp(np.random.randn(9, 1)),
                        SE3.wedge(np.random.randn(6, 1)),
                        np.random.randn(3, 1),
                        SO3.exp(np.random.randn(3, 1)))

    @staticmethod
    def identity():
        return SymGroup(SE23.identity(), np.zeros((4, 4)), np.zeros((3, 1)), SO3.identity())

    def inv(self) -> 'SymGroup':
        Cinv = self.C.inv()
        g5 = - Cinv.as_matrix() @ embed_se3_wedge_to_se23(self.gamma) @ self.C.as_matrix()
        g  = g5[0:4,0:4]
        d = -Gamma(Cinv).as_matrix() @ self.delta
        return SymGroup(Cinv, g, d, self.E.inv())

    @staticmethod
    def exp(groupArray: np.ndarray) -> 'SymGroup':
        if not isinstance(groupArray, np.ndarray):
            raise TypeError
        if groupArray.shape != (21, 1):
            raise ValueError("exp expects (21,1) vector")
        result = SymGroup()
        # navigation
        xi = groupArray[0:9, 0:1]
        result.C = SE23.exp(xi)
        # se3 bias part uses χ(C) = (R,v)
        se3_vec = groupArray[9:15, 0:1]
        if fake_exponential:
            result.gamma = SE3.wedge(se3_vec)
        else:
            result.gamma = SE3.wedge(SE3LeftJacobian(xi[0:6, :]) @ se3_vec)
        # R3 calibration part (lever arm) affected by rotation only
        dvec = groupArray[15:18, 0:1]
        if fake_exponential:
            result.delta = dvec
        else:
            result.delta = R3LeftJacobian(xi[0:3, :]) @ dvec
        # SO3 extrinsic part
        result.E = SO3.exp(groupArray[18:21, 0:1])
        return result

    @staticmethod
    def log(groupElement: 'SymGroup') -> np.ndarray:
        if not isinstance(groupElement, SymGroup):
            raise TypeError
        result = np.zeros((21, 1))
        result[0:9, 0:1] = np.asarray(SE23.log(groupElement.C)).reshape(9,1)
        # invert left Jacobians to recover coords
        xi = result[0:9, 0:1]
        result[18:21, 0:1] = np.asarray(SO3.log(groupElement.E)).reshape(3,1)
        if fake_exponential:
            result[9:15, 0:1] = np.asarray(SE3.vee(groupElement.gamma)).reshape(6,1)
            result[15:18, 0:1] = groupElement.delta
        else:
            result[9:15, 0:1] = np.linalg.pinv(SE3LeftJacobian(xi[0:6, :])) @ np.asarray(SE3.vee(groupElement.gamma)).reshape(6,1)
            result[15:18, 0:1] = np.linalg.pinv(R3LeftJacobian(xi[0:3, :])) @ groupElement.delta
        return result

    def vec(self):
        return np.vstack((
            self.C.R().as_euler().reshape(3, 1),
            self.C.x().as_vector(),
            self.C.w().as_vector(),
            SE3.vee(self.gamma),
            self.delta.reshape(3,1),
            self.E.R().as_euler().reshape(3, 1)
        ))

# -------------------------
# State & input actions
# -------------------------
def stateAction(X: SymGroup, xi: State) -> State:
    # T' = T * C
    Tnew = xi.T * X.C
    # b' = SE3.vee( C^{-1} (SE3.wedge(b) - γ) C )
    B5 = embed_se3_wedge_to_se23(SE3.wedge(np.vstack((xi.b[0:3], xi.b[3:6]))))
    G5 = embed_se3_wedge_to_se23(X.gamma)
    tmp5 = X.C.inv().as_matrix() @ (B5 - G5) @ X.C.as_matrix()
    bnew = SE3.vee(tmp5[0:4,0:4])
    # t' = Γ(C)^{-1}(t - δ)
    tnew = Gamma(X.C).inv().as_matrix() @ (xi.t - X.delta)
    # S' = S * E
    Snew = xi.S * X.E
    return State(Tnew, bnew, tnew, Snew)

def velocityAction(X: SymGroup, U: InputSpace) -> InputSpace:
    # Transform IMU & calibration-rate inputs
    result_vec = np.zeros((21, 1))
    # nav part - simplified approach
    try:
        # Get the SE23 matrix and compute vee
        C_inv_matrix = X.C.inv().as_matrix()
        W_matrix = U.as_W_mat()
        C_matrix = X.C.as_matrix()
        
        # Compute the transformation
        transformed = C_inv_matrix @ W_matrix @ C_matrix
        
        # Extract the vee part safely
        vee_result = SE23.vee(transformed)
        if vee_result.ndim == 1:
            vee_result = vee_result.reshape(-1, 1)
        result_vec[0:9, 0:1] = vee_result
    except Exception as e:
        print(f"Warning: velocityAction nav part failed: {e}")
        # Fallback: use identity transformation
        result_vec[0:9, 0:1] = U.as_W_vec()
    # bias random walk rates (se3) transformed
    try:
        tau_wedge = SE3.wedge(U.tau)
        # Convert SE3 to SE23 for matrix operations
        tau_wedge_5x5 = embed_se3_wedge_to_se23(tau_wedge)
        tau_trf = X.C.inv().as_matrix() @ tau_wedge_5x5 @ X.C.as_matrix()
        # Extract the 4x4 SE3 part and convert back
        tau_trf_4x4 = tau_trf[0:4, 0:4]
        vee_result = SE3.vee(tau_trf_4x4)
        if vee_result.ndim == 1:
            vee_result = vee_result.reshape(-1, 1)
        result_vec[9:15, 0:1] = vee_result
    except Exception as e:
        print(f"Warning: velocityAction bias part failed: {e}")
        # Fallback: use original tau
        result_vec[9:15, 0:1] = U.tau
    # lever-arm drift rate (R3)
    result_vec[15:18, 0:1] = Gamma(X.C).inv().as_matrix() @ U.mu
    # extrinsic SO3 drift (3)
    try:
        wM_wedge = SO3.wedge(U.wM)
        E_inv_matrix = X.E.inv().as_matrix()
        E_matrix = X.E.as_matrix()
        transformed = E_inv_matrix @ wM_wedge @ E_matrix
        vee_result = SO3.vee(transformed)
        if vee_result.ndim == 1:
            vee_result = vee_result.reshape(-1, 1)
        result_vec[18:21, 0:1] = vee_result
    except Exception as e:
        print(f"Warning: velocityAction extrinsic part failed: {e}")
        # Fallback: use original wM
        result_vec[18:21, 0:1] = U.wM
    # Pack back
    from System import input_from_vector
    return input_from_vector(result_vec)


def local_coords(e: State) -> np.ndarray:
    # theta_e in group s.t. stateAction(theta_e, xi_0) = e
    theta = SymGroup()
    theta.C = xi_0.T.inv() * e.T
    if exponential_coords:
        # b, t linear in exp coords
        beta = np.vstack((e.b[0:3], e.b[3:6]))
        eps = np.vstack((np.asarray(SE23.log(theta.C)).reshape(9,1), beta, e.t - xi_0.t, np.asarray(SO3.log(e.S * xi_0.S.inv()).reshape(3,1))))
    else:
        # Use 5x5 embedding for conjugation, then project back to 4x4 se(3)
        B0_5 = embed_se3_wedge_to_se23(SE3.wedge(xi_0.b))
        Be_5 = embed_se3_wedge_to_se23(SE3.wedge(e.b))
        C5   = theta.C.as_matrix()
        beta5 = B0_5 - C5 @ Be_5 @ np.linalg.inv(C5)
        beta  = beta5[0:4, 0:4]
        delta = (xi_0.t - Gamma(theta.C).as_matrix() @ e.t).reshape(3,1)
        Eeps  = SO3.log(xi_0.S.inv() * e.S)
        theta.gamma = beta
        theta.delta = delta
        theta.E = SO3.exp(Eeps)
        eps = SymGroup.log(theta)
    return eps


def local_coords_inv(eps: np.ndarray) -> "State":
    if eps.shape not in [(18,1), (21,1)]:
        raise ValueError("eps must be (18,1) or (21,1)")
    if eps.shape == (18,1):
        # backwards compatibility (no calibration parts)
        inv_theta = State()
        inv_theta.T = xi_0.T * SE23.exp(eps[0:9, 0:1])
        inv_theta.b = xi_0.b + eps[9:15, 0:1]
        inv_theta.t = xi_0.t
        inv_theta.S = xi_0.S
        return inv_theta
    else:
        inv_theta = stateAction(SymGroup.exp(eps), xi_0)
        return inv_theta

# (Dθ) * (Dφ_{xi}(E)) at E = Id
def stateActionDiff(xi: State) -> np.ndarray:
    coordsAction = lambda U: local_coords(stateAction(SymGroup.exp(U), xi))
    differential = numericalDifferential(coordsAction, np.zeros((21, 1)))
    return differential
