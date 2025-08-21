# === Final_EqF.py (APEqF filter) ===
# Equivariant Filter on (SE23 × se3 × R3 × SO3) with lever-arm and magnetometer extrinsics.

import numpy as np
from scipy.linalg import expm
from Symmetry import *
from System import measureMag, measureVel, input_from_vector, G
from pylie import SE23, SO3, SE3

def continuous_lift(xi : State, U : InputSpace) -> np.ndarray:
    """
    Group lift L(xi, U) in R^{21} such that
      d/dt stateAction(exp(L dt), xi) matches continuous-time INS with bias RW and calibration RW.
    Adapted from the SE23_se23 lift (reference code), extended to se3×R3×SO3.
    """
    L = np.zeros((21, 1))

    # Navigation (9): [ω, a, μ] - [bw, ba, 0] + frame terms
    b9 = np.vstack((xi.b[0:3], xi.b[3:6], np.zeros((3,1))))
    frame_term = SE23.vee(xi.T.inv().as_matrix() @ (G + f_10(xi.T.as_matrix())))
    if frame_term.ndim == 1:
        frame_term = frame_term.reshape(-1, 1)
    L[0:9, 0:1] = (U.as_W_vec() - b9) + frame_term

    # Bias part (6): equivariant form reduces to dot{b} = tau
    # Using the same structure as reference (A0t coupling), ensure tests reduce to dot_b = tau.
    L[9:15, 0:1] = SE3.adjoint(np.vstack((xi.b[0:3], xi.b[3:6]))) @ np.vstack((L[0:3], L[3:6])) - U.tau

    # Lever-arm (3): random walk rate μ
    L[15:18, 0:1] = U.mu

    # Extrinsic SO3 (3): body-fixed drift rate wM
    L[18:21, 0:1] = U.wM

    return L

class SE23_se3_R3_SO3_EqF:
    def __init__(self,
                 use_gcu=False,
                 gcu_alpha=0.0,
                 gcu_beta=0.0,
                 initial_att_noise=1.0,
                 initial_vel_noise=1.0,
                 initial_pos_noise=1.0,
                 initial_bias_noise=0.01,
                 initial_lever_noise=1.0,
                 initial_extrinsic_noise=0.1,
                 propagationonly=False,
                 equivariant_output=False,
                 curvature_correction=False):
        self.X_hat = SymGroup.identity()
        sigma_vec = np.concatenate((
            np.ones((1, 3)) * initial_att_noise**2,
            np.ones((1, 3)) * initial_vel_noise**2,
            np.ones((1, 3)) * initial_pos_noise**2,
            np.ones((1, 6)) * initial_bias_noise**2,
            np.ones((1, 3)) * initial_lever_noise**2,
            np.ones((1, 3)) * initial_extrinsic_noise**2
        ), axis=1)
        self.Sigma = np.eye(int(sigma_vec.shape[1])) * sigma_vec
        self.Dphi0 = stateActionDiff(xi_0)              # (Dθ) ∘ (Dφ_{xi0}(E))|_E=Id
        self.InnovationLift = np.linalg.pinv(self.Dphi0)
        self.propagation_only = propagationonly
        self.curvature_correction = curvature_correction
        self.dof = 21
        self.t = None
        self.u = None
        self.equivariant_output = equivariant_output
        print(f"Exponential coordinates = {exponential_coords}")
        print(f"Fake exponential        = {fake_exponential}")
        self.use_gcu = use_gcu
        self.gcu_alpha = gcu_alpha
        self.gcu_beta = gcu_beta

    def stateEstimate(self):
        return stateAction(self.X_hat, xi_0)

    def getEstimate(self):
        xi_hat = self.stateEstimate()
        R = xi_hat.T.R().as_matrix()
        v = xi_hat.T.x().as_vector()
        p = xi_hat.T.w().as_vector()
        # ensure b has shape (6,1)
        b = xi_hat.b
        if b is None:
            b = np.zeros((6,1))
        b = np.asarray(b)
        if b.ndim == 1:
            b = b.reshape(6,1)
        elif b.shape == (6,):
            b = b.reshape(6,1)
        bw = b[0:3, :]
        ba = b[3:6, :]
        # ensure t is column (3,1)
        t = np.asarray(xi_hat.t).reshape(3,1)
        
        # S could be SO3 object or numpy array
        try:
            S = xi_hat.S.R().as_matrix()
        except AttributeError:
            try:
                S = xi_hat.S.as_matrix()
            except AttributeError:
                S = np.asarray(xi_hat.S)
                if S.shape != (3, 3):
                    S = np.eye(3)  # fallback
        return R, p, v, bw, ba, t, S

    # --- Linearization matrices (A,B,C) ---
    def stateMatrixA(self, u : InputSpace) -> np.ndarray:
        # Analytical structure following reference; for robustness use numeric diff around xi_0
        u_0 = velocityAction(self.X_hat.inv(), u)
        # Analytical block structure
        A = np.zeros((21, 21))
        # nav 9x9
        A[0:9, 0:9] = np.block([
            [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))],
            [SO3.wedge(np.array([[0.0],[0.0],[-9.81]])), np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), np.eye(3), np.zeros((3,3))]
        ])
        # coupling bias -> nav
        A[0:9, 9:15] = -np.block([
            [np.eye(3), np.zeros((3,3))],
            [np.zeros((3,3)), np.eye(3)],
            [np.zeros((3,3)), np.zeros((3,3))]
        ])
        # bias 6x6
        A[9:15, 9:15] = np.zeros((6,6))
        # lever/extrinsic as static (identity in Bt)
        # lever does not affect nav kinematics directly here
        # SO3 extrinsic not affecting nav directly
        return A

    def inputMatrixB(self, u: InputSpace) -> np.ndarray:
        B = np.zeros((21, 21))
        AdB = self.X_hat.C.Adjoint()  # SE23 adjoint for nav
        # nav
        B[0:9, 0:9] = AdB
        # bias
        B[9:15, 9:15] = np.eye(6)
        # lever
        B[15:18, 15:18] = np.eye(3)
        # extrinsic
        B[18:21, 18:21] = np.eye(3)
        return B

    def outputMatrixC(self, y: np.ndarray) -> np.ndarray:
        opf = lambda eps : measurePos(stateAction(self.X_hat, local_coords_inv(eps)))
        C0 = numericalDifferential(opf, np.zeros((21, 1)))
        return C0[0:3, :]  # position-only measurement

    # --- Propagate & Update ---

    # --- GCU-inspired innovation covariance inflation ---
    def _inflate_S(self, S: np.ndarray, R_meas: np.ndarray) -> np.ndarray:
        if not self.use_gcu:
            return S
        # Simple inflation: S' = (1 + alpha) * S + beta * R
        # alpha >= 0, beta >= 0
        return (1.0 + float(self.gcu_alpha)) * S + float(self.gcu_beta) * R_meas

    def propagate(self, t: float, vel: np.ndarray,
                  omega_noise: float, acc_noise: float, bias_noise: float,
                  lever_noise: float, extrinsic_noise: float):
        if self.t is None:
            self.t = t
            self.u = input_from_vector(vel)
            return True
        dt = t - self.t
        if dt <= 0:
            return False
        self.t = t
        self.u = input_from_vector(vel)

        # Continuous lift at current state (right-invariant)
        u0 = velocityAction(self.X_hat.inv(), self.u)
        L = continuous_lift(xi_0, u0)

        # Propagate mean
        self.X_hat = SymGroup.exp(L * dt) * self.X_hat

        # Propagate covariance (first-order)
        A = self.stateMatrixA(self.u)
        B = self.inputMatrixB(self.u)
        Q = np.eye(21)
        Q[0:3, 0:3] *= omega_noise**2
        Q[3:6, 3:6] *= acc_noise**2
        Q[9:15, 9:15] *= bias_noise**2
        Q[15:18, 15:18] *= lever_noise**2
        Q[18:21, 18:21] *= extrinsic_noise**2
        F = np.eye(21) + A * dt
        self.Sigma = F @ self.Sigma @ F.T + B @ Q @ B.T * dt
        return True

    def update_position(self, y: np.ndarray, R_meas: np.ndarray, right_error: bool = True):
        # Design H = C(x) by numeric diff to respect equivariance
        Ct = self.outputMatrixC(y)
        xi_hat = self.stateEstimate()
        if right_error:
            delta = y - measurePos(xi_hat)
        else:
            # left error (equivariant output form)
            delta = self.X_hat.C.w().as_vector() - y
        S = Ct @ self.Sigma @ Ct.T + R_meas
        Sinv = np.linalg.inv(S)
        K = self.Sigma @ Ct.T @ Sinv
        Delta = self.InnovationLift @ K @ delta
        self.X_hat = SymGroup.exp(Delta) * self.X_hat
        self.Sigma = (np.eye(self.dof) - K @ Ct) @ self.Sigma
        if self.curvature_correction:
            GammaM = 0.5 * grp_adj(K @ delta)
            exp_Gamma = expm(GammaM)
            self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T
        nis = float(delta.T @ Sinv @ delta)
        return nis


    def outputMatrixC_mag(self, m_G) -> np.ndarray:
        opf = lambda eps : measureMag(stateAction(self.X_hat, local_coords_inv(eps)), m_G)
        C0 = numericalDifferential(opf, np.zeros((21, 1)))
        return C0  # full 3x21

    def update_magnetometer(self, y: np.ndarray, R_meas: np.ndarray, m_G: np.ndarray, right_error: bool = True):
        Ct = self.outputMatrixC_mag(m_G)
        xi_hat = self.stateEstimate()
        yhat = measureMag(xi_hat, m_G)
        if right_error:
            delta = y - yhat
        else:
            # left-error form would require output action; right-error suffices here
            delta = y - yhat
        S = Ct @ self.Sigma @ Ct.T + R_meas
        Sinv = np.linalg.inv(S)
        K = self.Sigma @ Ct.T @ Sinv
        Delta = self.InnovationLift @ K @ delta
        self.X_hat = SymGroup.exp(Delta) * self.X_hat
        self.Sigma = (np.eye(self.dof) - K @ Ct) @ self.Sigma
        if self.curvature_correction:
            from scipy.linalg import expm
            GammaM = 0.5 * grp_adj(K @ delta)
            exp_Gamma = expm(GammaM)
            self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T
        nis = float(delta.T @ Sinv @ delta)
        return nis


    def outputMatrixC_vel(self) -> np.ndarray:
        # Differentiates measureVel around current estimate using stored self.u
        if self.u is None:
            raise RuntimeError("No stored input 'u' for velocity measurement model.")
        opf = lambda eps : measureVel(stateAction(self.X_hat, local_coords_inv(eps)), self.u, True)
        C0 = numericalDifferential(opf, np.zeros((21, 1)))
        return C0

    def update_velocity(self, y: np.ndarray, R_meas: np.ndarray, right_error: bool = True):
        Ct = self.outputMatrixC_vel()
        xi_hat = self.stateEstimate()
        yhat = measureVel(xi_hat, self.u, True)
        if right_error:
            delta = y - yhat
        else:
            delta = y - yhat
        S = Ct @ self.Sigma @ Ct.T + R_meas
        S = self._inflate_S(S, R_meas)
        Sinv = np.linalg.inv(S)
        K = self.Sigma @ Ct.T @ Sinv
        Delta = self.InnovationLift @ K @ delta
        self.X_hat = SymGroup.exp(Delta) * self.X_hat
        self.Sigma = (np.eye(self.dof) - K @ Ct) @ self.Sigma
        if self.curvature_correction:
            from scipy.linalg import expm
            GammaM = 0.5 * grp_adj(K @ delta)
            exp_Gamma = expm(GammaM)
            self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T
        nis = float(delta.T @ Sinv @ delta)
        return nis
