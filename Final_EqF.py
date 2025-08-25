# === Final_EqF_fixed.py — APEqF (paper-consistent) with robust propagation & GCU ===
# NOTE: Keep imports consistent with your project.
import numpy as np
from scipy.linalg import expm
from Symmetry import State, stateAction, velocityAction, grp_adj, SymGroup
from pylie import SE3, SO3, SE23
from System import (measureMag_equivariant, measureVel_equivariant, measurePos_equivariant,
                    input_from_vector, G)

class SE23_se3_R3_SO3_EqF:
    """
    Paper-consistent equivariant filter on (SE23 × se3 × R3 × SO3).
    Assumes existence of:
      - State: (R, v, p, b:6x1, t:3x1, S:SO3)
      - stateAction(exp_xi, xi), velocityAction(X_inv, U), grp_adj(xi_hat) -> 21x21 adjoint
      - measure*_*equivariant(xi_hat, ...)
    """
    def __init__(self, xi0: State, Sigma0: np.ndarray, use_gcu: bool = True,
                 propagation_mode: str = "At0",
                 mag_use_direction_only: bool = True):
        self.X_hat: State = xi0
        self.Sigma = Sigma0.copy()
        self.t = None
        self.u = None
        self.use_gcu = use_gcu
        self.mag_use_direction_only = mag_use_direction_only
        self.propagation_mode = propagation_mode
        # Cache last measurement/reference values for C* computation
        self.m_G = None  # Earth magnetic field vector (3x1)
        self.pi_last = None  # Last GNSS position measurement π_G (3x1)
        self.nu_last = None  # Last GNSS velocity measurement ν_G (3x1)

    # ---------------------- LIFT ----------------------
    def continuous_lift(self, xi: State, U):

        """
        Build the 21x1 group lift vector L(xi, U) (right-invariant coordinates).
        Paper-aligned slots:
          L0:3   = (ω - b_ω)
          L3:6   = (a - b_a)
          L6:9   = t ∧ (ω - b_ω)                [lever-arm kinematics]
          L9:15  = τ (bias random-walk input)   [if provided, else 0]
          L15:18 = μ (lever-arm RW input)       [if provided, else 0]
          L18:21 = S^T (ω - b_ω)                [extrinsic rotation kinematics]
        """
        def col3(x):
            x = np.asarray(x).reshape(-1,1) if x is not None else np.zeros((3,1))
            return x[:3,:]
        def as_vec3_w(w):
            if w is None:
                return np.zeros((3,1))
            w_arr = np.asarray(w)
            if w_arr.shape == (3,3):
                # wedge -> vee
                try:
                    return SO3.vee(w_arr).reshape(3,1)
                except Exception:
                    # fallback using antisymmetric entries
                    return np.array([[w_arr[2,1]],[w_arr[0,2]],[w_arr[1,0]]])
            return w_arr.reshape(3,1)[:3,:]

        # Inputs (may be absent)
        w_in  = as_vec3_w(getattr(U, "w",  None))
        a_in  = col3(getattr(U, "a",  None))
        tau   = np.asarray(getattr(U, "tau", np.zeros((6,1)))).reshape(6,1)
        mu    = col3(getattr(U, "mu",  None))

        # State parts
        b = np.asarray(xi.b).reshape(6,1)
        b_w = b[0:3]
        b_a = b[3:6]
        t   = col3(xi.t)
        S   = xi.S.as_matrix()

        # Paper-aligned components
        w_tilde = w_in - b_w
        a_tilde = a_in - b_a

        def skew(v):
            v = v.reshape(3,)
            return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=float)

        L0 = w_tilde
        L1 = a_tilde
        L2 = skew(t) @ w_tilde      # t × ω̃
        L3 = tau
        L4 = mu
        L5 = S.T @ w_tilde          # ω̃ expressed in magnetometer frame

        L = np.vstack([L0, L1, L2, L3, L4, L5])  # (21,1)
        return L




    # ---------------------- A_t^0 (paper) ----------------------
    def _A_t0(self, xi: State, U) -> np.ndarray:
        """
        Continuous-time error Jacobian A_t^0 (21x21) in world-frame strapdown linearization.
        State order: [theta(0:3), v(3:6), p(6:9), b(9:15)=[bw,ba], t(15:18), s(18:21)].
        Uses bias-compensated rates w_tilde, a_tilde and current rotation R = G R_I.
        """
        import numpy as _np
        def _skew(v):
            v = _np.asarray(v).reshape(3,)
            return _np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=float)
        R = xi.T.R().as_matrix()
        b = _np.asarray(xi.b).reshape(6,1)
        bw = b[0:3,0:1]
        ba = b[3:6,0:1]
        w_raw = getattr(U, "w", _np.zeros((3,1)))
        if w_raw is None:
            w = _np.zeros((3,1))
        else:
            w_arr = _np.asarray(w_raw)
            if w_arr.shape == (3,3):
                try:
                    from pylie import SO3 as _SO3
                    w = _SO3.vee(w_arr).reshape(3,1)
                except Exception:
                    w = _np.array([[w_arr[2,1]],[w_arr[0,2]],[w_arr[1,0]]])
            else:
                w = w_arr.reshape(3,1)
        a = _np.asarray(getattr(U, "a", _np.zeros((3,1)))).reshape(3,1)
        w_tilde = (w - bw).reshape(3,)
        a_tilde = (a - ba).reshape(3,)

        Z = _np.zeros((3,3)); I = _np.eye(3)
        A = _np.zeros((21,21))
        # attitude
        A[0:3, 0:3] = -_skew(w_tilde)
        A[0:3, 9:12] = -I  # dθ depends on bw
        # velocity
        A[3:6, 0:3] = - R @ _skew(a_tilde)
        A[3:6, 12:15] = - R  # dv depends on ba
        # position
        A[6:9, 3:6] = I
        # others zero
        return A
    # ---------------------- PROPAGATE ----------------------
    def propagate(self, t: float, vel: np.ndarray,
                  omega_noise: float, acc_noise: float, bias_noise: float,
                  lever_noise: float, extrinsic_noise: float):

        """
        t: time in seconds
        vel: stacked IMU vector -> System.input_from_vector will parse
        *_noise: continuous-time std (per sqrt(s))
        """
        # time bookkeeping
        if self.t is None:
            self.t = float(t)
            self.u = input_from_vector(vel)
            return True

        dt = float(t) - float(self.t)
        if dt <= 0:
            return False
        self.t = float(t)
        self.u = input_from_vector(vel)

        # Right-invariant lift and on-manifold mean propagation (no fallbacks)
        u0 = velocityAction(self.X_hat.inv(), self.u) if hasattr(self.X_hat, "inv") else self.u
        L = self.continuous_lift(self.X_hat, u0)
        try:
            step = SymGroup.exp(L * dt)
            self.X_hat = stateAction(step, self.X_hat)
        except Exception as e:
            raise RuntimeError(f"On-manifold propagation failed: {e}")

        # Paper-style covariance propagation Σ ← Φ Σ Φᵀ + Qd with Φ = expm(A_t0 dt)
        from scipy.linalg import expm as _expm
        A = self._A_t0(self.X_hat, u0)
        Phi = _expm(A * dt)

        # Continuous-time noise to discrete (first-order) — diagonal structure
        Qd = np.zeros_like(self.Sigma)
        Qd[0:3,0:3]       = (omega_noise**2)     * dt * np.eye(3)
        Qd[3:6,3:6]       = (acc_noise**2)       * dt * np.eye(3)
        Qd[9:15,9:15]     = (bias_noise**2)      * dt * np.eye(6)
        Qd[15:18,15:18]   = (lever_noise**2)     * dt * np.eye(3)
        Qd[18:21,18:21]   = (extrinsic_noise**2) * dt * np.eye(3)

        self.Sigma = Phi @ self.Sigma @ Phi.T + Qd
        return True

        return True

    # ---------------------- GCU ----------------------
    
    def apply_gcu_inflation(self, innovation: np.ndarray, S: np.ndarray, R_meas: np.ndarray) -> np.ndarray:
        """Generalized Covariance Union (paper): S' = β (CΣCᵀ + α ỹỹᵀ) + R, with β piecewise."""
        # Ensure invertible S for r = ỹᵀ S⁻¹ ỹ
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S = S + 1e-6 * np.eye(S.shape[0])
            S_inv = np.linalg.inv(S)
        r = float(innovation.T @ S_inv @ innovation)
        # β(r)
        beta = ((1 + np.sqrt(max(r, 0.0)))**2) / (1 + r) if r < 1.0 else 2.0
        # α in [0,1]
        alpha = getattr(self, "gcu_alpha", 0.05)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        S_gcu = beta * (S + alpha * (innovation @ innovation.T)) + R_meas
        return S_gcu


    # ---------------------- UPDATES ----------------------
    def update_position_equivariant(self, y_raw: np.ndarray, R_meas: np.ndarray):
        xi_hat = self.X_hat
        innovation = measurePos_equivariant(xi_hat, y_raw)
        y_hat = innovation

        self.pi_last = y_raw.reshape(3,1)
        C = self.compute_output_matrix_equivariant(xi_hat, y_hat, innovation, sensor="pos")
        S = C @ self.Sigma @ C.T + R_meas
        if self.use_gcu:
            S = self.apply_gcu_inflation(innovation, S, R_meas)

        K = self.Sigma @ C.T @ np.linalg.inv(S)
        Gamma = 0.5 * grp_adj(K @ innovation)
        exp_Gamma = expm(Gamma)
        self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T
        
        # Convert K @ innovation to SymGroup object
        try:
            update_group = SymGroup()
            se3_input = np.vstack([K[0:3] @ innovation, K[3:6] @ innovation])
            
            # 수치적 안정성을 위한 입력 제한
            se3_input = np.clip(se3_input, -1e3, 1e3)
            
            # SE23.exp 호출 시 안전성 검사
            if np.linalg.norm(se3_input) < 1e-6:
                # 너무 작은 값이면 항등 변환 사용
                update_group.C = SE23.identity()
            else:
                update_group.C = SE23.exp(np.vstack([se3_input, np.zeros((3,1))]))
            
            update_group.gamma = SE3.wedge(se3_input)
            update_group.delta = np.clip(K[15:18] @ innovation, -1e3, 1e3)
            
            # SO3.exp 호출 시 안전성 검사
            mag_input = K[18:21] @ innovation
            if np.linalg.norm(mag_input) < 1e-6:
                update_group.E = SO3.identity()
            else:
                mag_input = np.clip(mag_input, -1e3, 1e3)
                update_group.E = SO3.exp(mag_input)
            
            self.X_hat = stateAction(update_group, xi_hat)
            
        except Exception as e:
            print(f"Warning: State update failed: {e}")
            # 대체 방법: 상태를 그대로 유지
            pass
        nis = float(innovation.T @ np.linalg.inv(S) @ innovation)
        return nis

    def update_velocity_equivariant(self, y_raw: np.ndarray, R_meas: np.ndarray, subtract_bias: bool = True):
        xi_hat = self.X_hat
        innovation = measureVel_equivariant(xi_hat, self.u, y_raw, subtract_bias=subtract_bias)
        y_hat = innovation

        self.nu_last = y_raw.reshape(3,1)
        C = self.compute_output_matrix_equivariant(xi_hat, y_hat, innovation, sensor="vel")
        S = C @ self.Sigma @ C.T + R_meas
        if self.use_gcu:
            S = self.apply_gcu_inflation(innovation, S, R_meas)

        K = self.Sigma @ C.T @ np.linalg.inv(S)
        Gamma = 0.5 * grp_adj(K @ innovation)
        exp_Gamma = expm(Gamma)
        self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T
        
        # Convert K @ innovation to SymGroup object
        try:
            update_group = SymGroup()
            se3_input = np.vstack([K[0:3] @ innovation, K[3:6] @ innovation])
            
            # 수치적 안정성을 위한 입력 제한
            se3_input = np.clip(se3_input, -1e3, 1e3)
            
            # SE23.exp 호출 시 안전성 검사
            if np.linalg.norm(se3_input) < 1e-6:
                # 너무 작은 값이면 항등 변환 사용
                update_group.C = SE23.identity()
            else:
                update_group.C = SE23.exp(np.vstack([se3_input, np.zeros((3,1))]))
            
            update_group.gamma = SE3.wedge(se3_input)
            update_group.delta = np.clip(K[15:18] @ innovation, -1e3, 1e3)
            
            # SO3.exp 호출 시 안전성 검사
            mag_input = K[18:21] @ innovation
            if np.linalg.norm(mag_input) < 1e-6:
                update_group.E = SO3.identity()
            else:
                mag_input = np.clip(mag_input, -1e3, 1e3)
                update_group.E = SO3.exp(mag_input)
            
            self.X_hat = stateAction(update_group, xi_hat)
            
        except Exception as e:
            print(f"Warning: State update failed: {e}")
            # 대체 방법: 상태를 그대로 유지
            pass
        
        nis = float(innovation.T @ np.linalg.inv(S) @ innovation)
        return nis

    def update_magnetometer_equivariant(self, y_raw: np.ndarray, R_meas: np.ndarray):
        xi_hat = self.X_hat
        y_in = y_raw.copy()
        # 지구 자기장 벡터 (대략적인 값)
        m_G = np.array([0.2, 0.0, -0.5]).reshape(3, 1)  # North, East, Down (Gauss)
        
        if self.mag_use_direction_only:
            # use only direction: normalize both predicted and measured
            norm = lambda v: v / (np.linalg.norm(v) + 1e-12)
            y_in = norm(y_in)
            y_hat, innovation = measureMag_equivariant(xi_hat, m_G, norm(y_in))
        else:
            y_hat, innovation = measureMag_equivariant(xi_hat, m_G, y_in)

        self.m_G = m_G.reshape(3,1)
        C = self.compute_output_matrix_equivariant(xi_hat, y_hat, innovation, sensor="mag")
        S = C @ self.Sigma @ C.T + R_meas
        if self.use_gcu:
            S = self.apply_gcu_inflation(innovation, S, R_meas)

        K = self.Sigma @ C.T @ np.linalg.inv(S)
        Gamma = 0.5 * grp_adj(K @ innovation)
        exp_Gamma = expm(Gamma)
        self.Sigma = exp_Gamma @ self.Sigma @ exp_Gamma.T
        
        # Convert K @ innovation to SymGroup object
        try:
            update_group = SymGroup()
            se3_input = np.vstack([K[0:3] @ innovation, K[3:6] @ innovation])
            
            # 수치적 안정성을 위한 입력 제한
            se3_input = np.clip(se3_input, -1e3, 1e3)
            
            # SE23.exp 호출 시 안전성 검사
            if np.linalg.norm(se3_input) < 1e-6:
                # 너무 작은 값이면 항등 변환 사용
                update_group.C = SE23.identity()
            else:
                update_group.C = SE23.exp(np.vstack([se3_input, np.zeros((3,1))]))
            
            update_group.gamma = SE3.wedge(se3_input)
            update_group.delta = np.clip(K[15:18] @ innovation, -1e3, 1e3)
            
            # SO3.exp 호출 시 안전성 검사
            mag_input = K[18:21] @ innovation
            if np.linalg.norm(mag_input) < 1e-6:
                update_group.E = SO3.identity()
            else:
                mag_input = np.clip(mag_input, -1e3, 1e3)
                update_group.E = SO3.exp(mag_input)
            
            self.X_hat = stateAction(update_group, xi_hat)
            
        except Exception as e:
            print(f"Warning: State update failed: {e}")
            # 대체 방법: 상태를 그대로 유지
            pass
        
        nis = float(innovation.T @ np.linalg.inv(S) @ innovation)
        return nis

    # ---------------------- C* ----------------------
    def compute_output_matrix_equivariant(self, xi_hat: State, y_hat: np.ndarray, innovation: np.ndarray, sensor: str) -> np.ndarray:
        """
        Paper-aligned equivariant output matrix C* for sensors in {mag,pos,vel}.
        Blocks correspond to right-invariant error coordinates:
          [δθ(0:3), δv(3:6), δp(6:9), δb(9:15), δt(15:18), δψ_M(18:21)]
        """
        import numpy as np
        C = np.zeros((3,21), dtype=float)

        def skew(v):
            v = v.reshape(3,)
            return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=float)

        if sensor == "mag":
            # y_hat = S^T R^T m_G
            y_m = y_hat.reshape(3,1)
            C[:,0:3]   = -0.5 * skew(y_m)      # attitude
            C[:,18:21] = -0.5 * skew(y_m)      # extrinsic rotation

        elif sensor == "pos":
            # y_hat = R^T (π - (p + R t))  =  R^T(π - p) - t
            t = np.asarray(xi_hat.t).reshape(3,1)
            y_p = y_hat.reshape(3,1)
            C[:,0:3]   = -0.5*skew(y_p + t)  # attitude coupling
            C[:,6:9]   = -np.eye(3)      # position
            C[:,15:18] = -np.eye(3)      # lever arm

        elif sensor == "vel":
            # y_hat = R^T(ν - v) - (ω̃ × t),  where ω̃ = (w - b_w)
            b_w = np.asarray(xi_hat.b).reshape(6,1)[0:3]
            t   = np.asarray(xi_hat.t).reshape(3,1)
            w_raw   = getattr(self.u, "w", np.zeros((3,1)))
            w_arr = np.asarray(w_raw)
            if w_arr.shape == (3,3):
                try:
                    w_vec = SO3.vee(w_arr).reshape(3,1)
                except Exception:
                    w_vec = np.array([[w_arr[2,1]],[w_arr[0,2]],[w_arr[1,0]]])
            else:
                w_vec = w_arr.reshape(3,1)
            w_tilde = w_vec - b_w
            C[:,0:3]   = -0.5*skew(y_hat.reshape(3,1) + (skew(w_tilde) @ t))
            C[:,3:6]   = -np.eye(3)
            C[:,15:18] = -skew(w_tilde)

        else:
            raise ValueError("Unknown sensor type for C*")

        return C
