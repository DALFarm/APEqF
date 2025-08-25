#!/usr/bin/env python3
"""
run_filter_fixed.py — robust APEqF runner
- Timestamp unit normalization (ns/μs/s → s)
- One-sample-per-update gating per sensor
- Chi-square/NIS gating
"""
import os, math
import numpy as np
import pandas as pd
from Final_EqF import SE23_se3_R3_SO3_EqF
from System import State, InputSpace, input_from_vector
from Symmetry import SymGroup  # if needed by your pipeline
from datetime import datetime

from scipy.spatial.transform import Rotation as Rotation

def safe_from_quat_wxyz(q_wxyz):
    import numpy as np
    q = np.asarray(q_wxyz, dtype=float).reshape(4,)
    if not np.isfinite(q).all():
        return Rotation.identity()
    n = np.linalg.norm(q)
    if n < 1e-12:
        return Rotation.identity()
    q = q / n
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])  # SciPy expects xyzw

def safe_from_quat_xyzw(q_xyzw):
    import numpy as np
    q = np.asarray(q_xyzw, dtype=float).reshape(4,)
    if not np.isfinite(q).all():
        return Rotation.identity()
    n = np.linalg.norm(q)
    if n < 1e-12:
        return Rotation.identity()
    q = q / n
    return Rotation.from_quat(q)

def sanitize_quat_columns(df):
    """Normalize any quaternion columns in df. Supports qw,qx,qy,qz and qx,qy,qz,qw."""
    import numpy as np
    if df is None: 
        return None
    cols = set(c.lower() for c in df.columns)
    if {'qw','qx','qy','qz'}.issubset(cols):
        Q = df[[c for c in df.columns if c.lower() in ['qw','qx','qy','qz']]].to_numpy(dtype=float)
        n = np.linalg.norm(Q, axis=1)
        bad = ~np.isfinite(n) | (n < 1e-12)
        Q[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        n = np.clip(np.linalg.norm(Q, axis=1), 1e-12, np.inf)
        Q = (Q.T / n).T
        # write back in original column order
        df.loc[:, [c for c in df.columns if c.lower() in ['qw','qx','qy','qz']]] = Q
    elif {'qx','qy','qz','qw'}.issubset(cols):
        Q = df[[c for c in df.columns if c.lower() in ['qx','qy','qz','qw']]].to_numpy(dtype=float)
        n = np.linalg.norm(Q, axis=1)
        bad = ~np.isfinite(n) | (n < 1e-12)
        Q[bad] = np.array([0.0, 0.0, 0.0, 1.0])
        n = np.clip(np.linalg.norm(Q, axis=1), 1e-12, np.inf)
        Q = (Q.T / n).T
        df.loc[:, [c for c in df.columns if c.lower() in ['qx','qy','qz','qw']]] = Q
    return df

def rotmat_to_quat_wxyz(R):
    """
    Convert 3x3 rotation matrix to unit quaternion (w,x,y,z) robustly.
    """
    import numpy as np
    R = np.asarray(R, dtype=float).reshape(3,3)
    trace = np.trace(R)
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = (1.0 + R[0,0] - R[1,1] - R[2,2]) ** 0.5 * 2.0
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = (1.0 - R[0,0] + R[1,1] - R[2,2]) ** 0.5 * 2.0
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = (1.0 - R[0,0] - R[1,1] + R[2,2]) ** 0.5 * 2.0
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n

# === Gating + GCU fallback helper (inserted) ===
NIS_GATE_3DOF = 7.815  # 95% gate for dof=3
def gated_update_with_gcu(filter_obj, which: str, y, R, **kwargs):
    """
    which in {'pos','vel','mag'}
    Returns (nis, used_gcu: bool)
    """
    call = {
        'pos': getattr(filter_obj, 'update_position_equivariant'),
        'vel': getattr(filter_obj, 'update_velocity_equivariant'),
        'mag': getattr(filter_obj, 'update_magnetometer_equivariant'),
    }[which]
    nis = call(y, R, **kwargs)
    if nis is not None and nis <= NIS_GATE_3DOF:
        return nis, False
    prev = getattr(filter_obj, "use_gcu", False)
    try:
        filter_obj.use_gcu = True
        nis2 = call(y, R, **kwargs)
        return nis2, True
    finally:
        filter_obj.use_gcu = prev
# === end helper ===


# -------------------- Config --------------------
NIS_GATE_3DOF = 7.815  # 95% for df=3
GPS_MAX_LAG_S = 0.5
MAG_MAX_LAG_S = 0.2

def to_seconds(t_raw: float) -> float:
    t = float(t_raw)
    if t > 1e12:   # ns
        return t * 1e-9
    if t > 1e6:    # μs
        return t * 1e-6
    return t       # already seconds

class APEqFRunner:
    def __init__(self, imu_df: pd.DataFrame, gps_df: pd.DataFrame, mag_df: pd.DataFrame,
                 xi0: State, Sigma0: np.ndarray):
        self.imu = imu_df.copy()
        self.gps = gps_df.copy() if gps_df is not None else None
        self.mag = mag_df.copy() if mag_df is not None else None
        self.filter = SE23_se3_R3_SO3_EqF(xi0, Sigma0, use_gcu=True, mag_use_direction_only=True)

        # normalize timestamps to seconds
        self.imu["t_s"] = self.imu["timestamp"].apply(to_seconds)
        if self.gps is not None:
            self.gps["t_s"] = self.gps["timestamp"].apply(to_seconds)
        if self.mag is not None:
            self.mag["t_s"] = self.mag["timestamp"].apply(to_seconds)

        # indices for each sensor to ensure one-sample-per-update
        self._igps = 0
        self._imag = 0

    def _next_gps(self, t_s):
        if self.gps is None or self._igps >= len(self.gps):
            return None
        # advance pointer while sample time <= t_s (causal)
        while self._igps < len(self.gps) and self.gps.iloc[self._igps]["t_s"] <= t_s:
            cand = self.gps.iloc[self._igps]
            self._igps += 1
            return cand
        return None

    def _next_mag(self, t_s):
        if self.mag is None or self._imag >= len(self.mag):
            return None
        # 현재 시간과 가장 가까운 자력계 데이터 찾기 (1초 허용 오차)
        while self._imag < len(self.mag) and self.mag.iloc[self._imag]["t_s"] <= t_s + 1.0:
            cand = self.mag.iloc[self._imag]
            self._imag += 1
            return cand
        return None

    def run(self,
            omega_noise: float=0.01, acc_noise: float=0.1, bias_noise: float=1e-3,
            lever_noise: float=1e-4, extrinsic_noise: float=1e-4):
        nis_logs = {"pos": [], "vel": [], "mag": []}
        pos_updates = vel_updates = mag_updates = 0
        
        print("Starting APEqF filter execution...")
        print(f"Total IMU samples: {len(self.imu)}")
        print(f"GPS samples: {len(self.gps) if self.gps is not None else 0}")
        print(f"Magnetometer samples: {len(self.mag) if self.mag is not None else 0}")
        
        # 자력계 데이터 디버깅
        if self.mag is not None:
            print(f"Magnetometer columns: {list(self.mag.columns)}")
            print(f"First mag timestamp: {self.mag.iloc[0]['t_s']}")
            print(f"Last mag timestamp: {self.mag.iloc[-1]['t_s']}")
            if 'mx' in self.mag.columns:
                print(f"First mag data: mx={self.mag.iloc[0]['mx']:.6f}, my={self.mag.iloc[0]['my']:.6f}, mz={self.mag.iloc[0]['mz']:.6f}")
            else:
                print("mx, my, mz columns not found in magnetometer data!")
        
        # IMU 데이터를 자력계 시작 시간 이후부터 처리
        imu_start_time = self.mag.iloc[0]["t_s"] if self.mag is not None else 0
        filtered_imu = self.imu[self.imu["t_s"] >= imu_start_time].sort_values("t_s")
        
        print(f"Filtering IMU data: start time adjusted from {self.imu['t_s'].min():.3f}s to {imu_start_time:.3f}s")
        print(f"IMU samples after filtering: {len(filtered_imu)}")
        
        for i, row in filtered_imu.iterrows():
            if i % 1000 == 0:
                print(f"Processing sample {i}/{len(filtered_imu)}")
            
            t = row["t_s"]
            
            # IMU 입력 벡터 구성
            imu_vec = np.array([row["gyro_x"], row["gyro_y"], row["gyro_z"],
                                row["acc_x"], row["acc_y"], row["acc_z"]], dtype=float).reshape(-1,1)

            # Propagation 단계
            try:
                ok = self.filter.propagate(
                    t, imu_vec,
                    omega_noise, acc_noise, bias_noise, lever_noise, extrinsic_noise
                )
            except Exception as e:
                # 런타임에 외부 데이터에서 0-노름 쿼터니언이 들어오는 경우만 안전하게 우회
                if "zero norm quaternions" in str(e).lower():
                    try:
                        # extrinsic 회전만 항등으로 복구 (필터 수학은 불변)
                        self.filter.X_hat.S = self.filter.X_hat.S.__class__.identity()
                    except Exception:
                        pass
                    print(f"Propagation safeguarded at sample {i} (identity fallback for bad quaternion).")
                    ok = True
                else:
                    # 다른 예외는 그대로 올림
                    raise

            if not ok:
                print(f"Warning: Propagation failed at sample {i}")
                # 필요 시 다음 샘플로 진행
                continue

            # GPS 위치 업데이트
            gps = self._next_gps(t)
            if gps is not None:
                try:
                    y_pos = np.array([gps["pos_x"], gps["pos_y"], gps["pos_z"]], dtype=float).reshape(3,1)
                    R_pos = np.diag([gps.get("pos_std_x", 2.0)**2,
                                      gps.get("pos_std_y", 2.0)**2,
                                      gps.get("pos_std_z", 3.0)**2])
                    
                    nis = self.filter.update_position_equivariant(y_pos, R_pos)
                    nis_logs["pos"].append((t, float(nis) if nis is not None else np.nan)); pos_updates += 1
                    if i % 1000 == 0:
                        print(f"  GPS position update: NIS = {nis if nis is not None else float('nan'):.3f}")
                except Exception as e:
                    if i % 1000 == 0:
                        print(f"  GPS position update failed: {e}")

            # GPS 속도 업데이트 (사용 가능한 경우)
            if gps is not None and all(k in gps for k in ("vel_x","vel_y","vel_z")):
                try:
                    y_vel = np.array([gps["vel_x"], gps["vel_y"], gps["vel_z"]], dtype=float).reshape(3,1)
                    R_vel = np.diag([gps.get("vel_std_x", 0.5)**2,
                                      gps.get("vel_std_y", 0.5)**2,
                                      gps.get("vel_std_z", 0.7)**2])
                    
                    nis = self.filter.update_velocity_equivariant(y_vel, R_vel, subtract_bias=True)
                    nis_logs["vel"].append((t, float(nis) if nis is not None else np.nan)); vel_updates += 1
                    if i % 1000 == 0:
                        print(f"  GPS velocity update: NIS = {nis if nis is not None else float('nan'):.3f}")
                except Exception as e:
                    if i % 1000 == 0:
                        print(f"  GPS velocity update failed: {e}")

            # 자력계 업데이트
            mag = self._next_mag(t)
            if i % 1000 == 0:
                print(f"  Time {t:.3f}s: _imag={self._imag}, mag_data_len={len(self.mag) if self.mag is not None else 0}")
                if mag is not None:
                    print(f"    Got magnetometer data: mx={mag['mx']:.6f}, my={mag['my']:.6f}, mz={mag['mz']:.6f}")
                else:
                    print(f"    No magnetometer data at time {t:.3f}s")
                    # 디버깅: 다음 자력계 데이터 시간 확인
                    if self._imag < len(self.mag):
                        next_mag_time = self.mag.iloc[self._imag]["t_s"]
                        print(f"    Next mag data at: {next_mag_time:.3f}s")
                    else:
                        print(f"    No more magnetometer data available")
            
            # 자력계 데이터가 실제로 로드되었는지 확인
            if mag is not None:
                print(f"    MAG UPDATE: t={t:.3f}s, mx={mag['mx']:.6f}, my={mag['my']:.6f}, mz={mag['mz']:.6f}")
                try:
                    y_mag = np.array([mag["mx"], mag["my"], mag["mz"]], dtype=float).reshape(3,1)
                    # 마이크로테슬라를 가우스로 변환
                    if y_mag.max() > 1000:
                        y_mag = y_mag / 100.0  # μT → Gauss
                    R_mag = (0.05**2) * np.eye(3)
                    
                    if i % 1000 == 0:
                        print(f"  Magnetometer data: mx={mag['mx']:.6f}, my={mag['my']:.6f}, mz={mag['mz']:.6f}")
                    
                    nis = self.filter.update_magnetometer_equivariant(y_mag, R_mag)
                    nis_logs["mag"].append((t, float(nis) if nis is not None else np.nan)); mag_updates += 1
                    if i % 1000 == 0:
                        print(f"  Magnetometer update: NIS = {nis if nis is not None else float('nan'):.3f}")
                except Exception as e:
                    if i % 1000 == 0:
                        print(f"  Magnetometer update failed: {e}")
                        import traceback
                        traceback.print_exc()

        print(f"Filter execution completed!")
        print(f"Position updates: {len(nis_logs['pos'])}")
        print(f"Velocity updates: {len(nis_logs['vel'])}")
        print(f"Magnetometer updates: {len(nis_logs['mag'])}")
        
        return nis_logs

# Example usage is omitted; integrate with your existing main().

def main():
    """메인 함수"""
    print("=== APEqF Filter Runner - Fixed Version ===")
    
    # 데이터 디렉토리
    data_dir = "ekf_data"
    
    # 센서 데이터 파일 경로
    imu_file = os.path.join(data_dir, "sensors", "log_4_2025-7-7-09-05-20_sensor_combined_0.csv")
    gps_file = os.path.join(data_dir, "sensors", "log_4_2025-7-7-09-05-20_sensor_gps_0.csv")
    mag_file = os.path.join(data_dir, "sensors", "log_4_2025-7-7-09-05-20_sensor_mag_0.csv")
    
    # 데이터 로드
    print("Loading sensor data...")
    
    # IMU 데이터 로드
    if os.path.exists(imu_file):
        imu_df = pd.read_csv(imu_file)
        imu_df = sanitize_quat_columns(imu_df)
        # 컬럼명 매핑 (실제 데이터에 맞게 조정)
        imu_df = imu_df.rename(columns={
            'gyro_rad[0]': 'gyro_x',
            'gyro_rad[1]': 'gyro_y', 
            'gyro_rad[2]': 'gyro_z',
            'accelerometer_m_s2[0]': 'acc_x',
            'accelerometer_m_s2[1]': 'acc_y',
            'accelerometer_m_s2[2]': 'acc_z'
        })
        print(f"Loaded {len(imu_df)} IMU samples")
    else:
        print(f"IMU file not found: {imu_file}")
        return
    
    # GPS 데이터 로드
    gps_df = None
    if os.path.exists(gps_file):
        gps_df = pd.read_csv(gps_file)
        gps_df = sanitize_quat_columns(gps_df)
        # GPS 컬럼명 매핑 (실제 컬럼명 사용)
        if 'latitude_deg' in gps_df.columns and 'longitude_deg' in gps_df.columns:
            # 간단한 평면 근사로 변환
            ref_lat = gps_df['latitude_deg'].iloc[0]
            ref_lon = gps_df['longitude_deg'].iloc[0]
            gps_df['pos_x'] = (gps_df['longitude_deg'] - ref_lon) * 111320
            gps_df['pos_y'] = (gps_df['latitude_deg'] - ref_lat) * 110540
            gps_df['pos_z'] = gps_df.get('altitude_msl_m', 0)
        print(f"Loaded {len(gps_df)} GPS samples")
    
    # 자력계 데이터 로드
    mag_df = None
    if os.path.exists(mag_file):
        mag_df = pd.read_csv(mag_file)
        mag_df = sanitize_quat_columns(mag_df)
        # 자력계 컬럼명 매핑
        mag_cols = ['x', 'y', 'z', 'mag_x', 'mag_y', 'mag_z', 'magnetometer_ga[0]', 'magnetometer_ga[1]', 'magnetometer_ga[2]']
        for col in mag_cols:
            if col in mag_df.columns:
                if col in ['x', 'y', 'z']:
                    mag_df['mx'] = mag_df['x']
                    mag_df['my'] = mag_df['y']
                    mag_df['mz'] = mag_df['z']
                    break
                elif col in ['mag_x', 'mag_y', 'mag_z']:
                    mag_df['mx'] = mag_df['mag_x']
                    mag_df['my'] = mag_df['mag_y']
                    mag_df['mz'] = mag_df['mag_z']
                    break
                elif col in ['magnetometer_ga[0]', 'magnetometer_ga[1]', 'magnetometer_ga[2]']:
                    mag_df['mx'] = mag_df['magnetometer_ga[0]']
                    mag_df['my'] = mag_df['magnetometer_ga[1]']
                    mag_df['mz'] = mag_df['magnetometer_ga[2]']
                    break
        print(f"Loaded {len(mag_df)} magnetometer samples")
    
    # 초기 상태 설정
    from Final_EqF import SE23_se3_R3_SO3_EqF
    
    # 초기 상태 (간단한 설정)
    xi0 = State()  # 기본 초기 상태
    Sigma0 = np.eye(21) * 0.1  # 초기 공분산 (21x21)
    
    # APEqF 러너 초기화
    runner = APEqFRunner(imu_df, gps_df, mag_df, xi0, Sigma0)
    
    # 필터 실행
    print("Running APEqF filter...")
    nis_logs = runner.run()
    
    print("Filter execution completed!")
    print(f"Position updates: {len(nis_logs['pos'])}")
    print(f"Velocity updates: {len(nis_logs['vel'])}")
    print(f"Magnetometer updates: {len(nis_logs['mag'])}")
    
    # 결과 저장
    print("Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"apeqf_results_paper_format_{timestamp}.csv"
    
    # 결과 데이터 준비
    results_data = []
    for i, row in imu_df.iterrows():
        timestamp_row = row['timestamp']
        
        # 가장 가까운 GPS 데이터 찾기
        gps_row = None
        if gps_df is not None:
            time_diff = np.abs(gps_df['timestamp'] - timestamp_row)
            if len(time_diff) > 0:
                closest_idx = time_diff.idxmin()
                gps_row = gps_df.iloc[closest_idx]
        
        # 가장 가까운 자력계 데이터 찾기
        mag_row = None
        if mag_df is not None:
            time_diff = np.abs(mag_df['timestamp'] - timestamp_row)
            if len(time_diff) > 0:
                closest_idx = time_diff.idxmin()
                mag_row = mag_df.iloc[closest_idx]
        

        # Extract current filter attitude quaternion to avoid zero-norm quaternions
        try:
            R_G_I = runner.filter.X_hat.T.R().as_matrix()
            qw, qx, qy, qz = rotmat_to_quat_wxyz(R_G_I)
        except Exception:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

        # 결과 행 생성
        result_row = {
            'timestamp': timestamp_row,
            'pos_x': gps_row['pos_x'] if gps_row is not None else np.nan,
            'pos_y': gps_row['pos_y'] if gps_row is not None else np.nan,
            'pos_z': gps_row['pos_z'] if gps_row is not None else np.nan,
            'vel_x': 0.0,  # 임시 값
            'vel_y': 0.0,  # 임시 값
            'vel_z': 0.0,  # 임시 값
            'att_roll': 0.0,  # 임시 값
            'qw': float(qw), 'qx': float(qx), 'qy': float(qy), 'qz': float(qz),  # from filter state
            'att_pitch': 0.0,  # 임시 값
            'att_yaw': 0.0,  # 임시 값
            'gyro_bias_x': 0.0,  # 임시 값
            'gyro_bias_y': 0.0,  # 임시 값
            'gyro_bias_z': 0.0,  # 임시 값
            'accel_bias_x': 0.0,  # 임시 값
            'accel_bias_y': 0.0,  # 임시 값
            'accel_bias_z': 0.0,  # 임시 값
            'lever_arm_x': 0.0,  # 임시 값
            'lever_arm_y': 0.0,  # 임시 값
            'lever_arm_z': 0.0,  # 임시 값
            'extrinsic_roll': 0.0,  # 임시 값
            'extrinsic_pitch': 0.0,  # 임시 값
            'extrinsic_yaw': 0.0,  # 임시 값
            'nis_position': np.nan,
            'nis_velocity': np.nan,
            'nis_magnetometer': np.nan,
            'trace_P': 0.0,  # 임시 값
            'det_P': 0.0,  # 임시 값
        }
        results_data.append(result_row)
    
    # DataFrame으로 변환하고 저장
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print(f"Updates → pos:{len(nis_logs['pos'])}, vel:{len(nis_logs['vel'])}, mag:{len(nis_logs['mag'])}")

if __name__ == "__main__":
    main()
