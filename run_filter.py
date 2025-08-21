#!/usr/bin/env python3
"""
APEqF Filter Runner
센서 데이터를 읽어서 APEqF 필터를 실행하고 결과를 저장합니다.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt

# APEqF 모듈들 import
from Final_EqF import SE23_se3_R3_SO3_EqF
from System import State, InputSpace, input_from_vector
from Symmetry import SymGroup
from pylie import SE23, SO3, SE3

class DataLoader:
    """센서 데이터를 로드하고 전처리하는 클래스"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.sensor_data = None
        self.gps_data = None
        self.mag_data = None
        self.ground_truth = None
        
    def load_data(self):
        """모든 센서 데이터를 로드합니다"""
        print("Loading sensor data...")
        
        # IMU + 센서 통합 데이터 (가장 큰 파일)
        sensor_file = os.path.join(self.data_dir, "sensors", 
                                 "log_4_2025-7-7-09-05-20_sensor_combined_0.csv")
        if os.path.exists(sensor_file):
            # 파일이 너무 크므로 필요한 컬럼만 읽기
            self.sensor_data = pd.read_csv(sensor_file, usecols=[
                'timestamp',
                'gyro_rad[0]', 'gyro_rad[1]', 'gyro_rad[2]',
                'accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]'
            ])
            print(f"Loaded {len(self.sensor_data)} IMU samples")
        
        # GPS 데이터
        gps_file = os.path.join(self.data_dir, "sensors", 
                               "log_4_2025-7-7-09-05-20_sensor_gps_0.csv")
        if os.path.exists(gps_file):
            self.gps_data = pd.read_csv(gps_file)
            print(f"Loaded {len(self.gps_data)} GPS samples")
        
        # 자력계 데이터
        mag_file = os.path.join(self.data_dir, "sensors", 
                               "log_4_2025-7-7-09-05-20_sensor_mag_0.csv")
        if os.path.exists(mag_file):
            self.mag_data = pd.read_csv(mag_file)
            print(f"Loaded {len(self.mag_data)} magnetometer samples")
        
        # Ground truth 데이터
        gt_file = os.path.join(self.data_dir, "ground_truth", 
                              "log_4_2025-7-7-09-05-20_vehicle_local_position_groundtruth_0.csv")
        if os.path.exists(gt_file):
            self.ground_truth = pd.read_csv(gt_file)
            print(f"Loaded {len(self.ground_truth)} ground truth samples")
    
    def get_synchronized_data(self):
        """타임스탬프를 기준으로 데이터를 동기화합니다"""
        if self.sensor_data is None:
            return None, None, None, None
        
        # 타임스탬프를 기준으로 정렬
        self.sensor_data = self.sensor_data.sort_values('timestamp')
        
        # GPS와 자력계 데이터를 센서 데이터와 동기화
        synced_data = []
        
        for _, row in self.sensor_data.iterrows():
            timestamp = row['timestamp']
            
            # GPS 데이터 찾기 (가장 가까운 타임스탬프)
            gps_row = self._find_closest_timestamp(self.gps_data, timestamp) if self.gps_data is not None else None
            
            # 자력계 데이터 찾기
            mag_row = self._find_closest_timestamp(self.mag_data, timestamp) if self.mag_data is not None else None
            
            # Ground truth 데이터 찾기
            gt_row = self._find_closest_timestamp(self.ground_truth, timestamp) if self.ground_truth is not None else None
            
            synced_data.append({
                'timestamp': timestamp,
                'gyro': np.array([row['gyro_rad[0]'], row['gyro_rad[1]'], row['gyro_rad[2]']]),
                'accel': np.array([row['accelerometer_m_s2[0]'], row['accelerometer_m_s2[1]'], row['accelerometer_m_s2[2]']]),
                'mag': np.array([mag_row['x'], mag_row['y'], mag_row['z']]) if mag_row is not None else None,
                'gps_pos': None,  # GPS 위치 데이터 없음
                'gps_vel': None,  # GPS 속도 데이터 없음
                'gt_pos': np.array([gt_row['x'], gt_row['y'], gt_row['z']]) if gt_row is not None else None,
                'gt_vel': np.array([gt_row['vx'], gt_row['vy'], gt_row['vz']]) if gt_row is not None else None
            })
        
        return synced_data
    
    def _find_closest_timestamp(self, df, target_timestamp):
        """가장 가까운 타임스탬프를 가진 행을 찾습니다"""
        if df is None or len(df) == 0:
            return None
        
        # 타임스탬프 차이 계산
        time_diff = np.abs(df['timestamp'] - target_timestamp)
        closest_idx = time_diff.idxmin()
        
        return df.iloc[closest_idx]

class APEqFRunner:
    """APEqF 필터를 실행하는 클래스"""
    
    def __init__(self):
        # 필터 초기화
        self.filter = SE23_se3_R3_SO3_EqF(
            initial_att_noise=0.1,      # 초기 자세 노이즈 (rad)
            initial_vel_noise=0.5,      # 초기 속도 노이즈 (m/s)
            initial_pos_noise=1.0,      # 초기 위치 노이즈 (m)
            initial_bias_noise=0.01,    # 초기 바이어스 노이즈
            initial_lever_noise=0.1,    # 초기 레버암 노이즈 (m)
            initial_extrinsic_noise=0.1, # 초기 외부 회전 노이즈 (rad)
            curvature_correction=True,   # 곡률 보정 사용
            equivariant_output=True      # 등변 출력 사용
        )
        
        # 결과 저장용 리스트
        self.results = []
        
        # 노이즈 파라미터
        self.omega_noise = 0.01         # 자이로 노이즈 (rad/s)
        self.acc_noise = 0.1            # 가속도계 노이즈 (m/s²)
        self.bias_noise = 0.001         # 바이어스 랜덤워크 노이즈
        self.lever_noise = 0.001        # 레버암 드리프트 노이즈 (m/s)
        self.extrinsic_noise = 0.001    # 외부 회전 드리프트 노이즈 (rad/s)
        
        # 측정 노이즈
        self.gps_pos_noise = 1.0        # GPS 위치 노이즈 (m)
        self.gps_vel_noise = 0.5        # GPS 속도 노이즈 (m/s)
        self.mag_noise = 0.1            # 자력계 노이즈 (Gauss)
        
        # 지구 자기장 (San Jose 근처)
        self.mag_field = np.array([0.2, 0.0, -0.5])  # North, East, Down (Gauss)
    
    def run_filter(self, synced_data):
        """필터를 실행합니다"""
        print("Running APEqF filter...")
        
        for i, data in enumerate(synced_data):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(synced_data)}")
            
            timestamp = data['timestamp']
            
            # IMU 데이터로 propagation
            imu_input = np.vstack([
                data['gyro'].reshape(3, 1),      # 자이로
                data['accel'].reshape(3, 1),     # 가속도
                np.zeros((3, 1)),       # μ (propagation에서는 0)
                np.zeros((6, 1)),       # τ (바이어스 랜덤워크)
                np.zeros((3, 1)),       # μ (레버암 드리프트)
                np.zeros((3, 1))        # wM (외부 회전 드리프트)
            ])
            
            # Propagation
            success = self.filter.propagate(
                t=timestamp,
                vel=imu_input,
                omega_noise=self.omega_noise,
                acc_noise=self.acc_noise,
                bias_noise=self.bias_noise,
                lever_noise=self.lever_noise,
                extrinsic_noise=self.extrinsic_noise
            )
            
            if not success:
                continue
            
            # GPS 위치 측정 업데이트
            if data['gps_pos'] is not None:
                nis_pos = self.filter.update_position(
                    y=data['gps_pos'].reshape(3, 1),
                    R_meas=np.eye(3) * self.gps_pos_noise**2
                )
            else:
                nis_pos = None
            
            # GPS 속도 측정 업데이트
            if data['gps_vel'] is not None:
                nis_vel = self.filter.update_velocity(
                    y=data['gps_vel'].reshape(3, 1),
                    R_meas=np.eye(3) * self.gps_vel_noise**2
                )
            else:
                nis_vel = None
            
            # 자력계 측정 업데이트
            if data['mag'] is not None:
                nis_mag = self.filter.update_magnetometer(
                    y=data['mag'].reshape(3, 1),
                    R_meas=np.eye(3) * self.mag_noise**2,
                    m_G=self.mag_field.reshape(3, 1)
                )
            else:
                nis_mag = None
            
            # 현재 추정 상태 저장
            R, p, v, bw, ba, t, S = self.filter.getEstimate()
            
            result = {
                'timestamp': timestamp,
                'position': p.flatten(),
                'velocity': v.flatten(),
                'attitude': R,
                'gyro_bias': bw.flatten(),
                'accel_bias': ba.flatten(),
                'lever_arm': t.flatten(),
                'extrinsic_rotation': S,
                'nis_position': nis_pos,
                'nis_velocity': nis_vel,
                'nis_magnetometer': nis_mag
            }
            
            self.results.append(result)
        
        print(f"Filter completed. Processed {len(self.results)} samples.")
    
    def save_results(self, output_file):
        """결과를 CSV 파일로 저장합니다"""
        print(f"Saving results to {output_file}...")
        
        # 결과를 DataFrame으로 변환
        df_data = []
        for result in self.results:
            row = {
                'timestamp': result['timestamp'],
                'pos_x': result['position'][0],
                'pos_y': result['position'][1],
                'pos_z': result['position'][2],
                'vel_x': result['velocity'][0],
                'vel_y': result['velocity'][1],
                'vel_z': result['velocity'][2],
                'att_roll': np.arctan2(result['attitude'][2, 1], result['attitude'][2, 2]),
                'att_pitch': np.arcsin(-result['attitude'][2, 0]),
                'att_yaw': np.arctan2(result['attitude'][1, 0], result['attitude'][0, 0]),
                'gyro_bias_x': result['gyro_bias'][0],
                'gyro_bias_y': result['gyro_bias'][1],
                'gyro_bias_z': result['gyro_bias'][2],
                'accel_bias_x': result['accel_bias'][0],
                'accel_bias_y': result['accel_bias'][1],
                'accel_bias_z': result['accel_bias'][2],
                'lever_arm_x': result['lever_arm'][0],
                'lever_arm_y': result['lever_arm'][1],
                'lever_arm_z': result['lever_arm'][2],
                'extrinsic_roll': np.arctan2(result['extrinsic_rotation'][2, 1], 
                                           result['extrinsic_rotation'][2, 2]),
                'extrinsic_pitch': np.arcsin(-result['extrinsic_rotation'][2, 0]),
                'extrinsic_yaw': np.arctan2(result['extrinsic_rotation'][1, 0], 
                                          result['extrinsic_rotation'][0, 0]),
                'nis_position': result['nis_position'] if result['nis_position'] is not None else np.nan,
                'nis_velocity': result['nis_velocity'] if result['nis_velocity'] is not None else np.nan,
                'nis_magnetometer': result['nis_magnetometer'] if result['nis_magnetometer'] is not None else np.nan
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

def main():
    """메인 함수"""
    print("=== APEqF Filter Runner ===")
    
    # 데이터 디렉토리
    data_dir = "ekf_data"
    
    # 데이터 로더 초기화 및 데이터 로드
    loader = DataLoader(data_dir)
    loader.load_data()
    
    # 동기화된 데이터 가져오기
    synced_data = loader.get_synchronized_data()
    if synced_data is None:
        print("Failed to load sensor data!")
        return
    
    print(f"Loaded {len(synced_data)} synchronized data samples")
    
    # APEqF 필터 실행
    runner = APEqFRunner()
    runner.run_filter(synced_data)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"apeqf_results_{timestamp}.csv"
    runner.save_results(output_file)
    
    print("=== Filter execution completed ===")
    print(f"Results saved to: {output_file}")
    print(f"Total samples processed: {len(runner.results)}")

if __name__ == "__main__":
    main()
