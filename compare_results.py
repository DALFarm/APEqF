#!/usr/bin/env python3
"""
Results Comparison and Visualization
APEqF 필터 결과, EKF 결과, Ground truth를 비교하고 시각화합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob

class ResultsComparator:
    """결과를 비교하고 분석하는 클래스"""
    
    def __init__(self, data_dir="ekf_data"):
        self.data_dir = data_dir
        self.apeqf_results = None
        self.ekf_results = None
        self.ground_truth = None
        
        # 색상 설정
        self.colors = {
            'apeqf': '#1f77b4',      # 파란색
            'ekf': '#ff7f0e',        # 주황색
            'ground_truth': '#2ca02c' # 초록색
        }
        
        # 라벨 설정
        self.labels = {
            'apeqf': 'APEqF',
            'ekf': 'EKF',
            'ground_truth': 'Ground Truth'
        }
    
    def load_data(self):
        """모든 결과 데이터를 로드합니다"""
        print("Loading comparison data...")
        
        # APEqF 결과 로드 (가장 최근 파일)
        apeqf_files = glob.glob("apeqf_results_*.csv")
        if apeqf_files:
            latest_apeqf = max(apeqf_files, key=os.path.getctime)
            self.apeqf_results = pd.read_csv(latest_apeqf)
            print(f"Loaded APEqF results: {latest_apeqf}")
        else:
            print("No APEqF results found!")
            return False
        
        # EKF 결과 로드
        ekf_file = os.path.join(self.data_dir, "ekf_results", 
                               "log_4_2025-7-7-09-05-20_estimator_local_position_0.csv")
        if os.path.exists(ekf_file):
            self.ekf_results = pd.read_csv(ekf_file)
            print(f"Loaded EKF results: {ekf_file}")
        else:
            print("EKF results not found!")
            return False
        
        # Ground truth 로드
        gt_file = os.path.join(self.data_dir, "ground_truth", 
                              "log_4_2025-7-7-09-05-20_vehicle_local_position_groundtruth_0.csv")
        if os.path.exists(gt_file):
            self.ground_truth = pd.read_csv(gt_file)
            print(f"Loaded ground truth: {gt_file}")
        else:
            print("Ground truth not found!")
            return False
        
        return True
    
    def synchronize_data(self):
        """데이터를 타임스탬프 기준으로 동기화합니다"""
        print("Synchronizing data...")
        
        # 타임스탬프를 기준으로 데이터 정렬
        if self.apeqf_results is not None:
            self.apeqf_results = self.apeqf_results.sort_values('timestamp')
        
        if self.ekf_results is not None:
            self.ekf_results = self.ekf_results.sort_values('timestamp')
        
        if self.ground_truth is not None:
            self.ground_truth = self.ground_truth.sort_values('timestamp')
        
        # 공통 타임스탬프 범위 찾기
        timestamps = []
        if self.apeqf_results is not None:
            timestamps.extend(self.apeqf_results['timestamp'].tolist())
        if self.ekf_results is not None:
            timestamps.extend(self.ekf_results['timestamp'].tolist())
        if self.ground_truth is not None:
            timestamps.extend(self.ground_truth['timestamp'].tolist())
        
        if not timestamps:
            return False
        
        min_time = min(timestamps)
        max_time = max(timestamps)
        print(f"Time range: {min_time} to {max_time}")
        
        return True
    
    def calculate_errors(self):
        """APEqF와 EKF의 Ground truth 대비 오차를 계산합니다"""
        print("Calculating errors...")
        
        if self.apeqf_results is None or self.ekf_results is None or self.ground_truth is None:
            return
        
        # Ground truth를 기준으로 오차 계산
        errors = {
            'apeqf': {'position': [], 'velocity': [], 'attitude': []},
            'ekf': {'position': [], 'velocity': [], 'attitude': []}
        }
        
        # 각 타임스탬프에서 오차 계산
        for _, gt_row in self.ground_truth.iterrows():
            timestamp = gt_row['timestamp']
            
            # APEqF 결과에서 가장 가까운 타임스탬프 찾기
            apeqf_idx = self._find_closest_timestamp(self.apeqf_results, timestamp)
            if apeqf_idx is not None:
                apeqf_row = self.apeqf_results.iloc[apeqf_idx]
                
                # 위치 오차
                gt_pos = np.array([gt_row['x'], gt_row['y'], gt_row['z']])
                apeqf_pos = np.array([apeqf_row['pos_x'], apeqf_row['pos_y'], apeqf_row['pos_z']])
                pos_error = np.linalg.norm(gt_pos - apeqf_pos)
                errors['apeqf']['position'].append(pos_error)
                
                # 속도 오차
                gt_vel = np.array([gt_row['vx'], gt_row['vy'], gt_row['vz']])
                apeqf_vel = np.array([apeqf_row['vel_x'], apeqf_row['vel_y'], apeqf_row['vel_z']])
                vel_error = np.linalg.norm(gt_vel - apeqf_vel)
                errors['apeqf']['velocity'].append(vel_error)
            
            # EKF 결과에서 가장 가까운 타임스탬프 찾기
            ekf_idx = self._find_closest_timestamp(self.ekf_results, timestamp)
            if ekf_idx is not None:
                ekf_row = self.ekf_results.iloc[ekf_idx]
                
                # 위치 오차
                ekf_pos = np.array([ekf_row['x'], ekf_row['y'], ekf_row['z']])
                pos_error = np.linalg.norm(gt_pos - ekf_pos)
                errors['ekf']['position'].append(pos_error)
                
                # 속도 오차
                ekf_vel = np.array([ekf_row['vx'], ekf_row['vy'], ekf_row['vz']])
                vel_error = np.linalg.norm(gt_vel - ekf_vel)
                errors['ekf']['velocity'].append(vel_error)
        
        # 통계 계산
        stats = {}
        for method in ['apeqf', 'ekf']:
            stats[method] = {}
            for metric in ['position', 'velocity']:
                if errors[method][metric]:
                    stats[method][metric] = {
                        'mean': np.mean(errors[method][metric]),
                        'std': np.std(errors[method][metric]),
                        'rmse': np.sqrt(np.mean(np.array(errors[method][metric])**2)),
                        'max': np.max(errors[method][metric])
                    }
        
        return stats, errors
    
    def _find_closest_timestamp(self, df, target_timestamp):
        """가장 가까운 타임스탬프를 가진 인덱스를 찾습니다"""
        if df is None or len(df) == 0:
            return None
        
        time_diff = np.abs(df['timestamp'] - target_timestamp)
        return time_diff.idxmin()
    
    def plot_trajectory_comparison(self):
        """궤적 비교를 시각화합니다"""
        print("Creating trajectory comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trajectory Comparison: APEqF vs EKF vs Ground Truth', fontsize=16)
        
        # XY 평면 궤적
        ax1 = axes[0, 0]
        if self.apeqf_results is not None:
            ax1.plot(self.apeqf_results['pos_x'], self.apeqf_results['pos_y'], 
                    color=self.colors['apeqf'], label=self.labels['apeqf'], linewidth=2)
        if self.ekf_results is not None:
            ax1.plot(self.ekf_results['x'], self.ekf_results['y'], 
                    color=self.colors['ekf'], label=self.labels['ekf'], linewidth=2)
        if self.ground_truth is not None:
            ax1.plot(self.ground_truth['x'], self.ground_truth['y'], 
                    color=self.colors['ground_truth'], label=self.labels['ground_truth'], linewidth=2)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('XY Trajectory')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')
        
        # XZ 평면 궤적
        ax2 = axes[0, 1]
        if self.apeqf_results is not None:
            ax2.plot(self.apeqf_results['pos_x'], self.apeqf_results['pos_z'], 
                    color=self.colors['apeqf'], label=self.labels['apeqf'], linewidth=2)
        if self.ekf_results is not None:
            ax2.plot(self.ekf_results['x'], self.ekf_results['z'], 
                    color=self.colors['ekf'], label=self.labels['ekf'], linewidth=2)
        if self.ground_truth is not None:
            ax2.plot(self.ground_truth['x'], self.ground_truth['z'], 
                    color=self.colors['ground_truth'], label=self.labels['ground_truth'], linewidth=2)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Z Position (m)')
        ax2.set_title('XZ Trajectory')
        ax2.legend()
        ax2.grid(True)
        
        # 시간에 따른 X 위치
        ax3 = axes[1, 0]
        if self.apeqf_results is not None:
            ax3.plot(self.apeqf_results['timestamp'], self.apeqf_results['pos_x'], 
                    color=self.colors['apeqf'], label=self.labels['apeqf'], linewidth=2)
        if self.ekf_results is not None:
            ax3.plot(self.ekf_results['timestamp'], self.ekf_results['x'], 
                    color=self.colors['ekf'], label=self.labels['ekf'], linewidth=2)
        if self.ground_truth is not None:
            ax3.plot(self.ground_truth['timestamp'], self.ground_truth['x'], 
                    color=self.colors['ground_truth'], label=self.labels['ground_truth'], linewidth=2)
        
        ax3.set_xlabel('Timestamp')
        ax3.set_ylabel('X Position (m)')
        ax3.set_title('X Position vs Time')
        ax3.legend()
        ax3.grid(True)
        
        # 시간에 따른 Y 위치
        ax4 = axes[1, 1]
        if self.apeqf_results is not None:
            ax4.plot(self.apeqf_results['timestamp'], self.apeqf_results['pos_y'], 
                    color=self.colors['apeqf'], label=self.labels['apeqf'], linewidth=2)
        if self.ekf_results is not None:
            ax4.plot(self.ekf_results['timestamp'], self.ekf_results['y'], 
                    color=self.colors['ekf'], label=self.labels['ekf'], linewidth=2)
        if self.ground_truth is not None:
            ax4.plot(self.ground_truth['timestamp'], self.ground_truth['y'], 
                    color=self.colors['ground_truth'], label=self.labels['ground_truth'], linewidth=2)
        
        ax4.set_xlabel('Timestamp')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('Y Position vs Time')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_analysis(self, stats, errors):
        """오차 분석을 시각화합니다"""
        print("Creating error analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Error Analysis: APEqF vs EKF', fontsize=16)
        
        # 위치 오차 비교
        ax1 = axes[0, 0]
        if errors['apeqf']['position'] and errors['ekf']['position']:
            ax1.hist(errors['apeqf']['position'], bins=50, alpha=0.7, 
                    color=self.colors['apeqf'], label=f"APEqF (RMSE: {stats['apeqf']['position']['rmse']:.3f}m)")
            ax1.hist(errors['ekf']['position'], bins=50, alpha=0.7, 
                    color=self.colors['ekf'], label=f"EKF (RMSE: {stats['ekf']['position']['rmse']:.3f}m)")
            ax1.set_xlabel('Position Error (m)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Position Error Distribution')
            ax1.legend()
            ax1.grid(True)
        
        # 속도 오차 비교
        ax2 = axes[0, 1]
        if errors['apeqf']['velocity'] and errors['ekf']['velocity']:
            ax2.hist(errors['apeqf']['velocity'], bins=50, alpha=0.7, 
                    color=self.colors['apeqf'], label=f"APEqF (RMSE: {stats['apeqf']['velocity']['rmse']:.3f}m/s)")
            ax2.hist(errors['ekf']['velocity'], bins=50, alpha=0.7, 
                    color=self.colors['ekf'], label=f"EKF (RMSE: {stats['ekf']['velocity']['rmse']:.3f}m/s)")
            ax2.set_xlabel('Velocity Error (m/s)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Velocity Error Distribution')
            ax2.legend()
            ax2.grid(True)
        
        # 시간에 따른 위치 오차
        ax3 = axes[1, 0]
        if self.ground_truth is not None and errors['apeqf']['position'] and errors['ekf']['position']:
            # Ground truth 타임스탬프에 맞춰 오차 플롯
            gt_timestamps = self.ground_truth['timestamp'].values
            if len(gt_timestamps) == len(errors['apeqf']['position']):
                ax3.plot(gt_timestamps, errors['apeqf']['position'], 
                        color=self.colors['apeqf'], label='APEqF', linewidth=2)
            if len(gt_timestamps) == len(errors['ekf']['position']):
                ax3.plot(gt_timestamps, errors['ekf']['position'], 
                        color=self.colors['ekf'], label='EKF', linewidth=2)
            
            ax3.set_xlabel('Timestamp')
            ax3.set_ylabel('Position Error (m)')
            ax3.set_title('Position Error vs Time')
            ax3.legend()
            ax3.grid(True)
        
        # 통계 요약 테이블
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # 통계 테이블 생성
        table_data = []
        for method in ['apeqf', 'ekf']:
            for metric in ['position', 'velocity']:
                if metric in stats[method]:
                    row = [
                        method.upper(),
                        metric.capitalize(),
                        f"{stats[method][metric]['rmse']:.3f}",
                        f"{stats[method][metric]['mean']:.3f}",
                        f"{stats[method][metric]['std']:.3f}",
                        f"{stats[method][metric]['max']:.3f}"
                    ]
                    table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Method', 'Metric', 'RMSE', 'Mean', 'Std', 'Max'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax4.set_title('Error Statistics Summary')
        
        plt.tight_layout()
        plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_additional_metrics(self):
        """추가 메트릭들을 시각화합니다"""
        print("Creating additional metric plots...")
        
        if self.apeqf_results is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('APEqF Additional Metrics', fontsize=16)
        
        # 자세 (Roll, Pitch, Yaw)
        ax1 = axes[0, 0]
        ax1.plot(self.apeqf_results['timestamp'], np.rad2deg(self.apeqf_results['att_roll']), 
                label='Roll', linewidth=2)
        ax1.plot(self.apeqf_results['timestamp'], np.rad2deg(self.apeqf_results['att_pitch']), 
                label='Pitch', linewidth=2)
        ax1.plot(self.apeqf_results['timestamp'], np.rad2deg(self.apeqf_results['att_yaw']), 
                label='Yaw', linewidth=2)
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Angle (degrees)')
        ax1.set_title('Attitude Estimation')
        ax1.legend()
        ax1.grid(True)
        
        # 바이어스 추정
        ax2 = axes[0, 1]
        ax2.plot(self.apeqf_results['timestamp'], self.apeqf_results['gyro_bias_x'], 
                label='Gyro X', linewidth=2)
        ax2.plot(self.apeqf_results['timestamp'], self.apeqf_results['gyro_bias_y'], 
                label='Gyro Y', linewidth=2)
        ax2.plot(self.apeqf_results['timestamp'], self.apeqf_results['gyro_bias_z'], 
                label='Gyro Z', linewidth=2)
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Bias (rad/s)')
        ax2.set_title('Gyroscope Bias Estimation')
        ax2.legend()
        ax2.grid(True)
        
        # 레버암 추정
        ax3 = axes[1, 0]
        ax3.plot(self.apeqf_results['timestamp'], self.apeqf_results['lever_arm_x'], 
                label='X', linewidth=2)
        ax3.plot(self.apeqf_results['timestamp'], self.apeqf_results['lever_arm_y'], 
                label='Y', linewidth=2)
        ax3.plot(self.apeqf_results['timestamp'], self.apeqf_results['lever_arm_z'], 
                label='Z', linewidth=2)
        ax3.set_xlabel('Timestamp')
        ax3.set_ylabel('Lever Arm (m)')
        ax3.set_title('GNSS Lever Arm Estimation')
        ax3.legend()
        ax3.grid(True)
        
        # NIS (Normalized Innovation Squared)
        ax4 = axes[1, 1]
        if 'nis_position' in self.apeqf_results.columns:
            ax4.plot(self.apeqf_results['timestamp'], self.apeqf_results['nis_position'], 
                    label='Position', linewidth=2)
        if 'nis_velocity' in self.apeqf_results.columns:
            ax4.plot(self.apeqf_results['timestamp'], self.apeqf_results['nis_velocity'], 
                    label='Velocity', linewidth=2)
        if 'nis_magnetometer' in self.apeqf_results.columns:
            ax4.plot(self.apeqf_results['timestamp'], self.apeqf_results['nis_magnetometer'], 
                    label='Magnetometer', linewidth=2)
        
        ax4.axhline(y=3.0, color='r', linestyle='--', label='95% Confidence')
        ax4.set_xlabel('Timestamp')
        ax4.set_ylabel('NIS')
        ax4.set_title('Normalized Innovation Squared')
        ax4.legend()
        ax4.grid(True)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('additional_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_statistics(self, stats):
        """통계 결과를 출력합니다"""
        print("\n" + "="*60)
        print("ERROR STATISTICS SUMMARY")
        print("="*60)
        
        for method in ['apeqf', 'ekf']:
            print(f"\n{method.upper()} Results:")
            print("-" * 30)
            for metric in ['position', 'velocity']:
                if metric in stats[method]:
                    s = stats[method][metric]
                    print(f"{metric.capitalize()}:")
                    print(f"  RMSE: {s['rmse']:.4f}")
                    print(f"  Mean: {s['mean']:.4f}")
                    print(f"  Std:  {s['std']:.4f}")
                    print(f"  Max:  {s['max']:.4f}")
        
        print("\n" + "="*60)

def main():
    """메인 함수"""
    print("=== Results Comparison and Analysis ===")
    
    # 결과 비교기 초기화
    comparator = ResultsComparator()
    
    # 데이터 로드
    if not comparator.load_data():
        print("Failed to load data!")
        return
    
    # 데이터 동기화
    if not comparator.synchronize_data():
        print("Failed to synchronize data!")
        return
    
    # 오차 계산
    stats, errors = comparator.calculate_errors()
    if stats is None:
        print("Failed to calculate errors!")
        return
    
    # 통계 출력
    comparator.print_statistics(stats)
    
    # 시각화
    comparator.plot_trajectory_comparison()
    comparator.plot_error_analysis(stats, errors)
    comparator.plot_additional_metrics()
    
    print("\n=== Comparison completed ===")
    print("Generated plots:")
    print("- trajectory_comparison.png")
    print("- error_analysis.png")
    print("- additional_metrics.png")

if __name__ == "__main__":
    main()
