#!/usr/bin/env python3
"""
Results Comparison and Visualization - Paper Format
논문 형식의 APEqF 필터 결과, EKF 결과, Ground truth를 비교하고 시각화합니다.
GCU 전략과 등변 측정 모델의 효과를 분석합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob

class ResultsComparator:
    """논문 형식의 결과를 비교하고 분석하는 클래스"""
    
    def __init__(self, data_dir="ekf_data", swap_xy=True, invert_z=True):
        self.data_dir = data_dir
        self.swap_xy = swap_xy
        self.invert_z = invert_z
        self.apeqf_results = None
        self.ekf_results = None
        self.ground_truth = None
        
        # 색상 설정 (논문 스타일)
        self.colors = {
            'apeqf': '#1f77b4',      # 파란색 (논문 Figure 1 스타일)
            'ekf': '#ff7f0e',        # 주황색
            'ground_truth': '#2ca02c' # 초록색
        }
        
        # 라벨 설정
        self.labels = {
            'apeqf': 'APEqF (Paper Format)',
            'ekf': 'EKF3',
            'ground_truth': 'Ground Truth'
        }
    def _swap_xy_and_flip_z(self, df, x, y, z):
        """DataFrame 열 기준으로 (x,y) 교환 및 z 부호 반전"""
        import pandas as pd  # noqa: F401
        if df is None:
            return
        try:
            cols = set(df.columns)
        except Exception:
            return
        # XY 스왑
        if {x, y}.issubset(cols) and getattr(self, 'swap_xy', False):
            df[x], df[y] = df[y].copy(), df[x].copy()
        # Z 부호 반전
        if z in cols and getattr(self, 'invert_z', False):
            df[z] = -df[z]

    def _apply_coord_transform(self):
        """좌표계 변환: (x,y) 축 교환(swap)과 z 부호 반전 옵션 적용
        APEqF 결과에만 적용하고 EKF/GT에는 적용하지 않습니다.
        """
        # APEqF only
        self._swap_xy_and_flip_z(self.apeqf_results, 'pos_x', 'pos_y', 'pos_z')
        self._swap_xy_and_flip_z(self.apeqf_results, 'vel_x', 'vel_y', 'vel_z')

    def load_data(self):
        """모든 결과 데이터를 로드합니다"""
        print("Loading comparison data...")
        
        # APEqF 결과 로드 (논문 형식)
        apeqf_files = glob.glob("apeqf_results_paper_format_*.csv")
        if not apeqf_files:
            # 일반 APEqF 결과도 시도
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
        
        self._apply_coord_transform()
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
            return None, None
        
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
                
                # 자세 오차 (각도 차이)
                att_error = np.sqrt(apeqf_row['att_roll']**2 + apeqf_row['att_pitch']**2)  # 단순화
                errors['apeqf']['attitude'].append(att_error)
            
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
                
                # 자세 오차 (EKF에는 자세 데이터가 다를 수 있음)
                att_error = 0.0  # 임시로 0 설정
                errors['ekf']['attitude'].append(att_error)
        
        # 통계 계산
        stats = {}
        for method in ['apeqf', 'ekf']:
            stats[method] = {}
            for metric in ['position', 'velocity', 'attitude']:
                if errors[method][metric]:
                    stats[method][metric] = {
                        'mean': np.mean(errors[method][metric]),
                        'std': np.std(errors[method][metric]),
                        'rmse': np.sqrt(np.mean(np.array(errors[method][metric])**2)),
                        'max': np.max(errors[method][metric]),
                        'median': np.median(errors[method][metric])
                    }
        
        return stats, errors
    
    def _find_closest_timestamp(self, df, target_timestamp):
        """가장 가까운 타임스탬프를 가진 인덱스를 찾습니다"""
        if df is None or len(df) == 0:
            return None
        
        time_diff = np.abs(df['timestamp'] - target_timestamp)
        return time_diff.idxmin()
    
    def plot_trajectory_comparison(self):
        """궤적 비교를 시각화합니다 (논문 Figure 1 스타일)"""
        print("Creating trajectory comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Trajectory Comparison: APEqF (Paper Format) vs EKF vs Ground Truth', fontsize=16)
        
        # XY 평면 궤적
        ax1 = axes[0, 0]
        if self.apeqf_results is not None:
            ax1.plot(self.apeqf_results['pos_x'], self.apeqf_results['pos_y'], 
                    color=self.colors['apeqf'], label=self.labels['apeqf'], linewidth=2)
        if self.ekf_results is not None:
            ax1.plot(self.ekf_results['x'], self.ekf_results['y'], 
                    color=self.colors['ekf'], label=self.labels['ekf'], linewidth=2, linestyle='--')
        if self.ground_truth is not None:
            ax1.plot(self.ground_truth['x'], self.ground_truth['y'], 
                    color=self.colors['ground_truth'], label=self.labels['ground_truth'], linewidth=3)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('XY Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # XZ 평면 궤적
        ax2 = axes[0, 1]
        if self.apeqf_results is not None:
            ax2.plot(self.apeqf_results['pos_x'], self.apeqf_results['pos_z'], 
                    color=self.colors['apeqf'], label=self.labels['apeqf'], linewidth=2)
        if self.ekf_results is not None:
            ax2.plot(self.ekf_results['x'], self.ekf_results['z'], 
                    color=self.colors['ekf'], label=self.labels['ekf'], linewidth=2, linestyle='--')
        if self.ground_truth is not None:
            ax2.plot(self.ground_truth['x'], self.ground_truth['z'], 
                    color=self.colors['ground_truth'], label=self.labels['ground_truth'], linewidth=3)
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Z Position (m)')
        ax2.set_title('XZ Trajectory')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 시간에 따른 X 위치
        ax3 = axes[1, 0]
        if self.apeqf_results is not None:
            time_apeqf = (self.apeqf_results['timestamp'] - self.apeqf_results['timestamp'].iloc[0]) / 1e6
            ax3.plot(time_apeqf, self.apeqf_results['pos_x'], 
                    color=self.colors['apeqf'], label=self.labels['apeqf'], linewidth=2)
        if self.ekf_results is not None:
            time_ekf = (self.ekf_results['timestamp'] - self.ekf_results['timestamp'].iloc[0]) / 1e6
            ax3.plot(time_ekf, self.ekf_results['x'], 
                    color=self.colors['ekf'], label=self.labels['ekf'], linewidth=2, linestyle='--')
        if self.ground_truth is not None:
            time_gt = (self.ground_truth['timestamp'] - self.ground_truth['timestamp'].iloc[0]) / 1e6
            ax3.plot(time_gt, self.ground_truth['x'], 
                    color=self.colors['ground_truth'], label=self.labels['ground_truth'], linewidth=3)
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('X Position (m)')
        ax3.set_title('X Position vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 시간에 따른 Y 위치
        ax4 = axes[1, 1]
        if self.apeqf_results is not None:
            ax4.plot(time_apeqf, self.apeqf_results['pos_y'], 
                    color=self.colors['apeqf'], label=self.labels['apeqf'], linewidth=2)
        if self.ekf_results is not None:
            ax4.plot(time_ekf, self.ekf_results['y'], 
                    color=self.colors['ekf'], label=self.labels['ekf'], linewidth=2, linestyle='--')
        if self.ground_truth is not None:
            ax4.plot(time_gt, self.ground_truth['y'], 
                    color=self.colors['ground_truth'], label=self.labels['ground_truth'], linewidth=3)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('Y Position vs Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('trajectory_comparison_paper_format.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_analysis(self, stats, errors):
        """오차 분석을 시각화합니다 (논문 스타일)"""
        print("Creating error analysis plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Error Analysis: APEqF (Paper Format) vs EKF', fontsize=16)
        
        # 위치 오차 분포
        ax1 = axes[0, 0]
        if errors['apeqf']['position'] and errors['ekf']['position']:
            ax1.hist(errors['apeqf']['position'], bins=50, alpha=0.7, density=True,
                    color=self.colors['apeqf'], label=f"APEqF (RMSE: {stats['apeqf']['position']['rmse']:.3f}m)")
            ax1.hist(errors['ekf']['position'], bins=50, alpha=0.7, density=True,
                    color=self.colors['ekf'], label=f"EKF (RMSE: {stats['ekf']['position']['rmse']:.3f}m)")
            ax1.set_xlabel('Position Error (m)')
            ax1.set_ylabel('Probability Density')
            ax1.set_title('Position Error Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 속도 오차 분포
        ax2 = axes[0, 1]
        if errors['apeqf']['velocity'] and errors['ekf']['velocity']:
            ax2.hist(errors['apeqf']['velocity'], bins=50, alpha=0.7, density=True,
                    color=self.colors['apeqf'], label=f"APEqF (RMSE: {stats['apeqf']['velocity']['rmse']:.3f}m/s)")
            ax2.hist(errors['ekf']['velocity'], bins=50, alpha=0.7, density=True,
                    color=self.colors['ekf'], label=f"EKF (RMSE: {stats['ekf']['velocity']['rmse']:.3f}m/s)")
            ax2.set_xlabel('Velocity Error (m/s)')
            ax2.set_ylabel('Probability Density')
            ax2.set_title('Velocity Error Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # NIS 분석 (논문의 진단 지표)
        ax3 = axes[0, 2]
        if 'nis_position' in self.apeqf_results.columns:
            time_apeqf = (self.apeqf_results['timestamp'] - self.apeqf_results['timestamp'].iloc[0]) / 1e6
            valid_nis = ~self.apeqf_results['nis_position'].isna()
            ax3.plot(time_apeqf[valid_nis], self.apeqf_results['nis_position'][valid_nis], 
                    color=self.colors['apeqf'], label='Position NIS', linewidth=1)
            ax3.axhline(y=3.0, color='r', linestyle='--', label='95% Confidence')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('NIS')
            ax3.set_title('Normalized Innovation Squared')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        
        # 시간에 따른 위치 오차
        ax4 = axes[1, 0]
        if self.ground_truth is not None and errors['apeqf']['position'] and errors['ekf']['position']:
            time_gt = (self.ground_truth['timestamp'] - self.ground_truth['timestamp'].iloc[0]) / 1e6
            if len(time_gt) == len(errors['apeqf']['position']):
                ax4.plot(time_gt, errors['apeqf']['position'], 
                        color=self.colors['apeqf'], label='APEqF', linewidth=2)
            if len(time_gt) == len(errors['ekf']['position']):
                ax4.plot(time_gt, errors['ekf']['position'], 
                        color=self.colors['ekf'], label='EKF', linewidth=2, linestyle='--')
            
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Position Error (m)')
            ax4.set_title('Position Error vs Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 공분산 추적 (논문의 진단 지표)
        ax5 = axes[1, 1]
        if 'trace_P' in self.apeqf_results.columns:
            ax5.semilogy(time_apeqf, self.apeqf_results['trace_P'], 
                        color=self.colors['apeqf'], label='Trace(P)', linewidth=2)
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Trace(P)')
            ax5.set_title('Covariance Trace Evolution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 통계 요약 테이블
        ax6 = axes[1, 2]
        ax6.axis('tight')
        ax6.axis('off')
        
        # 통계 테이블 생성
        table_data = []
        metrics = ['position', 'velocity']
        for method in ['apeqf', 'ekf']:
            for metric in metrics:
                if metric in stats[method]:
                    row = [
                        method.upper(),
                        metric.capitalize(),
                        f"{stats[method][metric]['rmse']:.4f}",
                        f"{stats[method][metric]['mean']:.4f}",
                        f"{stats[method][metric]['std']:.4f}",
                        f"{stats[method][metric]['median']:.4f}",
                        f"{stats[method][metric]['max']:.4f}"
                    ]
                    table_data.append(row)
        
        table = ax6.table(cellText=table_data,
                         colLabels=['Method', 'Metric', 'RMSE', 'Mean', 'Std', 'Median', 'Max'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        ax6.set_title('Error Statistics Summary')
        
        plt.tight_layout()
        plt.savefig('error_analysis_paper_format.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_calibration_convergence(self):
        """캘리브레이션 매개변수 수렴 과정을 시각화합니다 (논문 스타일)"""
        print("Creating calibration convergence plots...")
        
        if self.apeqf_results is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Self-Calibration Convergence (Paper Format)', fontsize=16)
        
        time_apeqf = (self.apeqf_results['timestamp'] - self.apeqf_results['timestamp'].iloc[0]) / 1e6
        
        # 자이로 바이어스 수렴
        ax1 = axes[0, 0]
        ax1.plot(time_apeqf, self.apeqf_results['gyro_bias_x'], label='X', linewidth=2)
        ax1.plot(time_apeqf, self.apeqf_results['gyro_bias_y'], label='Y', linewidth=2)
        ax1.plot(time_apeqf, self.apeqf_results['gyro_bias_z'], label='Z', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Gyro Bias (rad/s)')
        ax1.set_title('Gyroscope Bias Estimation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 가속도계 바이어스 수렴
        ax2 = axes[0, 1]
        ax2.plot(time_apeqf, self.apeqf_results['accel_bias_x'], label='X', linewidth=2)
        ax2.plot(time_apeqf, self.apeqf_results['accel_bias_y'], label='Y', linewidth=2)
        ax2.plot(time_apeqf, self.apeqf_results['accel_bias_z'], label='Z', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Accel Bias (m/s²)')
        ax2.set_title('Accelerometer Bias Estimation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 레버암 수렴
        ax3 = axes[1, 0]
        ax3.plot(time_apeqf, self.apeqf_results['lever_arm_x'], label='X', linewidth=2)
        ax3.plot(time_apeqf, self.apeqf_results['lever_arm_y'], label='Y', linewidth=2)
        ax3.plot(time_apeqf, self.apeqf_results['lever_arm_z'], label='Z', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Lever Arm (m)')
        ax3.set_title('GNSS Lever Arm Estimation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 자력계 외부 회전 수렴
        ax4 = axes[1, 1]
        ax4.plot(time_apeqf, np.rad2deg(self.apeqf_results['extrinsic_roll']), 
                label='Roll', linewidth=2)
        ax4.plot(time_apeqf, np.rad2deg(self.apeqf_results['extrinsic_pitch']), 
                label='Pitch', linewidth=2)
        ax4.plot(time_apeqf, np.rad2deg(self.apeqf_results['extrinsic_yaw']), 
                label='Yaw', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Extrinsic Rotation (deg)')
        ax4.set_title('Magnetometer Extrinsic Calibration')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('calibration_convergence_paper_format.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_statistics(self, stats):
        """통계 결과를 출력합니다 (논문 스타일)"""
        print("\n" + "="*80)
        print("ERROR STATISTICS SUMMARY (Paper Format)")
        print("="*80)
        
        for method in ['apeqf', 'ekf']:
            method_name = "APEqF (Paper Format)" if method == 'apeqf' else "EKF3"
            print(f"\n{method_name} Results:")
            print("-" * 40)
            for metric in ['position', 'velocity']:
                if metric in stats[method]:
                    s = stats[method][metric]
                    print(f"{metric.capitalize()}:")
                    print(f"  RMSE:   {s['rmse']:.6f}")
                    print(f"  Mean:   {s['mean']:.6f}")
                    print(f"  Std:    {s['std']:.6f}")
                    print(f"  Median: {s['median']:.6f}")
                    print(f"  Max:    {s['max']:.6f}")
        
        # 성능 개선 비교
        if 'apeqf' in stats and 'ekf' in stats:
            print(f"\n{'Performance Improvement:'}")
            print("-" * 40)
            for metric in ['position', 'velocity']:
                if metric in stats['apeqf'] and metric in stats['ekf']:
                    apeqf_rmse = stats['apeqf'][metric]['rmse']
                    ekf_rmse = stats['ekf'][metric]['rmse']
                    improvement = ((ekf_rmse - apeqf_rmse) / ekf_rmse) * 100
                    print(f"{metric.capitalize()} RMSE improvement: {improvement:.1f}%")
        
        print("\n" + "="*80)

def main():
    """메인 함수"""
    print("=== Results Comparison and Analysis (Paper Format) ===")
    
    # 결과 비교기 초기화
    comparator = ResultsComparator(swap_xy=True, invert_z=True)
    
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
    comparator.plot_calibration_convergence()
    
    print("\n=== Comparison completed ===")
    print("Generated plots:")
    print("- trajectory_comparison_paper_format.png")
    print("- error_analysis_paper_format.png")
    print("- calibration_convergence_paper_format.png")

if __name__ == "__main__":
    main()
