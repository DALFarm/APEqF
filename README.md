# APEqF Filter Execution Guide

이 가이드는 APEqF 필터를 실행하고 결과를 비교하는 방법을 설명합니다.

## 파일 구조

```
APEqF/
├── Final_EqF.py          # APEqF 필터 구현
├── Symmetry.py           # 대칭 그룹 정의
├── System.py             # 시스템 상태 및 입력 정의
├── run_filter.py         # 필터 실행 코드
├── compare_results.py    # 결과 비교 및 시각화 코드
├── requirements.txt      # 필요한 패키지 목록
├── ekf_data/            # 센서 데이터 및 EKF 결과
└── README_execution.md   # 이 파일
```

## 설치 및 설정

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. pylie 패키지 설치

`pylie` 패키지가 설치되어 있지 않다면:

```bash
pip install pylie
```

또는 소스에서 설치:

```bash
git clone https://github.com/utiasSTARS/pylie.git
cd pylie
pip install -e .
```

## 실행 방법

### 1단계: APEqF 필터 실행

센서 데이터를 사용하여 APEqF 필터를 실행합니다:

```bash
python run_filter.py
```

이 스크립트는:
- `ekf_data/sensors/`에서 IMU, GPS, 자력계 데이터를 로드
- `ekf_data/ground_truth/`에서 Ground truth 데이터를 로드
- APEqF 필터를 실행하여 상태를 추정
- 결과를 `apeqf_results_YYYYMMDD_HHMMSS.csv` 파일로 저장

### 2단계: 결과 비교 및 시각화

APEqF 결과, EKF 결과, Ground truth를 비교합니다:

```bash
python compare_results.py
```

이 스크립트는:
- APEqF 결과 파일을 자동으로 찾아서 로드
- `ekf_data/ekf_results/`에서 EKF 결과를 로드
- `ekf_data/ground_truth/`에서 Ground truth를 로드
- 궤적 비교, 오차 분석, 추가 메트릭을 시각화
- 다음 파일들을 생성:
  - `trajectory_comparison.png`: 궤적 비교
  - `error_analysis.png`: 오차 분석
  - `additional_metrics.png`: 추가 메트릭

## 출력 파일

### APEqF 결과 파일 (`apeqf_results_*.csv`)

다음 컬럼들을 포함:
- `timestamp`: 타임스탬프
- `pos_x, pos_y, pos_z`: 위치 추정 (m)
- `vel_x, vel_y, vel_z`: 속도 추정 (m/s)
- `att_roll, att_pitch, att_yaw`: 자세 추정 (rad)
- `gyro_bias_x, gyro_bias_y, gyro_bias_z`: 자이로 바이어스 (rad/s)
- `accel_bias_x, accel_bias_y, accel_bias_z`: 가속도계 바이어스 (m/s²)
- `lever_arm_x, lever_arm_y, lever_arm_z`: GNSS 레버암 (m)
- `extrinsic_roll, extrinsic_pitch, extrinsic_yaw`: 외부 회전 (rad)
- `nis_position, nis_velocity, nis_magnetometer`: NIS 값

### 시각화 파일

1. **trajectory_comparison.png**
   - XY, XZ 평면 궤적 비교
   - 시간에 따른 위치 변화

2. **error_analysis.png**
   - 위치 및 속도 오차 분포
   - 시간에 따른 오차 변화
   - 통계 요약 테이블

3. **additional_metrics.png**
   - 자세 추정 결과
   - 바이어스 추정 결과
   - 레버암 추정 결과
   - NIS 값

## 주요 파라미터 조정

### 노이즈 파라미터 (`run_filter.py`의 `APEqFRunner` 클래스)

```python
# 프로세스 노이즈
self.omega_noise = 0.01         # 자이로 노이즈 (rad/s)
self.acc_noise = 0.1            # 가속도계 노이즈 (m/s²)
self.bias_noise = 0.001         # 바이어스 랜덤워크 노이즈
self.lever_noise = 0.001        # 레버암 드리프트 노이즈 (m/s)
self.extrinsic_noise = 0.001    # 외부 회전 드리프트 노이즈 (rad/s)

# 측정 노이즈
self.gps_pos_noise = 1.0        # GPS 위치 노이즈 (m)
self.gps_vel_noise = 0.5        # GPS 속도 노이즈 (m/s)
self.mag_noise = 0.1            # 자력계 노이즈 (Gauss)
```

### 필터 초기화 파라미터

```python
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
```

## 문제 해결

### 일반적인 오류

1. **ImportError: No module named 'pylie'**
   - `pip install pylie` 실행

2. **FileNotFoundError: ekf_data/sensors/...**
   - `ekf_data` 폴더가 올바른 위치에 있는지 확인

3. **MemoryError**
   - 센서 데이터 파일이 너무 클 경우, `run_filter.py`에서 필요한 컬럼만 로드하도록 수정

### 성능 최적화

1. **데이터 샘플링**: 긴 데이터셋의 경우 일부만 사용하여 테스트
2. **노이즈 조정**: 센서 특성에 맞게 노이즈 파라미터 조정
3. **초기값 설정**: 실제 시스템에 맞는 초기 상태 및 불확실성 설정

## 참고 사항

- 센서 데이터는 타임스탬프 기준으로 동기화됩니다
- GPS 데이터가 없는 경우 위치 업데이트는 건너뜁니다
- 자력계 데이터가 없는 경우 자력계 업데이트는 건너뜁니다
- 모든 결과는 CSV 형식으로 저장되어 추가 분석이 가능합니다

## 연락처

문제가 발생하거나 질문이 있으면 개발자에게 문의하세요.
