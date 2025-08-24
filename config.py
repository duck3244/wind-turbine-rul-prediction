"""
Configuration file for Wind Turbine Bearing RUL Prediction
풍력 터빈 베어링 RUL 예측을 위한 설정 파일
"""

import os
from datetime import datetime

# =============================================================================
# 데이터 관련 설정
# =============================================================================

# GitHub 데이터셋 정보
GITHUB_REPO_URL = "https://github.com/mathworks/WindTurbineHighSpeedBearingPrognosis-Data/archive/refs/heads/main.zip"
DATA_FOLDER = "WindTurbineHighSpeedBearingPrognosis-Data-main"
DATASET_ZIP_NAME = "dataset.zip"

# 신호 처리 파라미터
SAMPLING_FREQUENCY = 97656  # Hz
SIGNAL_DURATION = 6  # seconds
EXPECTED_SIGNAL_LENGTH = 292968  # samples

# 데이터 분할 비율
TRAIN_SPLIT = 0.6
VALIDATION_SPLIT = 0.8  # 0.6-0.8 구간이 검증 세트
# 0.8-1.0 구간이 테스트 세트

# =============================================================================
# 특징 추출 설정
# =============================================================================

# 스펙트럴 커토시스 파라미터
SPECTRAL_KURTOSIS_WINDOW = 128

# 특징 평활화 파라미터
SMOOTHING_WINDOW = 5

# 단조성 임계값
MONOTONICITY_THRESHOLD = 0.3
MIN_SELECTED_FEATURES = 8

# =============================================================================
# 모델 관련 설정
# =============================================================================

# 지수적 열화 모델 파라미터
EXP_MODEL_CONFIG = {
    'theta': 1,
    'theta_variance': 1e6,
    'beta': 1,
    'beta_variance': 1e6,
    'phi': -1,
    'noise_variance_factor': 0.1,
    'slope_detection_level': 0.05
}

# LSTM 모델 파라미터
LSTM_CONFIG = {
    'sequence_length': 10,
    'lstm_units': 64,
    'dropout_rate': 0.3,
    'epochs': 150,
    'batch_size': 16,
    'learning_rate': 0.001,
    'patience_early_stopping': 20,
    'patience_lr_reduction': 10,
    'lr_reduction_factor': 0.5,
    'min_learning_rate': 1e-6
}

# 앙상블 설정
ENSEMBLE_CONFIG = {
    'n_models': 5,
    'model_configs': [
        {'sequence_length': 8, 'lstm_units': 64, 'dropout_rate': 0.2},
        {'sequence_length': 10, 'lstm_units': 64, 'dropout_rate': 0.3},
        {'sequence_length': 12, 'lstm_units': 32, 'dropout_rate': 0.3},
        {'sequence_length': 10, 'lstm_units': 128, 'dropout_rate': 0.4},
        {'sequence_length': 15, 'lstm_units': 64, 'dropout_rate': 0.25}
    ]
}

# =============================================================================
# 성능 평가 설정
# =============================================================================

# α-λ 분석 파라미터
ALPHA_BOUND = 0.2  # 20% 허용 오차

# 알람 임계값 (일)
ALERT_THRESHOLDS = {
    'critical': 2,    # 2일 이하
    'warning': 5,     # 5일 이하
    'normal': float('inf')
}

# 성능 벤치마크 설정
BENCHMARK_CONFIG = {
    'n_runs': 5,
    'test_split': 0.3
}

# =============================================================================
# 시각화 설정
# =============================================================================

PLOT_CONFIG = {
    'figure_size': (20, 16),
    'dpi': 300,
    'colors': {
        'true_rul': 'black',
        'exponential': 'blue',
        'lstm': 'red',
        'ensemble': 'green',
        'threshold': 'red',
        'train_split': 'gray',
        'val_split': 'orange'
    },
    'line_styles': {
        'true_rul': '-',
        'exponential': '--',
        'lstm': '--',
        'ensemble': '-.'
    },
    'line_widths': {
        'true_rul': 3,
        'exponential': 2,
        'lstm': 2,
        'ensemble': 2,
        'threshold': 2
    }
}

# =============================================================================
# 파일 경로 설정
# =============================================================================

# 프로젝트 루트 디렉토리
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 결과 저장 디렉토리
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')

# 디렉토리 생성
for directory in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# 로깅 설정
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': os.path.join(RESULTS_DIR, f'rul_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
}

# =============================================================================
# TensorFlow 설정
# =============================================================================

# GPU 메모리 증가 허용
TENSORFLOW_CONFIG = {
    'allow_memory_growth': True,
    'log_level': '2'  # INFO 및 WARNING 로그 숨기기
}

# =============================================================================
# 실시간 모니터링 설정
# =============================================================================

MONITORING_CONFIG = {
    'history_window': 20,      # 최근 20개 데이터 포인트 사용
    'trend_analysis_days': 10, # 10일 트렌드 분석
    'alert_check_interval': 1, # 1회 측정마다 알람 체크
    'data_retention_days': 365 # 1년간 데이터 보관
}

# =============================================================================
# 하이퍼파라미터 최적화 설정
# =============================================================================

HYPERPARAMETER_GRID = {
    'sequence_length': [8, 10, 12, 15],
    'lstm_units': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    'learning_rate': [0.001, 0.005, 0.01],
    'batch_size': [8, 16, 32]
}

OPTIMIZATION_CONFIG = {
    'cv_folds': 3,
    'max_trials': 50,
    'timeout_hours': 24
}

# =============================================================================
# 특징 이름 정의
# =============================================================================

TIME_DOMAIN_FEATURES = [
    'Mean', 'Std', 'Skewness', 'Kurtosis', 'Peak2Peak',
    'RMS', 'CrestFactor', 'ShapeFactor', 'ImpulseFactor', 
    'MarginFactor', 'Energy'
]

SPECTRAL_KURTOSIS_FEATURES = [
    'SKMean', 'SKStd', 'SKSkewness', 'SKKurtosis'
]

ALL_FEATURES = TIME_DOMAIN_FEATURES + SPECTRAL_KURTOSIS_FEATURES

# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_timestamp():
    """현재 타임스탬프 반환"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_results_path(filename):
    """결과 파일 경로 생성"""
    return os.path.join(RESULTS_DIR, filename)

def get_model_path(model_name):
    """모델 저장 경로 생성"""
    return os.path.join(MODELS_DIR, f"{model_name}_{get_timestamp()}.h5")

def get_plot_path(plot_name):
    """플롯 저장 경로 생성"""
    return os.path.join(PLOTS_DIR, f"{plot_name}_{get_timestamp()}.png")

def print_config_summary():
    """설정 요약 출력"""
    print("Wind Turbine RUL Prediction - Configuration Summary")
    print("="*60)
    print(f"Sampling Frequency: {SAMPLING_FREQUENCY} Hz")
    print(f"Train/Val/Test Split: {TRAIN_SPLIT:.1%}/{VALIDATION_SPLIT-TRAIN_SPLIT:.1%}/{1-VALIDATION_SPLIT:.1%}")
    print(f"LSTM Sequence Length: {LSTM_CONFIG['sequence_length']}")
    print(f"LSTM Units: {LSTM_CONFIG['lstm_units']}")
    print(f"Training Epochs: {LSTM_CONFIG['epochs']}")
    print(f"Batch Size: {LSTM_CONFIG['batch_size']}")
    print(f"Alert Thresholds: Critical={ALERT_THRESHOLDS['critical']}d, Warning={ALERT_THRESHOLDS['warning']}d")
    print(f"Results Directory: {RESULTS_DIR}")
    print("="*60)

if __name__ == "__main__":
    print_config_summary()
