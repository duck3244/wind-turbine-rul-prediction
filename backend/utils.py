"""
Utility functions for Wind Turbine Bearing RUL Prediction
풍력 터빈 베어링 RUL 예측을 위한 유틸리티 함수들
"""

import numpy as np
import pandas as pd
import os
import random
import json
import pickle
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from config import *


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """재현성 확보를 위해 random/numpy/tensorflow 시드를 통합 설정."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

class ProgressTracker:
    """진행 상황 추적 클래스"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_times = []
        
    def update(self, step: Optional[int] = None, message: str = ""):
        """진행 상황 업데이트"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        current_time = time.time()
        self.step_times.append(current_time)
        
        # 진행률 계산
        progress = self.current_step / self.total_steps
        elapsed_time = current_time - self.start_time
        
        # 예상 완료 시간 계산
        if self.current_step > 0:
            avg_step_time = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * avg_step_time
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        # 진행률 바 생성
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # 출력
        print(f"\r{self.description}: |{bar}| {progress:.1%} ({self.current_step}/{self.total_steps}) - {eta_str} {message}", 
              end='', flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # 완료 시 새 줄
    
    def finish(self, message: str = "Complete!"):
        """완료 처리"""
        total_time = time.time() - self.start_time
        print(f"\n{self.description} {message} (Total time: {total_time:.1f}s)")


class DataValidator:
    """데이터 검증 유틸리티"""
    
    @staticmethod
    def validate_signal(signal: np.ndarray, 
                       expected_length: Optional[int] = None,
                       max_amplitude: float = 100.0) -> Tuple[bool, List[str]]:
        """
        진동 신호 검증
        
        Args:
            signal (np.ndarray): 검증할 신호
            expected_length (int, optional): 예상 신호 길이
            max_amplitude (float): 최대 허용 진폭
            
        Returns:
            Tuple[bool, List[str]]: (유효성, 오류 메시지 리스트)
        """
        errors = []
        
        if signal is None:
            errors.append("신호가 None입니다")
            return False, errors
        
        if len(signal) == 0:
            errors.append("빈 신호입니다")
            return False, errors
        
        # 길이 확인
        if expected_length is not None:
            tolerance = int(expected_length * 0.1)  # 10% 허용 오차
            if abs(len(signal) - expected_length) > tolerance:
                errors.append(f"예상치 않은 신호 길이: {len(signal)} (예상: {expected_length}±{tolerance})")
        
        # 진폭 확인
        max_val = np.max(np.abs(signal))
        if max_val > max_amplitude:
            errors.append(f"과도한 진폭: {max_val:.2f} (최대 허용: {max_amplitude})")
        
        # NaN, Inf 확인
        if np.any(np.isnan(signal)):
            errors.append("NaN 값이 포함되어 있습니다")
        
        if np.any(np.isinf(signal)):
            errors.append("무한대 값이 포함되어 있습니다")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_features(features: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        특징 DataFrame 검증
        
        Args:
            features (pd.DataFrame): 검증할 특징 데이터
            
        Returns:
            Tuple[bool, List[str]]: (유효성, 오류 메시지 리스트)
        """
        errors = []
        
        if features is None or features.empty:
            errors.append("특징 데이터가 비어있습니다")
            return False, errors
        
        # NaN 값 확인
        nan_cols = features.columns[features.isnull().any()].tolist()
        if nan_cols:
            errors.append(f"NaN 값이 포함된 컬럼: {nan_cols}")
        
        # 무한대 값 확인
        for col in features.select_dtypes(include=[np.number]).columns:
            if np.any(np.isinf(features[col])):
                errors.append(f"무한대 값이 포함된 컬럼: {col}")
        
        # 상수 컬럼 확인
        constant_cols = []
        for col in features.select_dtypes(include=[np.number]).columns:
            if features[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            errors.append(f"상수 컬럼 (분산=0): {constant_cols}")
        
        return len(errors) == 0, errors


class PerformanceAnalyzer:
    """성능 분석 유틸리티"""
    
    @staticmethod
    def calculate_metrics(true_values: np.ndarray, 
                         predicted_values: np.ndarray) -> Dict[str, float]:
        """
        성능 메트릭 계산
        
        Args:
            true_values (np.ndarray): 실제 값들
            predicted_values (np.ndarray): 예측 값들
            
        Returns:
            Dict[str, float]: 성능 메트릭 딕셔너리
        """
        # 길이 맞추기
        min_len = min(len(true_values), len(predicted_values))
        true_aligned = true_values[:min_len]
        pred_aligned = predicted_values[:min_len]
        
        # 유효한 값들만 사용
        valid_mask = np.isfinite(true_aligned) & np.isfinite(pred_aligned)
        true_valid = true_aligned[valid_mask]
        pred_valid = pred_aligned[valid_mask]
        
        if len(true_valid) == 0:
            return {
                'mae': np.inf, 'rmse': np.inf, 'mape': np.inf,
                'r2': -np.inf, 'max_error': np.inf, 'n_valid': 0
            }
        
        # 메트릭 계산
        mae = mean_absolute_error(true_valid, pred_valid)
        rmse = np.sqrt(mean_squared_error(true_valid, pred_valid))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((true_valid - pred_valid) / (true_valid + 1e-10))) * 100
        
        # R² Score
        ss_res = np.sum((true_valid - pred_valid) ** 2)
        ss_tot = np.sum((true_valid - np.mean(true_valid)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        # 최대 오차
        max_error = np.max(np.abs(true_valid - pred_valid))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'max_error': max_error,
            'n_valid': len(true_valid)
        }
    
    @staticmethod
    def calculate_prognostic_horizon(true_rul: np.ndarray, 
                                   predicted_rul: np.ndarray,
                                   alpha: float = 0.2) -> float:
        """
        예측 정확도 계산 (α-bound 내 예측 비율)
        
        Args:
            true_rul (np.ndarray): 실제 RUL
            predicted_rul (np.ndarray): 예측 RUL
            alpha (float): 허용 오차 비율
            
        Returns:
            float: α-bound 내 예측 비율
        """
        min_len = min(len(true_rul), len(predicted_rul))
        true_aligned = true_rul[:min_len]
        pred_aligned = predicted_rul[:min_len]
        
        # 유효한 값들만 사용
        valid_mask = (np.isfinite(true_aligned) & np.isfinite(pred_aligned) & 
                     (true_aligned > 0))
        
        if np.sum(valid_mask) == 0:
            return 0.0
        
        true_valid = true_aligned[valid_mask]
        pred_valid = pred_aligned[valid_mask]
        
        # α-bound 계산
        lower_bound = true_valid * (1 - alpha)
        upper_bound = true_valid * (1 + alpha)
        
        within_bound = (pred_valid >= lower_bound) & (pred_valid <= upper_bound)
        
        return np.mean(within_bound)
    
    @staticmethod
    def analyze_prediction_trends(predictions: Dict[str, np.ndarray],
                                true_values: np.ndarray) -> Dict[str, Any]:
        """
        예측 트렌드 분석
        
        Args:
            predictions (Dict): 모델별 예측 결과
            true_values (np.ndarray): 실제 값들
            
        Returns:
            Dict[str, Any]: 트렌드 분석 결과
        """
        analysis_results = {}
        
        for model_name, pred_values in predictions.items():
            if len(pred_values) == 0:
                continue
            
            # 길이 맞추기
            min_len = min(len(pred_values), len(true_values))
            pred_aligned = pred_values[:min_len]
            true_