"""
Feature extraction module for Wind Turbine Bearing RUL Prediction
풍력 터빈 베어링 RUL 예측을 위한 특징 추출 모듈
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any
import warnings

from config import *

class FeatureExtractor:
    """특징 추출 클래스"""
    
    def __init__(self, sampling_frequency: int = SAMPLING_FREQUENCY):
        self.fs = sampling_frequency
        self.spectral_kurtosis_window = SPECTRAL_KURTOSIS_WINDOW
        self.smoothing_window = SMOOTHING_WINDOW
        
    def extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        시간 영역 통계적 특징 추출
        
        Args:
            signal (np.ndarray): 입력 신호
            
        Returns:
            Dict[str, float]: 추출된 특징들
        """
        if len(signal) == 0:
            return {feature: 0.0 for feature in TIME_DOMAIN_FEATURES}
        
        features = {}
        
        # 기본 통계량
        features['Mean'] = np.mean(signal)
        features['Std'] = np.std(signal)
        features['Skewness'] = stats.skew(signal)
        features['Kurtosis'] = stats.kurtosis(signal)
        features['Peak2Peak'] = np.ptp(signal)
        
        # RMS (Root Mean Square)
        features['RMS'] = np.sqrt(np.mean(signal**2))
        
        # 형태 인자들
        mean_abs = np.mean(np.abs(signal))
        max_abs = np.max(np.abs(signal))
        
        if mean_abs > 1e-10:  # 0으로 나누기 방지
            features['CrestFactor'] = max_abs / features['RMS']
            features['ShapeFactor'] = features['RMS'] / mean_abs
            features['ImpulseFactor'] = max_abs / mean_abs
            features['MarginFactor'] = max_abs / (mean_abs ** 2) if mean_abs > 1e-5 else 0
        else:
            features['CrestFactor'] = 0
            features['ShapeFactor'] = 0
            features['ImpulseFactor'] = 0
            features['MarginFactor'] = 0
        
        # 에너지
        features['Energy'] = np.sum(signal**2)
        
        return features
    
    def compute_spectral_kurtosis(self, signal: np.ndarray, 
                                 window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        스펙트럴 커토시스 계산
        
        Args:
            signal (np.ndarray): 입력 신호
            window_size (int, optional): 윈도우 크기
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (스펙트럴 커토시스, 주파수 벡터)
        """
        if window_size is None:
            window_size = self.spectral_kurtosis_window
        
        if len(signal) < window_size:
            # 신호가 윈도우보다 작으면 전체 신호 사용
            window_size = len(signal) // 2
        
        try:
            # Welch 방법으로 파워 스펙트럴 밀도 계산
            f, psd = welch(signal, self.fs, nperseg=window_size, 
                          noverlap=window_size//2, nfft=window_size*2)
            
            # 스펙트럴 커토시스 계산 (간단한 구현)
            # 실제로는 더 복잡한 알고리즘이 사용되지만, 여기서는 PSD 기반으로 근사
            sk = np.zeros_like(f)
            
            # 각 주파수 빈에서 국소적 커토시스 계산
            for i in range(len(f)):
                # 주변 주파수 빈들의 윈도우
                window_half = min(5, len(psd)//10)  # 적응적 윈도우 크기
                start_idx = max(0, i - window_half)
                end_idx = min(len(psd), i + window_half + 1)
                
                local_psd = psd[start_idx:end_idx]
                
                if len(local_psd) > 3:  # 최소 4개 포인트 필요
                    # 정규화된 4차 모멘트 (커토시스)
                    sk[i] = stats.kurtosis(local_psd)
                else:
                    sk[i] = 0
            
            # 무한대나 NaN 값 제거
            sk = np.nan_to_num(sk, nan=0, posinf=0, neginf=0)
            
            return sk, f
            
        except Exception as e:
            warnings.warn(f"스펙트럴 커토시스 계산 실패: {e}")
            # 오류 시 빈 배열 반환
            f = np.linspace(0, self.fs/2, 100)
            sk = np.zeros_like(f)
            return sk, f
    
    def extract_spectral_kurtosis_features(self, sk: np.ndarray) -> Dict[str, float]:
        """
        스펙트럴 커토시스로부터 특징 추출
        
        Args:
            sk (np.ndarray): 스펙트럴 커토시스 배열
            
        Returns:
            Dict[str, float]: 추출된 특징들
        """
        if len(sk) == 0:
            return {feature: 0.0 for feature in SPECTRAL_KURTOSIS_FEATURES}
        
        features = {}
        
        # 유효한 값들만 사용 (무한대나 NaN 제외)
        valid_sk = sk[np.isfinite(sk)]
        
        if len(valid_sk) > 0:
            features['SKMean'] = np.mean(valid_sk)
            features['SKStd'] = np.std(valid_sk)
            features['SKSkewness'] = stats.skew(valid_sk) if len(valid_sk) > 2 else 0
            features['SKKurtosis'] = stats.kurtosis(valid_sk) if len(valid_sk) > 3 else 0
        else:
            features['SKMean'] = 0
            features['SKStd'] = 0
            features['SKSkewness'] = 0
            features['SKKurtosis'] = 0
        
        return features
    
    def extract_frequency_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        주파수 영역 특징 추출 (추가적인 특징들)
        
        Args:
            signal (np.ndarray): 입력 신호
            
        Returns:
            Dict[str, float]: 추출된 특징들
        """
        try:
            # FFT 계산
            fft = np.fft.fft(signal)
            frequencies = np.fft.fftfreq(len(signal), 1/self.fs)
            
            # 양의 주파수만 사용
            positive_freq_idx = frequencies >= 0
            fft_positive = fft[positive_freq_idx]
            freq_positive = frequencies[positive_freq_idx]
            
            # 파워 스펙트럼 계산
            power_spectrum = np.abs(fft_positive)**2
            
            features = {}
            
            # 스펙트럼 중심 주파수
            total_power = np.sum(power_spectrum)
            if total_power > 0:
                features['SpectralCentroid'] = np.sum(freq_positive * power_spectrum) / total_power
            else:
                features['SpectralCentroid'] = 0
            
            # 스펙트럼 확산 (대역폭)
            if total_power > 0 and features['SpectralCentroid'] > 0:
                features['SpectralSpread'] = np.sqrt(
                    np.sum(((freq_positive - features['SpectralCentroid'])**2) * power_spectrum) / total_power
                )
            else:
                features['SpectralSpread'] = 0
            
            # 스펙트럼 평탄도 (Spectral Flatness)
            if len(power_spectrum) > 0 and np.all(power_spectrum > 0):
                geometric_mean = np.exp(np.mean(np.log(power_spectrum)))
                arithmetic_mean = np.mean(power_spectrum)
                features['SpectralFlatness'] = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
            else:
                features['SpectralFlatness'] = 0
            
            # 주파수 대역별 파워 비율
            freq_bands = [
                (0, 500, 'LowFreqPower'),      # 저주파 (0-500 Hz)
                (500, 2000, 'MidFreqPower'),   # 중간주파 (500-2000 Hz)
                (2000, 10000, 'HighFreqPower') # 고주파 (2000-10000 Hz)
            ]
            
            for low_freq, high_freq, feature_name in freq_bands:
                band_idx = (freq_positive >= low_freq) & (freq_positive < high_freq)
                band_power = np.sum(power_spectrum[band_idx])
                features[feature_name] = band_power / total_power if total_power > 0 else 0
            
            return features
            
        except Exception as e:
            warnings.warn(f"주파수 영역 특징 추출 실패: {e}")
            return {
                'SpectralCentroid': 0,
                'SpectralSpread': 0,
                'SpectralFlatness': 0,
                'LowFreqPower': 0,
                'MidFreqPower': 0,
                'HighFreqPower': 0
            }
    
    def extract_all_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        모든 특징 추출
        
        Args:
            signal (np.ndarray): 입력 신호
            
        Returns:
            Dict[str, float]: 모든 추출된 특징들
        """
        all_features = {}
        
        # 시간 영역 특징
        time_features = self.extract_time_domain_features(signal)
        all_features.update(time_features)
        
        # 스펙트럴 커토시스 특징
        sk, f = self.compute_spectral_kurtosis(signal)
        sk_features = self.extract_spectral_kurtosis_features(sk)
        all_features.update(sk_features)
        
        # 추가 주파수 영역 특징
        freq_features = self.extract_frequency_domain_features(signal)
        all_features.update(freq_features)
        
        return all_features
    
    def extract_features_from_dataframe(self, data: pd.DataFrame, 
                                      show_progress: bool = True) -> pd.DataFrame:
        """
        DataFrame의 모든 신호에서 특징 추출
        
        Args:
            data (pd.DataFrame): 진동 신호를 포함한 DataFrame
            show_progress (bool): 진행 상황 표시 여부
            
        Returns:
            pd.DataFrame: 추출된 특징들을 포함한 DataFrame
        """
        features_list = []
        
        for idx, row in data.iterrows():
            vibration = row['vibration']
            
            if vibration is None or len(vibration) == 0:
                # 빈 신호인 경우 0으로 채운 특징 생성
                features = {feature: 0.0 for feature in ALL_FEATURES}
                features.update({
                    'SpectralCentroid': 0,
                    'SpectralSpread': 0,
                    'SpectralFlatness': 0,
                    'LowFreqPower': 0,
                    'MidFreqPower': 0,
                    'HighFreqPower': 0
                })
            else:
                # 특징 추출
                features = self.extract_all_features(vibration)
            
            # 날짜 정보 추가
            features['Date'] = row['Date']
            features_list.append(features)
            
            if show_progress and (idx + 1) % 10 == 0:
                print(f"  특징 추출 진행: {idx + 1}/{len(data)} 완료")
        
        # DataFrame 생성 및 날짜를 인덱스로 설정
        feature_df = pd.DataFrame(features_list)
        feature_df.set_index('Date', inplace=True)
        
        return feature_df
    
    def apply_smoothing(self, feature_df: pd.DataFrame, 
                       window: Optional[int] = None) -> pd.DataFrame:
        """
        특징에 인과적 이동 평균 필터 적용
        
        Args:
            feature_df (pd.DataFrame): 특징 DataFrame
            window (int, optional): 윈도우 크기
            
        Returns:
            pd.DataFrame: 평활화된 특징 DataFrame
        """
        if window is None:
            window = self.smoothing_window
        
        # 인과적 이동 평균 (과거 데이터만 사용)
        smoothed_df = feature_df.rolling(window=window, min_periods=1).mean()
        
        return smoothed_df


class FeatureSelector:
    """특징 선택 클래스"""
    
    def __init__(self):
        self.monotonicity_threshold = MONOTONICITY_THRESHOLD
        self.min_selected_features = MIN_SELECTED_FEATURES
        
    def compute_monotonicity(self, feature_data: pd.DataFrame) -> Dict[str, float]:
        """
        특징들의 단조성 계산
        
        Args:
            feature_data (pd.DataFrame): 특징 데이터
            
        Returns:
            Dict[str, float]: 각 특징의 단조성 점수
        """
        monotonicity_scores = {}
        
        for column in feature_data.columns:
            if column == 'Date':
                continue
                
            values = feature_data[column].values
            n = len(values)
            
            if n <= 1:
                monotonicity_scores[column] = 0
                continue
            
            # 차분 계산
            diffs = np.diff(values)
            
            # 0이 아닌 차분의 개수
            non_zero_diffs = diffs[np.abs(diffs) > 1e-10]
            
            if len(non_zero_diffs) == 0:
                monotonicity_scores[column] = 0
                continue
            
            # 양수와 음수 차분의 개수
            pos_diffs = np.sum(non_zero_diffs > 0)
            neg_diffs = np.sum(non_zero_diffs < 0)
            
            # 단조성 점수: |positive - negative| / total
            monotonicity = abs(pos_diffs - neg_diffs) / len(non_zero_diffs)
            monotonicity_scores[column] = monotonicity
        
        return monotonicity_scores
    
    def select_features_by_monotonicity(self, feature_data: pd.DataFrame, 
                                      threshold: Optional[float] = None,
                                      min_features: Optional[int] = None) -> List[str]:
        """
        단조성에 기반한 특징 선택
        
        Args:
            feature_data (pd.DataFrame): 특징 데이터
            threshold (float, optional): 단조성 임계값
            min_features (int, optional): 최소 선택 특징 수
            
        Returns:
            List[str]: 선택된 특징명 리스트
        """
        if threshold is None:
            threshold = self.monotonicity_threshold
        if min_features is None:
            min_features = self.min_selected_features
        
        # 단조성 계산
        monotonicity_scores = self.compute_monotonicity(feature_data)
        
        # 점수에 따라 정렬
        sorted_features = sorted(monotonicity_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # 임계값 이상의 특징 선택
        selected_features = [feat for feat, score in sorted_features 
                           if score > threshold]
        
        # 최소 개수 보장
        if len(selected_features) < min_features:
            selected_features = [feat for feat, score in sorted_features[:min_features]]
        
        return selected_features
    
    def get_feature_importance_ranking(self, feature_data: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        특징 중요도 순위 반환
        
        Args:
            feature_data (pd.DataFrame): 특징 데이터
            
        Returns:
            List[Tuple[str, float]]: (특징명, 중요도 점수) 튜플 리스트
        """
        monotonicity_scores = self.compute_monotonicity(feature_data)
        return sorted(monotonicity_scores.items(), key=lambda x: x[1], reverse=True)


class DimensionReducer:
    """차원 축소 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.is_fitted = False
        
    def fit_transform(self, train_features: pd.DataFrame) -> np.ndarray:
        """
        훈련 데이터로 PCA 모델 학습 및 변환
        
        Args:
            train_features (pd.DataFrame): 훈련 특징 데이터
            
        Returns:
            np.ndarray: 변환된 주성분들
        """
        # 정규화
        train_normalized = self.scaler.fit_transform(train_features.values)
        
        # PCA 적용
        pca_components = self.pca.fit_transform(train_normalized)
        
        self.is_fitted = True
        return pca_components
    
    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """
        학습된 모델로 특징 변환
        
        Args:
            features (pd.DataFrame): 변환할 특징 데이터
            
        Returns:
            np.ndarray: 변환된 주성분들
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit_transform을 먼저 호출하세요.")
        
        # 정규화 및 PCA 변환
        normalized = self.scaler.transform(features.values)
        pca_components = self.pca.transform(normalized)
        
        return pca_components
    
    def get_health_indicator(self, pca_components: np.ndarray, 
                           component_idx: int = 0) -> np.ndarray:
        """
        건강 지표 생성 (첫 번째 주성분 사용)
        
        Args:
            pca_components (np.ndarray): PCA 주성분들
            component_idx (int): 사용할 주성분 인덱스
            
        Returns:
            np.ndarray: 건강 지표
        """
        health_indicator = pca_components[:, component_idx]
        
        # 0에서 시작하도록 조정
        health_indicator = health_indicator - health_indicator[0]
        
        return health_indicator
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """PCA 설명 분산 비율 반환"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        return self.pca.explained_variance_ratio_


def main():
    """테스트 및 데모 함수"""
    from data_loader import DataLoader
    
    print("특징 추출 모듈 테스트")
    print("=" * 50)
    
    # 데이터 로드
    loader = DataLoader()
    data = loader.load_wind_turbine_data()
    
    # 특징 추출기 생성
    extractor = FeatureExtractor()
    
    # 특징 추출
    print("모든 신호에서 특징 추출 중...")
    features = extractor.extract_features_from_dataframe(data)
    
    print(f"추출된 특징 수: {len(features.columns)}")
    print(f"특징명: {list(features.columns)}")
    
    # 평활화
    print("특징 평활화 적용...")
    smoothed_features = extractor.apply_smoothing(features)
    
    # 특징 선택
    print("특징 중요도 분석...")
    selector = FeatureSelector()
    
    # 훈련 데이터 분할 (처음 60%)
    train_end = int(0.6 * len(smoothed_features))
    train_features = smoothed_features.iloc[:train_end]
    
    # 특징 중요도 순위
    importance_ranking = selector.get_feature_importance_ranking(train_features)
    
    print("상위 10개 특징 중요도:")
    for i, (feature, score) in enumerate(importance_ranking[:10]):
        print(f"  {i+1:2d}. {feature:20s}: {score:.4f}")
    
    # 특징 선택
    selected_features = selector.select_features_by_monotonicity(train_features)
    print(f"\n선택된 특징 수: {len(selected_features)}")
    
    # PCA 차원 축소
    print("PCA 차원 축소...")
    reducer = DimensionReducer()
    
    selected_train_features = train_features[selected_features]
    train_pca = reducer.fit_transform(selected_train_features)
    
    # 전체 데이터 변환
    selected_all_features = smoothed_features[selected_features]
    all_pca = reducer.transform(selected_all_features)
    
    # 건강 지표 생성
    health_indicator = reducer.get_health_indicator(all_pca)
    
    # 설명 분산 출력
    explained_var = reducer.get_explained_variance_ratio()
    print(f"PCA 설명 분산 (상위 3개): {explained_var[:3]}")
    print(f"건강 지표 범위: {health_indicator.min():.4f} ~ {health_indicator.max():.4f}")
    
    print("특징 추출 완료!")


if __name__ == "__main__":
    main()
