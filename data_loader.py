"""
Data loading and preprocessing module for Wind Turbine Bearing RUL Prediction
풍력 터빈 베어링 RUL 예측을 위한 데이터 로딩 및 전처리 모듈
"""

import numpy as np
import pandas as pd
import os
import zipfile
import requests
import glob
import re
from datetime import datetime, timedelta
from scipy.io import loadmat
import warnings
from typing import Optional, List, Tuple, Dict, Any

from config import *

class DataLoader:
    """데이터 로딩 및 관리 클래스"""
    
    def __init__(self):
        self.data_folder = DATA_FOLDER
        self.github_url = GITHUB_REPO_URL
        self.sampling_frequency = SAMPLING_FREQUENCY
        
    def download_wind_turbine_data(self) -> bool:
        """
        GitHub에서 풍력 터빈 베어링 데이터셋 다운로드
        
        Returns:
            bool: 다운로드 성공 여부
        """
        if os.path.exists(self.data_folder):
            print(f"데이터 폴더가 이미 존재합니다: {self.data_folder}")
            return True
            
        print("GitHub에서 풍력 터빈 베어링 데이터셋을 다운로드 중...")
        
        try:
            response = requests.get(self.github_url, timeout=300)
            response.raise_for_status()
            
            with open(DATASET_ZIP_NAME, "wb") as f:
                f.write(response.content)
            
            with zipfile.ZipFile(DATASET_ZIP_NAME, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            os.remove(DATASET_ZIP_NAME)
            print("데이터셋 다운로드 및 압축 해제 완료!")
            return True
            
        except Exception as e:
            print(f"데이터셋 다운로드 실패: {e}")
            print("다음 링크에서 수동으로 다운로드하세요:")
            print("https://github.com/mathworks/WindTurbineHighSpeedBearingPrognosis-Data")
            return False
    
    def extract_timestamp_from_filename(self, filename: str) -> datetime:
        """
        파일명에서 타임스탬프 추출
        
        Args:
            filename (str): 파일명
            
        Returns:
            datetime: 추출된 타임스탬프
        """
        match = re.search(r'(\d{8}T\d{6}Z)', filename)
        if match:
            timestamp_str = match.group(1)
            try:
                return datetime.strptime(timestamp_str, '%Y%m%dT%H%M%SZ')
            except ValueError:
                pass
        
        # 타임스탬프 추출 실패 시 기본값 반환
        return datetime.min
    
    def load_mat_file(self, mat_file: str) -> Dict[str, Any]:
        """
        .mat 파일 로드 및 데이터 추출
        
        Args:
            mat_file (str): .mat 파일 경로
            
        Returns:
            Dict[str, Any]: 추출된 데이터
        """
        try:
            mat_data = loadmat(mat_file)
            
            # 타임스탬프 추출
            timestamp = self.extract_timestamp_from_filename(mat_file)
            
            # 진동 및 타코미터 데이터 추출
            vibration_data = None
            tach_data = None
            
            for key in mat_data.keys():
                if not key.startswith('__'):  # 메타데이터 키 제외
                    data_array = mat_data[key]
                    
                    if data_array.size > 100000:  # 큰 배열은 진동 데이터로 간주
                        vibration_data = data_array.flatten()
                    elif data_array.size < 1000:  # 작은 배열은 타코미터 데이터로 간주
                        tach_data = data_array.flatten()
            
            # 진동 데이터가 없으면 첫 번째 큰 배열 사용
            if vibration_data is None:
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        data_array = mat_data[key]
                        if data_array.size > 1000:
                            vibration_data = data_array.flatten()
                            break
            
            # 타코미터 데이터가 없으면 합성 데이터 생성
            if tach_data is None:
                tach_data = np.random.normal(180, 10, 187)
            
            return {
                'Date': timestamp,
                'vibration': vibration_data,
                'tach': tach_data,
                'filename': os.path.basename(mat_file)
            }
            
        except Exception as e:
            print(f"{mat_file} 로드 실패: {e}")
            return None
    
    def load_wind_turbine_data(self) -> pd.DataFrame:
        """
        실제 풍력 터빈 베어링 데이터 로드
        
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        # 데이터 다운로드 시도
        if not self.download_wind_turbine_data():
            print("실제 데이터 로드 실패, 시뮬레이션 데이터 사용")
            return self.load_simulated_data()
        
        # .mat 파일 검색
        mat_files = glob.glob(os.path.join(self.data_folder, "*.mat"))
        
        if not mat_files:
            print(".mat 파일을 찾을 수 없습니다. 시뮬레이션 데이터 사용")
            return self.load_simulated_data()
        
        print(f"{len(mat_files)}개의 데이터 파일 발견")
        
        # 파일을 타임스탬프 순으로 정렬
        mat_files.sort(key=self.extract_timestamp_from_filename)
        
        # 데이터 로드
        data_list = []
        successful_loads = 0
        
        for mat_file in mat_files:
            data_dict = self.load_mat_file(mat_file)
            if data_dict and data_dict['vibration'] is not None:
                data_list.append(data_dict)
                successful_loads += 1
        
        if not data_list:
            print("유효한 데이터 파일이 없습니다. 시뮬레이션 데이터 사용")
            return self.load_simulated_data()
        
        print(f"{successful_loads}개 파일 성공적으로 로드")
        
        # DataFrame 생성
        df = pd.DataFrame(data_list)
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def load_simulated_data(self) -> pd.DataFrame:
        """
        시뮬레이션 풍력 터빈 베어링 데이터 생성
        
        Returns:
            pd.DataFrame: 시뮬레이션된 데이터
        """
        print("시뮬레이션 풍력 터빈 베어링 데이터 생성 중...")
        
        np.random.seed(42)
        dates = pd.date_range('2013-03-11', periods=50, freq='D')
        
        vibration_data = []
        tach_data = []
        
        for i in range(50):
            # 시간에 따른 점진적 열화 시뮬레이션
            degradation_factor = 1 + i * 0.15
            
            # 6초, 97656 Hz 샘플링
            t = np.linspace(0, 6, EXPECTED_SIGNAL_LENGTH)
            
            # 기본 신호 (증가하는 진폭)
            base_signal = np.random.normal(0, 1.5 * degradation_factor, len(t))
            
            # 베어링 고장 주파수 성분 추가
            fault_freq = 180  # Hz (베어링 고장 주파수)
            fault_signal = 0.2 * degradation_factor * np.sin(2 * np.pi * fault_freq * t)
            
            # 고조파 성분
            harmonic2 = 0.1 * degradation_factor * np.sin(2 * np.pi * 2 * fault_freq * t)
            harmonic3 = 0.05 * degradation_factor * np.sin(2 * np.pi * 3 * fault_freq * t)
            
            # 충격 성분 (시간에 따라 증가)
            num_impulses = int(10 * degradation_factor)
            impulse_locations = np.random.choice(len(t), num_impulses, replace=False)
            impulse_signal = np.zeros_like(t)
            
            for loc in impulse_locations:
                # 지수 감쇠 충격
                impulse_amplitude = 5 * degradation_factor
                decay_rate = 1000
                impulse_width = int(0.01 * self.sampling_frequency)  # 10ms
                
                start_idx = max(0, loc - impulse_width // 2)
                end_idx = min(len(t), loc + impulse_width // 2)
                
                impulse_time = np.arange(len(t[start_idx:end_idx])) / self.sampling_frequency
                impulse_signal[start_idx:end_idx] += (
                    impulse_amplitude * np.exp(-decay_rate * impulse_time)
                )
            
            # 모든 성분 결합
            vibration = base_signal + fault_signal + harmonic2 + harmonic3 + impulse_signal
            
            # 고주파 노이즈 추가
            high_freq_noise = 0.1 * np.random.normal(0, 1, len(t))
            vibration += high_freq_noise
            
            # 타코미터 데이터 (RPM 변동)
            tachometer = np.random.normal(180, 5, 187)
            
            vibration_data.append(vibration)
            tach_data.append(tachometer)
        
        return pd.DataFrame({
            'Date': dates,
            'vibration': vibration_data,
            'tach': tach_data
        })
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        데이터 유효성 검사
        
        Args:
            data (pd.DataFrame): 검사할 데이터
            
        Returns:
            Tuple[bool, List[str]]: (유효성 여부, 오류 메시지 리스트)
        """
        errors = []
        
        # 기본 컬럼 확인
        required_columns = ['Date', 'vibration', 'tach']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"필수 컬럼 누락: {missing_columns}")
        
        # 데이터 개수 확인
        if len(data) == 0:
            errors.append("데이터가 비어있습니다")
            return False, errors
        
        # 진동 데이터 확인
        for idx, vibration in enumerate(data['vibration']):
            if vibration is None or len(vibration) == 0:
                errors.append(f"인덱스 {idx}에서 빈 진동 데이터")
                continue
            
            # 신호 길이 확인 (허용 범위: ±10%)
            expected_length = EXPECTED_SIGNAL_LENGTH
            min_length = int(expected_length * 0.9)
            max_length = int(expected_length * 1.1)
            
            if not (min_length <= len(vibration) <= max_length):
                errors.append(
                    f"인덱스 {idx}에서 예상치 않은 신호 길이: {len(vibration)} "
                    f"(예상: {expected_length}±10%)"
                )
            
            # 신호 값 범위 확인 (합리적 범위)
            if np.max(np.abs(vibration)) > 100:  # 100g 이상은 비현실적
                errors.append(f"인덱스 {idx}에서 과도한 진동 값: {np.max(np.abs(vibration)):.2f}g")
            
            # NaN 또는 무한대 값 확인
            if np.any(np.isnan(vibration)) or np.any(np.isinf(vibration)):
                errors.append(f"인덱스 {idx}에서 유효하지 않은 값 (NaN 또는 Inf)")
        
        # 시간 순서 확인
        if not data['Date'].is_monotonic_increasing:
            errors.append("날짜가 시간 순서대로 정렬되지 않았습니다")
        
        return len(errors) == 0, errors
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        데이터 요약 정보 생성
        
        Args:
            data (pd.DataFrame): 요약할 데이터
            
        Returns:
            Dict[str, Any]: 요약 정보
        """
        if len(data) == 0:
            return {'error': '데이터가 비어있습니다'}
        
        # 기본 정보
        summary = {
            'total_samples': len(data),
            'date_range': {
                'start': data['Date'].min(),
                'end': data['Date'].max(),
                'duration_days': (data['Date'].max() - data['Date'].min()).days
            },
            'sampling_info': {
                'frequency': self.sampling_frequency,
                'expected_signal_length': EXPECTED_SIGNAL_LENGTH,
                'signal_duration': EXPECTED_SIGNAL_LENGTH / self.sampling_frequency
            }
        }
        
        # 진동 신호 통계
        vibration_stats = []
        for vibration in data['vibration']:
            if vibration is not None and len(vibration) > 0:
                stats = {
                    'length': len(vibration),
                    'mean': np.mean(vibration),
                    'std': np.std(vibration),
                    'min': np.min(vibration),
                    'max': np.max(vibration),
                    'rms': np.sqrt(np.mean(vibration**2))
                }
                vibration_stats.append(stats)
        
        if vibration_stats:
            summary['vibration_statistics'] = {
                'mean_length': np.mean([s['length'] for s in vibration_stats]),
                'mean_rms': np.mean([s['rms'] for s in vibration_stats]),
                'std_rms': np.std([s['rms'] for s in vibration_stats]),
                'max_amplitude': np.max([s['max'] for s in vibration_stats]),
                'min_amplitude': np.min([s['min'] for s in vibration_stats])
            }
        
        return summary
    
    def save_data_summary(self, data: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        데이터 요약을 파일로 저장
        
        Args:
            data (pd.DataFrame): 요약할 데이터
            filename (str, optional): 저장 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        if filename is None:
            filename = f"data_summary_{get_timestamp()}.txt"
        
        filepath = get_results_path(filename)
        
        summary = self.get_data_summary(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Wind Turbine Bearing Data Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # 기본 정보
            f.write("Basic Information:\n")
            f.write(f"  Total Samples: {summary['total_samples']}\n")
            f.write(f"  Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}\n")
            f.write(f"  Duration: {summary['date_range']['duration_days']} days\n\n")
            
            # 샘플링 정보
            f.write("Sampling Information:\n")
            f.write(f"  Frequency: {summary['sampling_info']['frequency']} Hz\n")
            f.write(f"  Signal Length: {summary['sampling_info']['expected_signal_length']} samples\n")
            f.write(f"  Signal Duration: {summary['sampling_info']['signal_duration']:.1f} seconds\n\n")
            
            # 진동 통계
            if 'vibration_statistics' in summary:
                stats = summary['vibration_statistics']
                f.write("Vibration Statistics:\n")
                f.write(f"  Average Signal Length: {stats['mean_length']:.0f} samples\n")
                f.write(f"  Average RMS: {stats['mean_rms']:.4f} g\n")
                f.write(f"  RMS Std Dev: {stats['std_rms']:.4f} g\n")
                f.write(f"  Max Amplitude: {stats['max_amplitude']:.4f} g\n")
                f.write(f"  Min Amplitude: {stats['min_amplitude']:.4f} g\n")
        
        print(f"데이터 요약이 저장되었습니다: {filepath}")
        return filepath


def main():
    """테스트 및 데모 함수"""
    loader = DataLoader()
    
    print("풍력 터빈 베어링 데이터 로딩 테스트")
    print("=" * 50)
    
    # 데이터 로드
    data = loader.load_wind_turbine_data()
    
    # 데이터 검증
    is_valid, errors = loader.validate_data(data)
    
    if is_valid:
        print("✓ 데이터 검증 통과")
    else:
        print("✗ 데이터 검증 실패:")
        for error in errors:
            print(f"  - {error}")
    
    # 요약 정보 출력
    summary = loader.get_data_summary(data)
    print(f"\n데이터 요약:")
    print(f"  총 샘플 수: {summary['total_samples']}")
    print(f"  기간: {summary['date_range']['duration_days']}일")
    
    if 'vibration_statistics' in summary:
        stats = summary['vibration_statistics']
        print(f"  평균 RMS: {stats['mean_rms']:.4f}g")
        print(f"  최대 진폭: {stats['max_amplitude']:.4f}g")
    
    # 요약 파일 저장
    loader.save_data_summary(data)
    
    return data


if __name__ == "__main__":
    main()