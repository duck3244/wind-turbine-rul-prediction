#!/usr/bin/env python3
"""
Wind Turbine Bearing RUL Prediction - Main Execution File
풍력 터빈 베어링 잔여 수명 예측 - 메인 실행 파일 (새로 작성)

간결하고 안정적인 파이프라인 구현
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import warnings
from datetime import datetime

# 경고 메시지 필터링
warnings.filterwarnings('ignore')


def check_imports():
    """필요한 모듈들을 안전하게 import"""
    try:
        import config
        from data_loader import DataLoader
        from feature_extractor import FeatureExtractor, FeatureSelector, DimensionReducer
        from models import ExponentialDegradationModel

        # TensorFlow 및 LSTM 확인
        try:
            from models import LSTMRULPredictor, TENSORFLOW_AVAILABLE
        except ImportError:
            LSTMRULPredictor = None
            TENSORFLOW_AVAILABLE = False

        # 시각화 모듈 확인
        try:
            from visualization import RULVisualizer
        except ImportError:
            RULVisualizer = None

        return True, {
            'config': config,
            'data_loader': DataLoader,
            'feature_extractor': (FeatureExtractor, FeatureSelector, DimensionReducer),
            'exp_model': ExponentialDegradationModel,
            'lstm_model': LSTMRULPredictor,
            'visualizer': RULVisualizer,
            'tensorflow_available': TENSORFLOW_AVAILABLE
        }

    except ImportError as e:
        print(f"❌ 모듈 import 실패: {e}")
        return False, None


class SimpleRULPipeline:
    """간단하고 안정적인 RUL 예측 파이프라인"""

    def __init__(self, components: dict, use_lstm=True, use_visualization=True,
                 lstm_sequence_length: int = 8):
        self.components = components
        self.use_lstm = use_lstm
        self.use_visualization = use_visualization
        self.lstm_sequence_length = lstm_sequence_length

        # 결과 저장용
        self.data = None
        self.features = None
        self.health_indicator = None
        self.true_rul = None
        self.exp_predictions = None
        self.lstm_predictions = None
        self.results = {}

        print("🚀 Simple RUL Prediction Pipeline 초기화")

    def step1_load_data(self):
        """1단계: 데이터 로딩"""
        print("\n📊 1단계: 데이터 로딩...")

        try:
            loader = self.components['data_loader']()
            self.data = loader.load_wind_turbine_data()

            print(f"✅ 데이터 로딩 완료: {len(self.data)}개 샘플")
            print(f"   기간: {self.data['Date'].min()} ~ {self.data['Date'].max()}")

            return True

        except Exception as e:
            print(f"❌ 데이터 로딩 실패: {e}")
            return False

    def step2_extract_features(self):
        """2단계: 특징 추출 및 건강 지표 생성"""
        print("\n🔍 2단계: 특징 추출...")

        try:
            FeatureExtractor, FeatureSelector, DimensionReducer = self.components['feature_extractor']
            from config import TRAIN_SPLIT

            # 특징 추출
            extractor = FeatureExtractor()
            features_raw = extractor.extract_features_from_dataframe(self.data)

            # 평활화
            features_smooth = extractor.apply_smoothing(features_raw)

            # 특징 선택 (훈련 구간만 사용)
            train_size = int(TRAIN_SPLIT * len(features_smooth))
            selector = FeatureSelector()
            selected_features = selector.select_features_by_monotonicity(
                features_smooth.iloc[:train_size]
            )

            # PCA로 건강 지표 생성
            reducer = DimensionReducer()
            features_selected = features_smooth[selected_features]

            # 훈련 데이터로 PCA 학습
            train_pca = reducer.fit_transform(features_selected.iloc[:train_size])

            # 전체 데이터 변환
            all_pca = reducer.transform(features_selected)

            # 건강 지표 생성 (첫 번째 주성분)
            self.health_indicator = reducer.get_health_indicator(all_pca)
            self.features = np.column_stack([
                features_selected.values,
                self.health_indicator.reshape(-1, 1)
            ])

            print(f"✅ 특징 추출 완료:")
            print(f"   - 원본 특징: {len(features_raw.columns)}개")
            print(f"   - 선택된 특징: {len(selected_features)}개")
            print(f"   - 건강 지표 범위: {self.health_indicator.min():.3f} ~ {self.health_indicator.max():.3f}")

            return True

        except Exception as e:
            print(f"❌ 특징 추출 실패: {e}")
            return False

    def step3_prepare_targets(self):
        """3단계: RUL 타겟 준비"""
        print("\n🎯 3단계: RUL 타겟 준비...")

        try:
            n_samples = len(self.health_indicator)
            self.true_rul = np.array([n_samples - i - 1 for i in range(n_samples)])

            print(f"✅ RUL 타겟 준비 완료: {n_samples}개 샘플")

            return True

        except Exception as e:
            print(f"❌ RUL 타겟 준비 실패: {e}")
            return False

    def step4_train_exponential_model(self):
        """4단계: 지수적 모델 훈련"""
        print("\n📈 4단계: 지수적 모델 훈련...")

        try:
            from config import RUL_PREDICTION_UPPER_BOUND, RUL_PREDICTION_FALLBACK

            model = self.components['exp_model']()

            # 모델 훈련
            model.train(self.features, self.true_rul, self.health_indicator)

            # 예측 수행
            self.exp_predictions = model.predict(
                self.features[:-1], self.health_indicator[:-1]
            )

            # 성능 계산
            true_aligned = self.true_rul[:-1]

            # 무한대/과도한 값 후처리
            exp_pred_clean = np.where(
                np.isinf(self.exp_predictions) | (self.exp_predictions > RUL_PREDICTION_UPPER_BOUND),
                RUL_PREDICTION_FALLBACK,
                np.maximum(self.exp_predictions, 0),
            )

            mae = np.mean(np.abs(true_aligned - exp_pred_clean))
            rmse = np.sqrt(np.mean((true_aligned - exp_pred_clean) ** 2))

            self.results['Exponential'] = {
                'predictions': exp_pred_clean,
                'mae': mae,
                'rmse': rmse
            }

            print(f"✅ 지수적 모델 훈련 완료:")
            print(f"   - MAE: {mae:.3f} days")
            print(f"   - RMSE: {rmse:.3f} days")

            return True

        except Exception as e:
            print(f"❌ 지수적 모델 훈련 실패: {e}")
            return False

    def step5_train_lstm_model(self):
        """5단계: LSTM 모델 훈련 (선택사항)"""
        if not self.use_lstm or not self.components['tensorflow_available']:
            print("\n⚠️  LSTM 모델 건너뛰기 (TensorFlow 없음 또는 비활성화)")
            return True

        print("\n🧠 5단계: LSTM 모델 훈련...")

        try:
            LSTMRULPredictor = self.components['lstm_model']
            seq_len = self.lstm_sequence_length
            model = LSTMRULPredictor(
                sequence_length=seq_len,
                lstm_units=32,
                epochs=50,
                dropout_rate=0.3,
            )

            print("   신경망 훈련 중 (잠시 기다려주세요)...")
            model.train(self.features, self.true_rul, verbose=0)

            lstm_pred = model.predict(self.features)

            if len(lstm_pred) == 0:
                print("⚠️  LSTM 예측 결과가 비어있음")
                return True

            # create_sequences 규약: lstm_pred[k] 는 true_rul[k + seq_len] 예측
            true_aligned = self.true_rul[seq_len:seq_len + len(lstm_pred)]
            lstm_pred_aligned = lstm_pred[:len(true_aligned)]

            mae = np.mean(np.abs(true_aligned - lstm_pred_aligned))
            rmse = np.sqrt(np.mean((true_aligned - lstm_pred_aligned) ** 2))

            self.results['LSTM'] = {
                'predictions': lstm_pred_aligned,
                'targets': true_aligned,
                'sequence_offset': seq_len,
                'mae': mae,
                'rmse': rmse,
            }

            print(f"✅ LSTM 모델 훈련 완료:")
            print(f"   - MAE: {mae:.3f} days")
            print(f"   - RMSE: {rmse:.3f} days")

            return True

        except Exception as e:
            print(f"❌ LSTM 모델 훈련 실패: {e}")
            return False

    def step6_generate_visualizations(self):
        """6단계: 시각화 생성 (선택사항)"""
        if not self.use_visualization or not self.components['visualizer']:
            print("\n⚠️  시각화 건너뛰기")
            return True

        print("\n📊 6단계: 시각화 생성...")

        try:
            visualizer = self.components['visualizer']()

            # 건강 지표 시각화
            hi_fig = visualizer.plot_health_indicator(
                self.health_indicator,
                threshold=self.health_indicator[-1]
            )

            # 예측 결과 비교 시각화
            predictions_dict = {}
            if 'Exponential' in self.results:
                predictions_dict['Exponential'] = self.results['Exponential']['predictions']
            if 'LSTM' in self.results:
                predictions_dict['LSTM'] = self.results['LSTM']['predictions']

            if predictions_dict:
                pred_fig = visualizer.plot_rul_predictions(
                    self.true_rul[:-1], predictions_dict
                )

            # 성능 메트릭 시각화
            if len(self.results) > 0:
                metrics_fig = visualizer.plot_performance_metrics(self.results)

            # 시각화 저장
            figures = {
                'health_indicator': hi_fig,
            }

            if predictions_dict:
                figures['rul_predictions'] = pred_fig

            if len(self.results) > 0:
                figures['performance_metrics'] = metrics_fig

            saved_paths = visualizer.save_plots(figures)

            print(f"✅ 시각화 완료: {len(saved_paths)}개 파일 저장")

            return True

        except Exception as e:
            print(f"❌ 시각화 생성 실패: {e}")
            return False

    def step7_print_summary(self):
        """7단계: 결과 요약"""
        print("\n📋 7단계: 결과 요약")
        print("=" * 60)

        # 데이터 정보
        if self.data is not None:
            print(f"📊 데이터: {len(self.data)}개 샘플")
            print(f"   기간: {(self.data['Date'].max() - self.data['Date'].min()).days}일")

        # 건강 지표 정보
        if self.health_indicator is not None:
            print(f"🔍 건강 지표: {self.health_indicator.min():.3f} ~ {self.health_indicator.max():.3f}")

        # 모델 성능 비교
        print(f"🎯 모델 성능:")
        for model_name, result in self.results.items():
            print(f"   {model_name}:")
            print(f"     MAE: {result['mae']:.3f} days")
            print(f"     RMSE: {result['rmse']:.3f} days")

        # 최고 성능 모델
        if len(self.results) > 1:
            best_model = min(self.results.keys(), key=lambda k: self.results[k]['mae'])
            best_mae = self.results[best_model]['mae']
            print(f"\n🏆 최고 성능: {best_model} (MAE: {best_mae:.3f} days)")

        return True

    def run_pipeline(self):
        """전체 파이프라인 실행"""
        start_time = time.time()

        steps = [
            self.step1_load_data,
            self.step2_extract_features,
            self.step3_prepare_targets,
            self.step4_train_exponential_model,
            self.step5_train_lstm_model,
            self.step6_generate_visualizations,
            self.step7_print_summary
        ]

        for i, step in enumerate(steps, 1):
            try:
                if not step():
                    print(f"\n❌ 파이프라인이 {i}단계에서 중단되었습니다.")
                    return False
            except KeyboardInterrupt:
                print(f"\n⚠️ 사용자에 의해 중단되었습니다.")
                return False
            except Exception as e:
                print(f"\n❌ {i}단계에서 예상치 못한 오류: {e}")
                return False

        # 실행 시간
        execution_time = time.time() - start_time
        print(f"\n⏱️  총 실행 시간: {execution_time:.1f}초")
        print("🎉 파이프라인 실행 완료!")

        return True


def main():
    """메인 실행 함수"""
    print("🌪️  Wind Turbine Bearing RUL Prediction")
    print("=" * 60)

    # 모듈 import 확인
    import_ok, resolved_components = check_imports()
    if not import_ok:
        print("❌ 필수 모듈을 import할 수 없습니다.")
        print("다음 명령으로 필요한 패키지를 설치하세요:")
        print("pip install numpy pandas scipy scikit-learn matplotlib seaborn requests")
        print("LSTM 기능을 위해: pip install tensorflow")
        return False

    # 재현성 확보 및 출력 디렉토리 준비
    from config import ensure_output_dirs
    from utils import set_global_seed
    set_global_seed()
    ensure_output_dirs()

    # 명령행 인자 처리
    use_lstm = '--no-lstm' not in sys.argv
    use_visualization = '--no-plot' not in sys.argv

    if not resolved_components['tensorflow_available']:
        use_lstm = False
        print("⚠️  TensorFlow를 사용할 수 없어 LSTM 기능이 비활성화됩니다.")

    if not resolved_components['visualizer']:
        use_visualization = False
        print("⚠️  시각화 모듈을 사용할 수 없어 시각화 기능이 비활성화됩니다.")

    # 파이프라인 실행
    pipeline = SimpleRULPipeline(
        components=resolved_components,
        use_lstm=use_lstm,
        use_visualization=use_visualization,
    )

    success = pipeline.run_pipeline()

    if not success:
        print("\n❌ 실행 중 오류가 발생했습니다.")
        print("문제 해결 방법:")
        print("  1. 필요한 패키지 설치: pip install -r requirements.txt")
        print("  2. Python 버전 확인 (3.7 이상 권장)")
        print("  3. 인터넷 연결 확인 (데이터 다운로드용)")

    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 치명적 오류: {e}")
        sys.exit(1)