"""
Prediction models for Wind Turbine Bearing RUL Prediction
풍력 터빈 베어링 RUL 예측을 위한 예측 모델들
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Optional, List, Tuple, Dict, Any
import warnings
import pickle
import json
from abc import ABC, abstractmethod

# TensorFlow 및 LSTM 관련 import
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow가 설치되지 않았습니다. LSTM 기능이 비활성화됩니다.")
    print("LSTM 기능을 사용하려면: pip install tensorflow")

from config import *

class BaseRULPredictor(ABC):
    """RUL 예측 모델의 기본 클래스"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        
    @abstractmethod
    def train(self, features: np.ndarray, targets: np.ndarray, **kwargs):
        """모델 훈련"""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """RUL 예측"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """모델 저장"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """모델 로드"""
        pass
    
    def evaluate(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """모델 성능 평가"""
        predictions = self.predict(features)
        
        # 길이 맞추기
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
        
        # 유효한 값들만 사용
        valid_mask = np.isfinite(predictions) & np.isfinite(targets)
        predictions_valid = predictions[valid_mask]
        targets_valid = targets[valid_mask]
        
        if len(predictions_valid) == 0:
            return {'mae': np.inf, 'rmse': np.inf, 'mape': np.inf}
        
        mae = mean_absolute_error(targets_valid, predictions_valid)
        rmse = np.sqrt(mean_squared_error(targets_valid, predictions_valid))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((targets_valid - predictions_valid) / 
                             (targets_valid + 1e-10))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'n_valid_predictions': len(predictions_valid)
        }


class ExponentialDegradationModel(BaseRULPredictor):
    """지수적 열화 모델"""
    
    def __init__(self, **config):
        super().__init__("ExponentialDegradation")
        
        # 설정값 로드
        model_config = EXP_MODEL_CONFIG.copy()
        model_config.update(config)
        
        self.theta = model_config['theta']
        self.theta_variance = model_config['theta_variance']
        self.beta = model_config['beta']
        self.beta_variance = model_config['beta_variance']
        self.phi = model_config['phi']
        self.noise_variance_factor = model_config['noise_variance_factor']
        self.slope_detection_level = model_config['slope_detection_level']
        
        # 사후 분포 파라미터 초기화
        self.theta_posterior = self.theta
        self.theta_variance_posterior = self.theta_variance
        self.beta_posterior = self.beta
        self.beta_variance_posterior = self.beta_variance
        
        self.observations = []
        self.slope_detection_instant = None
        self.threshold = None
        self.noise_variance = 0.0
        
    def set_threshold(self, health_indicator: np.ndarray):
        """임계값 설정"""
        self.threshold = health_indicator[-1]  # 마지막 값을 임계값으로 사용
        
        # 노이즈 분산 계산
        noise_std = self.noise_variance_factor * self.threshold / (self.threshold + 1)
        self.noise_variance = noise_std ** 2
    
    def update(self, observation: List[float]):
        """새로운 관측값으로 모델 파라미터 업데이트"""
        time, hi_value = observation
        self.observations.append([time, hi_value])
        
        # 베이지안 파라미터 업데이트 (단순화된 구현)
        if len(self.observations) > 1:
            times = np.array([obs[0] for obs in self.observations])
            values = np.array([obs[1] for obs in self.observations])
            
            # 선형 회귀로 기울기 추정
            if len(times) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(times, values)
                
                # 베타 (기울기 파라미터) 사후 분포 업데이트
                self.beta_posterior = slope
                self.beta_variance_posterior = std_err**2 if std_err > 0 else self.beta_variance
                
                # 기울기 검출
                if p_value < self.slope_detection_level and self.slope_detection_instant is None:
                    self.slope_detection_instant = time
    
    def predict_rul(self, current_observation: List[float]) -> Tuple[float, List[float], Optional[Any]]:
        """현재 관측값에서 RUL 예측"""
        time, hi_value = current_observation
        
        if self.threshold is None:
            return float('inf'), [float('inf'), float('inf')], None
        
        if self.beta_posterior <= 0:
            # 양의 기울기가 검출되지 않은 경우
            return float('inf'), [float('inf'), float('inf')], None
        
        # 지수적 모델: hi(t) = theta * exp(beta * t + phi) + noise
        # 임계값에 도달하는 시간 계산
        if hi_value <= 0:
            hi_value = 1e-10  # 로그 계산을 위해 양수로 만듦
        
        try:
            time_to_failure = (np.log(self.threshold) - np.log(hi_value)) / self.beta_posterior

            if time_to_failure <= 0:
                return 0, [0, 0], None

            # 델타 방법으로 분산 전파: t* = (log(T) - log(h)) / β
            # Var(t*) ≈ (t*/β)^2 · Var(β) + (1/(h·β))^2 · Var(h)
            d_t_d_beta = -time_to_failure / self.beta_posterior
            d_t_d_h = -1.0 / (hi_value * self.beta_posterior)

            var_t = (
                d_t_d_beta ** 2 * self.beta_variance_posterior
                + d_t_d_h ** 2 * self.noise_variance
            )
            std_dev = float(np.sqrt(max(var_t, 0.0)))

            ci_lower = max(0, time_to_failure - 1.96 * std_dev)
            ci_upper = time_to_failure + 1.96 * std_dev

            return time_to_failure, [ci_lower, ci_upper], None
            
        except (ValueError, ZeroDivisionError):
            return float('inf'), [float('inf'), float('inf')], None
    
    def train(self, features: np.ndarray, targets: np.ndarray, 
              health_indicator: np.ndarray, **kwargs):
        """모델 훈련 (실시간 업데이트 방식)"""
        # 임계값 설정
        self.set_threshold(health_indicator)
        
        # 관측값들로 순차 업데이트
        for i, hi_value in enumerate(health_indicator[:-1]):  # 마지막 값 제외
            self.update([i, hi_value])
        
        self.is_trained = True
    
    def predict(self, features: np.ndarray, health_indicator: Optional[np.ndarray] = None) -> np.ndarray:
        """배치 예측"""
        if health_indicator is None:
            # 간단히 첫 번째 특징을 건강 지표로 사용
            health_indicator = features[:, 0] if features.ndim > 1 else features
        
        predictions = []
        
        for i, hi_value in enumerate(health_indicator):
            pred, _, _ = self.predict_rul([i, hi_value])
            # 매우 큰 값들을 제한
            pred = min(pred, 200) if not np.isinf(pred) else 100
            predictions.append(pred)
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        """모델 저장"""
        model_data = {
            'model_name': self.model_name,
            'theta': self.theta,
            'theta_variance': self.theta_variance,
            'beta': self.beta,
            'beta_variance': self.beta_variance,
            'phi': self.phi,
            'noise_variance_factor': self.noise_variance_factor,
            'slope_detection_level': self.slope_detection_level,
            'theta_posterior': self.theta_posterior,
            'theta_variance_posterior': self.theta_variance_posterior,
            'beta_posterior': self.beta_posterior,
            'beta_variance_posterior': self.beta_variance_posterior,
            'observations': self.observations,
            'slope_detection_instant': self.slope_detection_instant,
            'threshold': self.threshold,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    def load_model(self, filepath: str):
        """모델 로드"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # 모델 파라미터 복원
        for key, value in model_data.items():
            setattr(self, key, value)


class LSTMRULPredictor(BaseRULPredictor):
    """LSTM 기반 RUL 예측 모델"""
    
    def __init__(self, **config):
        super().__init__("LSTM")

        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다. pip install tensorflow")

        # 설정값 로드
        lstm_config = LSTM_CONFIG.copy()
        lstm_config.update(config)

        self.sequence_length = lstm_config['sequence_length']
        self.lstm_units = lstm_config['lstm_units']
        self.dropout_rate = lstm_config['dropout_rate']
        self.epochs = lstm_config['epochs']
        self.batch_size = lstm_config['batch_size']
        self.learning_rate = lstm_config['learning_rate']
        self.patience_early_stopping = lstm_config['patience_early_stopping']
        self.patience_lr_reduction = lstm_config['patience_lr_reduction']
        self.lr_reduction_factor = lstm_config['lr_reduction_factor']
        self.min_learning_rate = lstm_config['min_learning_rate']
        self.random_seed = lstm_config.get('random_seed', RANDOM_SEED)

        # 가중치 초기화 재현성 확보
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.model = None
        self.scaler_features = MinMaxScaler()
        self.scaler_targets = MinMaxScaler()
        self.training_history = None
    
    def create_sequences(self, features: np.ndarray, targets: Optional[np.ndarray] = None) -> Tuple:
        """시계열 시퀀스 생성.

        윈도우 features[i : i+seq_len] 로 targets[i+seq_len] 를 예측하는
        one-step-ahead 설정. 따라서 반환되는 예측 배열의 k번째 원소는
        원본 타임스텝 k+sequence_length 에 대응한다.
        """
        sequences = []
        sequence_targets = []

        for i in range(len(features) - self.sequence_length):
            sequences.append(features[i:i+self.sequence_length])
            if targets is not None:
                sequence_targets.append(targets[i+self.sequence_length])
        
        sequences = np.array(sequences)
        if targets is not None:
            sequence_targets = np.array(sequence_targets)
            return sequences, sequence_targets
        
        return sequences
    
    def build_model(self, n_features: int):
        """LSTM 모델 아키텍처 구축"""
        model = Sequential([
            LSTM(self.lstm_units, 
                 return_sequences=True, 
                 input_shape=(self.sequence_length, n_features),
                 name='lstm_1'),
            Dropout(self.dropout_rate, name='dropout_1'),
            
            LSTM(self.lstm_units // 2, 
                 return_sequences=False,
                 name='lstm_2'),
            Dropout(self.dropout_rate, name='dropout_2'),
            
            Dense(25, activation='relu', name='dense_1'),
            Dropout(self.dropout_rate, name='dropout_3'),
            
            Dense(1, activation='linear', name='output')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='huber',  # 이상치에 강건한 손실 함수
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, features: np.ndarray, targets: np.ndarray,
                    train_split: float = 0.7) -> Tuple:
        """LSTM 훈련을 위한 데이터 준비.

        데이터 누수 방지를 위해 스케일러는 훈련 구간으로만 fit하고
        검증 구간에는 transform만 적용한다.
        """
        split_idx = int(len(features) * train_split)
        if split_idx <= self.sequence_length:
            raise ValueError(
                f"훈련 구간이 sequence_length({self.sequence_length})보다 커야 합니다."
            )

        train_features = features[:split_idx]
        val_features = features[split_idx:]
        train_targets = targets[:split_idx]
        val_targets = targets[split_idx:]

        self.scaler_features.fit(train_features)
        self.scaler_targets.fit(train_targets.reshape(-1, 1))

        train_features_scaled = self.scaler_features.transform(train_features)
        val_features_scaled = self.scaler_features.transform(val_features)
        train_targets_scaled = self.scaler_targets.transform(
            train_targets.reshape(-1, 1)
        ).flatten()
        val_targets_scaled = self.scaler_targets.transform(
            val_targets.reshape(-1, 1)
        ).flatten()

        X_train, y_train = self.create_sequences(train_features_scaled, train_targets_scaled)
        X_val, y_val = self.create_sequences(val_features_scaled, val_targets_scaled)

        return X_train, X_val, y_train, y_val
    
    def train(self, features: np.ndarray, targets: np.ndarray, **kwargs):
        """LSTM 모델 훈련"""
        verbose = kwargs.get('verbose', 1)
        
        # 데이터 준비
        X_train, X_val, y_train, y_val = self.prepare_data(features, targets)
        
        if len(X_train) == 0:
            raise ValueError("훈련 데이터가 충분하지 않습니다.")
        
        # 모델 구축
        if self.model is None:
            self.build_model(X_train.shape[2])
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.patience_early_stopping,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.lr_reduction_factor,
                patience=self.patience_lr_reduction,
                min_lr=self.min_learning_rate,
                verbose=verbose
            )
        ]
        
        # 모델 훈련
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        return self.training_history
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """RUL 예측"""
        if not self.is_trained or self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 특징 정규화
        features_scaled = self.scaler_features.transform(features)
        
        # 시퀀스 생성
        X = self.create_sequences(features_scaled)
        
        if len(X) == 0:
            return np.array([])
        
        # 예측
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            predictions_scaled = self.model.predict(X, verbose=0)
        
        # 역정규화
        predictions = self.scaler_targets.inverse_transform(predictions_scaled).flatten()
        
        # 음수 예측값 제거
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_sequence(self, initial_features: np.ndarray, 
                        n_steps: int) -> np.ndarray:
        """순환적 다단계 예측"""
        if not self.is_trained or self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 초기 특징 정규화
        features_scaled = self.scaler_features.transform(initial_features)
        
        if len(features_scaled) < self.sequence_length:
            raise ValueError(f"최소 {self.sequence_length}개 데이터 포인트가 필요합니다.")
        
        # 마지막 시퀀스로 시작
        current_sequence = features_scaled[-self.sequence_length:].copy()
        predictions = []
        
        for _ in range(n_steps):
            # 현재 시퀀스로 예측
            X = current_sequence.reshape(1, self.sequence_length, -1)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                pred_scaled = self.model.predict(X, verbose=0)[0, 0]
            
            # 역정규화
            pred = self.scaler_targets.inverse_transform([[pred_scaled]])[0, 0]
            predictions.append(max(0, pred))
            
            # 시퀀스 업데이트 (간단한 방법: 마지막 특징 벡터 복사)
            current_sequence[:-1] = current_sequence[1:]
            # 실제로는 새로운 특징을 예측하거나 가정해야 하지만, 여기서는 단순화
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        # Keras 모델 저장
        model_path = filepath.replace('.pkl', '.h5')
        self.model.save(model_path)
        
        # 스케일러 및 기타 정보 저장
        metadata = {
            'model_name': self.model_name,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'is_trained': self.is_trained,
            'model_path': model_path
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'metadata': metadata,
                'scaler_features': self.scaler_features,
                'scaler_targets': self.scaler_targets,
                'training_history': self.training_history.history if self.training_history else None
            }, f)
    
    def load_model(self, filepath: str):
        """모델 로드"""
        # 메타데이터 및 스케일러 로드
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        metadata = data['metadata']
        self.scaler_features = data['scaler_features']
        self.scaler_targets = data['scaler_targets']
        
        # 모델 속성 복원
        for key, value in metadata.items():
            if key != 'model_path':
                setattr(self, key, value)
        
        # Keras 모델 로드
        self.model = load_model(metadata['model_path'])


class EnsembleRULPredictor(BaseRULPredictor):
    """앙상블 RUL 예측 모델"""
    
    def __init__(self, model_configs: Optional[List[Dict]] = None):
        super().__init__("Ensemble")
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다.")
        
        if model_configs is None:
            model_configs = ENSEMBLE_CONFIG['model_configs']
        
        self.model_configs = model_configs
        self.models = []
        self.weights = None
    
    def create_models(self):
        """앙상블 모델들 생성"""
        self.models = []
        
        for config in self.model_configs:
            model = LSTMRULPredictor(**config)
            self.models.append(model)
    
    def train(self, features: np.ndarray, targets: np.ndarray, **kwargs):
        """앙상블 모델들 훈련"""
        if not self.models:
            self.create_models()
        
        verbose = kwargs.get('verbose', 0)
        
        trained_models = []
        model_performances = []
        
        for i, model in enumerate(self.models):
            print(f"앙상블 모델 {i+1}/{len(self.models)} 훈련 중...")
            
            try:
                # 부트스트랩 샘플링
                n_samples = len(features)
                bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
                
                bootstrap_features = features[bootstrap_idx]
                bootstrap_targets = targets[bootstrap_idx]
                
                # 모델 훈련
                model.train(bootstrap_features, bootstrap_targets, verbose=verbose)
                
                # 성능 평가 (out-of-bag 샘플 사용)
                oob_idx = np.setdiff1d(np.arange(n_samples), np.unique(bootstrap_idx))
                if len(oob_idx) > model.sequence_length:
                    oob_features = features[oob_idx]
                    oob_targets = targets[oob_idx]
                    
                    performance = model.evaluate(oob_features, oob_targets)
                    model_performances.append(performance['mae'])
                else:
                    model_performances.append(np.inf)
                
                trained_models.append(model)
                
            except Exception as e:
                print(f"모델 {i+1} 훈련 실패: {e}")
                model_performances.append(np.inf)
        
        self.models = trained_models
        
        # 성능에 기반한 가중치 계산 (역수 사용)
        performances = np.array(model_performances)
        valid_performances = performances[np.isfinite(performances)]
        
        if len(valid_performances) > 0:
            # MAE가 낮을수록 높은 가중치
            weights = 1.0 / (valid_performances + 1e-6)
            weights = weights / np.sum(weights)
            
            # 무한대 성능의 모델들에는 0 가중치
            full_weights = np.zeros(len(performances))
            full_weights[np.isfinite(performances)] = weights
            
            self.weights = full_weights
        else:
            # 모든 모델이 실패한 경우 균등 가중치
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        self.is_trained = len(trained_models) > 0
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """앙상블 예측"""
        if not self.is_trained:
            raise ValueError("앙상블 모델이 훈련되지 않았습니다.")
        
        predictions_list = []
        
        for model in self.models:
            try:
                pred = model.predict(features)
                if len(pred) > 0:
                    predictions_list.append(pred)
                else:
                    predictions_list.append(None)
            except Exception as e:
                print(f"모델 예측 실패: {e}")
                predictions_list.append(None)
        
        # 유효한 예측들만 사용
        valid_predictions = [p for p in predictions_list if p is not None]
        
        if not valid_predictions:
            return np.array([])
        
        # 길이 맞추기
        min_length = min(len(p) for p in valid_predictions)
        aligned_predictions = [p[:min_length] for p in valid_predictions]
        
        # 가중 평균 계산
        if self.weights is not None:
            valid_weights = self.weights[:len(aligned_predictions)]
            valid_weights = valid_weights / np.sum(valid_weights)
            
            ensemble_pred = np.average(aligned_predictions, axis=0, weights=valid_weights)
        else:
            ensemble_pred = np.mean(aligned_predictions, axis=0)
        
        return ensemble_pred
    
    def get_prediction_uncertainty(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """예측 불확실성 계산"""
        predictions_list = []
        
        for model in self.models:
            try:
                pred = model.predict(features)
                if len(pred) > 0:
                    predictions_list.append(pred)
            except Exception:
                continue
        
        if not predictions_list:
            return np.array([]), np.array([])
        
        # 길이 맞추기
        min_length = min(len(p) for p in predictions_list)
        aligned_predictions = [p[:min_length] for p in predictions_list]
        
        # 평균과 표준편차 계산
        ensemble_mean = np.mean(aligned_predictions, axis=0)
        ensemble_std = np.std(aligned_predictions, axis=0)
        
        return ensemble_mean, ensemble_std
    
    def save_model(self, filepath: str):
        """앙상블 모델 저장"""
        # 개별 모델들 저장
        model_paths = []
        
        for i, model in enumerate(self.models):
            model_path = filepath.replace('.pkl', f'_model_{i}.pkl')
            model.save_model(model_path)
            model_paths.append(model_path)
        
        # 앙상블 메타데이터 저장
        metadata = {
            'model_name': self.model_name,
            'model_configs': self.model_configs,
            'model_paths': model_paths,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, filepath: str):
        """앙상블 모델 로드"""
        # 메타데이터 로드
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        # 속성 복원
        self.model_configs = metadata['model_configs']
        self.weights = np.array(metadata['weights']) if metadata['weights'] else None
        self.is_trained = metadata['is_trained']
        
        # 개별 모델들 로드
        self.models = []
        for model_path in metadata['model_paths']:
            model = LSTMRULPredictor()
            model.load_model(model_path)
            self.models.append(model)


def compare_models(models: Dict[str, BaseRULPredictor], 
                  features: np.ndarray, 
                  targets: np.ndarray) -> Dict[str, Dict[str, float]]:
    """여러 모델의 성능 비교"""
    results = {}
    
    for model_name, model in models.items():
        try:
            performance = model.evaluate(features, targets)
            results[model_name] = performance
            
            print(f"{model_name} 모델:")
            print(f"  MAE: {performance['mae']:.3f}")
            print(f"  RMSE: {performance['rmse']:.3f}")
            print(f"  MAPE: {performance['mape']:.3f}%")
            print(f"  유효 예측 수: {performance['n_valid_predictions']}")
            
        except Exception as e:
            print(f"{model_name} 모델 평가 실패: {e}")
            results[model_name] = {'mae': np.inf, 'rmse': np.inf, 'mape': np.inf}
    
    # 최고 성능 모델 찾기
    best_model = min(results.keys(), key=lambda k: results[k]['mae'])
    print(f"\n🏆 최고 성능 모델: {best_model} (MAE: {results[best_model]['mae']:.3f})")
    
    return results


def main():
    """테스트 및 데모 함수"""
    from data_loader import DataLoader
    from feature_extractor import FeatureExtractor, FeatureSelector, DimensionReducer
    
    print("예측 모델 테스트")
    print("=" * 50)
    
    # 데이터 로드 및 특징 추출
    loader = DataLoader()
    data = loader.load_wind_turbine_data()
    
    extractor = FeatureExtractor()
    features_df = extractor.extract_features_from_dataframe(data, show_progress=False)
    smoothed_features = extractor.apply_smoothing(features_df)
    
    # 특징 선택
    selector = FeatureSelector()
    train_end = int(0.6 * len(smoothed_features))
    train_features_df = smoothed_features.iloc[:train_end]
    
    selected_features = selector.select_features_by_monotonicity(train_features_df)
    feature_selected = smoothed_features[selected_features]
    
    # PCA
    reducer = DimensionReducer()
    train_pca = reducer.fit_transform(feature_selected.iloc[:train_end])
    all_pca = reducer.transform(feature_selected)
    
    health_indicator = reducer.get_health_indicator(all_pca)
    
    # RUL 타겟 생성
    total_days = len(health_indicator)
    true_rul = np.array([total_days - i - 1 for i in range(total_days)])
    
    # 데이터 분할
    features_array = np.column_stack([feature_selected.values, health_indicator.reshape(-1, 1)])
    
    # 지수적 모델 테스트
    print("1. 지수적 열화 모델 테스트...")
    exp_model = ExponentialDegradationModel()
    exp_model.train(features_array, true_rul, health_indicator)
    exp_predictions = exp_model.predict(features_array, health_indicator[:-1])
    
    print("지수적 모델 훈련 완료")
    
    # LSTM 모델 테스트 (TensorFlow 사용 가능한 경우)
    if TENSORFLOW_AVAILABLE:
        print("\n2. LSTM 모델 테스트...")
        lstm_model = LSTMRULPredictor(sequence_length=8, epochs=50)
        
        try:
            lstm_model.train(features_array, true_rul, verbose=0)
            lstm_predictions = lstm_model.predict(features_array)
            
            print("LSTM 모델 훈련 완료")
            
            # 모델 비교
            print("\n3. 모델 성능 비교...")
            models = {
                'Exponential': exp_model,
                'LSTM': lstm_model
            }
            
            # 테스트 데이터로 평가
            test_start = int(0.8 * len(features_array))
            test_features = features_array[test_start:]
            test_targets = true_rul[test_start:]
            
            results = compare_models(models, test_features, test_targets)
            
        except Exception as e:
            print(f"LSTM 모델 테스트 실패: {e}")
    
    else:
        print("\nTensorFlow를 사용할 수 없어 LSTM 테스트를 건너뜁니다.")
    
    print("모델 테스트 완료!")


if __name__ == "__main__":
    main()
