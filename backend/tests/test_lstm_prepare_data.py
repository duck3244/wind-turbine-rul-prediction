"""LSTM prepare_data의 데이터 누수 회귀 테스트."""
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from models import LSTMRULPredictor


def _synthetic(n=60, d=3):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, d))
    targets = np.arange(n, dtype=float)[::-1]
    return features, targets


def test_prepare_data_fits_scaler_on_train_only():
    features, targets = _synthetic()
    model = LSTMRULPredictor(sequence_length=5, epochs=1)

    split = 0.7
    model.prepare_data(features, targets, train_split=split)

    train_end = int(len(features) * split)
    expected_min = features[:train_end].min(axis=0)
    expected_max = features[:train_end].max(axis=0)

    # MinMaxScaler 는 data_min_ / data_max_ 에 fit 범위를 저장
    np.testing.assert_allclose(model.scaler_features.data_min_, expected_min)
    np.testing.assert_allclose(model.scaler_features.data_max_, expected_max)


def test_prepare_data_raises_when_train_too_small():
    features, targets = _synthetic(n=20)
    model = LSTMRULPredictor(sequence_length=15, epochs=1)
    with pytest.raises(ValueError):
        model.prepare_data(features, targets, train_split=0.5)


def test_create_sequences_shapes_and_alignment():
    model = LSTMRULPredictor(sequence_length=4, epochs=1)
    features = np.arange(20).reshape(20, 1).astype(float)
    targets = np.arange(20, dtype=float)
    X, y = model.create_sequences(features, targets)

    # 규약: X[k] = features[k:k+seq_len], y[k] = targets[k+seq_len]
    assert X.shape == (16, 4, 1)
    assert y.shape == (16,)
    np.testing.assert_array_equal(X[0].flatten(), np.arange(4))
    assert y[0] == 4
    np.testing.assert_array_equal(X[-1].flatten(), np.arange(15, 19))
    assert y[-1] == 19
