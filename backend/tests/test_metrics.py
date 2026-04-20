import numpy as np

from utils import PerformanceAnalyzer


def test_calculate_metrics_perfect_prediction():
    true_vals = np.array([10.0, 20.0, 30.0, 40.0])
    result = PerformanceAnalyzer.calculate_metrics(true_vals, true_vals)
    assert result['mae'] == 0
    assert result['rmse'] == 0
    assert result['r2'] > 0.999


def test_prognostic_horizon_within_alpha():
    true_rul = np.array([100.0, 50.0, 10.0])
    pred_rul = np.array([100.0, 45.0, 12.0])  # 모두 20% 이내
    assert PerformanceAnalyzer.calculate_prognostic_horizon(true_rul, pred_rul, alpha=0.2) == 1.0


def test_calculate_metrics_handles_inf():
    true_vals = np.array([1.0, 2.0, 3.0])
    preds = np.array([1.0, np.inf, 3.0])
    result = PerformanceAnalyzer.calculate_metrics(true_vals, preds)
    assert result['n_valid'] == 2
    assert np.isfinite(result['mae'])
