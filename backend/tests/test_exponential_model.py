import numpy as np

from models import ExponentialDegradationModel


def test_predict_rul_ci_widens_with_observation_noise():
    model = ExponentialDegradationModel()
    model.beta_posterior = 0.05
    model.beta_variance_posterior = 1e-6
    model.threshold = 5.0
    # 노이즈가 0일 때 CI
    model.noise_variance = 0.0
    pred_no_noise, (lo0, hi0), _ = model.predict_rul([0.0, 1.0])
    # 노이즈가 큰 경우
    model.noise_variance = 1.0
    pred_with_noise, (lo1, hi1), _ = model.predict_rul([0.0, 1.0])

    assert pred_no_noise == pred_with_noise
    assert (hi1 - lo1) > (hi0 - lo0)


def test_predict_rul_returns_zero_when_threshold_passed():
    model = ExponentialDegradationModel()
    model.beta_posterior = 0.1
    model.beta_variance_posterior = 1e-6
    model.threshold = 2.0
    # hi_value 가 threshold 보다 크면 time_to_failure <= 0 → 0 반환
    rul, ci, _ = model.predict_rul([0.0, 10.0])
    assert rul == 0
    assert ci == [0, 0]
