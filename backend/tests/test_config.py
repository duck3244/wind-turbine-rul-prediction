import os

import config


def test_random_seed_is_defined():
    assert isinstance(config.RANDOM_SEED, int)


def test_split_ratios_are_ordered():
    assert 0 < config.TRAIN_SPLIT < config.VALIDATION_SPLIT < 1


def test_ensure_output_dirs_creates_paths(tmp_path, monkeypatch):
    results = tmp_path / "results"
    models = tmp_path / "models"
    plots = tmp_path / "plots"
    monkeypatch.setattr(config, "RESULTS_DIR", str(results))
    monkeypatch.setattr(config, "MODELS_DIR", str(models))
    monkeypatch.setattr(config, "PLOTS_DIR", str(plots))

    config.ensure_output_dirs()

    assert results.is_dir()
    assert models.is_dir()
    assert plots.is_dir()


def test_prediction_bounds_are_reasonable():
    assert config.RUL_PREDICTION_UPPER_BOUND > 0
    assert 0 <= config.RUL_PREDICTION_FALLBACK <= config.RUL_PREDICTION_UPPER_BOUND
