import random

import numpy as np

from utils import set_global_seed


def test_set_global_seed_reproducible_numpy():
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_global_seed_reproducible_random():
    set_global_seed(7)
    a = [random.random() for _ in range(5)]
    set_global_seed(7)
    b = [random.random() for _ in range(5)]
    assert a == b
