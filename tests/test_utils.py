import numpy as np

from fjd.utils.config import load_params
from fjd.utils.seed import set_seed


def test_load_params_has_seed() -> None:
    params = load_params()
    assert isinstance(params["seed"], int)


def test_set_seed_is_deterministic() -> None:
    set_seed(123)
    a = np.random.rand(3)
    set_seed(123)
    b = np.random.rand(3)
    assert np.array_equal(a, b)
