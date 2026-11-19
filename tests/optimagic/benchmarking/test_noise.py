import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic.benchmarking.noise import (
    AdditiveBenchmarkNoise,
    BenchmarkNoise,
    CombinedBenchmarkNoise,
    MultiplicativeBenchmarkNoise,
    pre_process_noise,
)
from optimagic.utilities import get_rng

NOISE_CLASSES = [AdditiveBenchmarkNoise, MultiplicativeBenchmarkNoise]


@pytest.mark.parametrize("noise_class", NOISE_CLASSES)
def test_noise_has_type_and_shape_of_function_value(noise_class):
    noise = noise_class()
    params = np.arange(3.0)

    scalar_draw = noise.draw_noise(2.0, params=params, rng=get_rng(0))
    assert isinstance(scalar_draw, float)

    array_draw = noise.draw_noise(np.arange(5.0), params=params, rng=get_rng(0))
    assert isinstance(array_draw, np.ndarray)
    assert array_draw.shape == (5,)


@pytest.mark.parametrize("noise_class", NOISE_CLASSES)
def test_noise_is_reproducible_with_same_rng(noise_class):
    noise = noise_class(distribution="laplace", correlation=0.3)
    fval = np.arange(5.0)
    first = noise.draw_noise(fval, params=np.ones(2), rng=get_rng(1))
    second = noise.draw_noise(fval, params=np.ones(2), rng=get_rng(1))
    aaae(first, second)


def test_additive_noise_has_requested_std():
    noise = AdditiveBenchmarkNoise(std=0.5)
    draw = noise.draw_noise(np.zeros(100_000), params=np.ones(2), rng=get_rng(0))
    assert np.allclose(draw.mean(), 0, atol=0.01)
    assert np.allclose(draw.std(), 0.5, atol=0.01)


def test_multiplicative_noise_scales_with_function_value():
    noise = MultiplicativeBenchmarkNoise(std=0.1, clipping_value=1.0)
    fval = np.concatenate([np.full(100_000, 10.0), np.full(100_000, -50.0)])
    draw = noise.draw_noise(fval, params=np.ones(2), rng=get_rng(0))
    assert np.allclose(draw[:100_000].std(), 1.0, atol=0.02)
    assert np.allclose(draw[100_000:].std(), 5.0, atol=0.1)


def test_multiplicative_noise_is_not_zero_at_zero():
    noise = MultiplicativeBenchmarkNoise(std=0.1, clipping_value=2.0)
    draw = noise.draw_noise(np.zeros(100_000), params=np.ones(2), rng=get_rng(0))
    assert np.allclose(draw.std(), 0.2, atol=0.01)

    scalar_draw = noise.draw_noise(0.0, params=np.ones(2), rng=get_rng(0))
    assert scalar_draw != 0


def test_adding_noise_combines_the_components():
    additive = AdditiveBenchmarkNoise(std=0.3)
    multiplicative = MultiplicativeBenchmarkNoise(std=0.2)
    combined = additive + multiplicative

    assert isinstance(combined, CombinedBenchmarkNoise)
    assert combined.components == (additive, multiplicative)

    fval = np.arange(4.0)
    params = np.ones(2)
    rng = get_rng(5)
    expected = additive.draw_noise(fval, params, rng) + multiplicative.draw_noise(
        fval, params, rng
    )
    aaae(combined.draw_noise(fval, params, get_rng(5)), expected)


def test_adding_combined_noise_keeps_a_flat_list_of_components():
    first = AdditiveBenchmarkNoise(std=0.3)
    second = MultiplicativeBenchmarkNoise(std=0.2)
    third = AdditiveBenchmarkNoise(distribution="uniform")

    assert ((first + second) + third).components == (first, second, third)
    assert (first + (second + third)).components == (first, second, third)


def test_combined_noise_without_components_raises():
    with pytest.raises(ValueError, match="at least one source of noise"):
        CombinedBenchmarkNoise(components=())


def test_adding_something_else_to_noise_raises():
    with pytest.raises(TypeError):
        AdditiveBenchmarkNoise() + 1  # ty: ignore[unsupported-operator]


def test_custom_noise_can_depend_on_params():
    class Wiggle(BenchmarkNoise):
        def draw_noise(self, fval, params, rng):  # noqa: ARG002
            return 0.1 * np.sin(params.sum()) + 0 * fval

    combined = Wiggle() + AdditiveBenchmarkNoise(std=0.0)
    got = combined.draw_noise(3.0, params=np.array([0.5, 0.5]), rng=get_rng(0))
    assert np.allclose(got, 0.1 * np.sin(1.0))


@pytest.mark.parametrize("noise_class", NOISE_CLASSES)
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"distribution": "cauchy"}, "Invalid distribution"),
        ({"std": -1.0}, "std must be non-negative"),
        ({"correlation": -0.1}, "correlation must be between 0 and 1"),
        ({"correlation": 1.1}, "correlation must be between 0 and 1"),
    ],
)
def test_invalid_noise_arguments_raise(noise_class, kwargs, match):
    with pytest.raises(ValueError, match=match):
        noise_class(**kwargs)


def test_negative_clipping_value_raises():
    with pytest.raises(ValueError, match="clipping_value must be non-negative"):
        MultiplicativeBenchmarkNoise(clipping_value=-1.0)


def test_pre_process_noise():
    noise = MultiplicativeBenchmarkNoise()
    assert pre_process_noise(noise) is noise
    assert pre_process_noise(False) is None
    assert pre_process_noise(True) == AdditiveBenchmarkNoise()


def test_pre_process_noise_with_invalid_type_raises():
    with pytest.raises(TypeError, match="noise must be a bool or an instance"):
        pre_process_noise("normal")  # ty: ignore[invalid-argument-type]
