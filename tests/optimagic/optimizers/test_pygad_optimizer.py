"""Test helper functions for PyGAD optimizer."""

import warnings

import pytest

from optimagic.optimizers.pygad_optimizer import (
    AdaptiveMutation,
    InversionMutation,
    RandomMutation,
    ScrambleMutation,
    SwapMutation,
    _convert_mutation_to_pygad_params,
    _create_mutation_from_string,
    _determine_effective_batch_size,
    _get_default_mutation_params,
)


@pytest.mark.parametrize(
    "batch_size, n_cores, expected",
    [
        (None, 1, None),
        (None, 4, 4),
        (10, 4, 10),
        (4, 4, 4),
        (2, 4, 2),
        (5, 1, 5),
        (0, 4, 0),
        (None, 100, 100),
        (1, 1, 1),
    ],
)
def test_determine_effective_batch_size_return_values(batch_size, n_cores, expected):
    result = _determine_effective_batch_size(batch_size, n_cores)
    assert result == expected


@pytest.mark.parametrize(
    "batch_size, n_cores, should_warn",
    [
        (2, 4, True),
        (1, 8, True),
        (0, 4, True),
        (4, 4, False),
        (8, 4, False),
        (None, 4, False),
        (5, 1, False),
        (None, 1, False),
    ],
)
def test_determine_effective_batch_size_warnings(batch_size, n_cores, should_warn):
    if should_warn:
        warning_pattern = (
            f"batch_size \\({batch_size}\\) is smaller than "
            f"n_cores \\({n_cores}\\)\\. This may reduce parallel efficiency\\. "
            f"Consider setting batch_size >= n_cores\\."
        )
        with pytest.warns(UserWarning, match=warning_pattern):
            result = _determine_effective_batch_size(batch_size, n_cores)
            assert result == batch_size
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _determine_effective_batch_size(batch_size, n_cores)


# Tests for _get_default_mutation_params
@pytest.mark.parametrize(
    "mutation_type, expected",
    [
        (
            "random",
            {
                "mutation_type": "random",
                "mutation_probability": None,
                "mutation_percent_genes": "default",
                "mutation_num_genes": None,
                "mutation_by_replacement": False,
            },
        ),
        (
            None,
            {
                "mutation_type": None,
                "mutation_probability": None,
                "mutation_percent_genes": None,
                "mutation_num_genes": None,
                "mutation_by_replacement": None,
            },
        ),
    ],
)
def test_get_default_mutation_params(mutation_type, expected):
    result = _get_default_mutation_params(mutation_type)
    assert result == expected


# Tests for _create_mutation_from_string
@pytest.mark.parametrize(
    "mutation_type, expected_class",
    [
        ("random", RandomMutation),
        ("swap", SwapMutation),
        ("inversion", InversionMutation),
        ("scramble", ScrambleMutation),
        ("adaptive", AdaptiveMutation),
    ],
)
def test_create_mutation_from_string_valid(mutation_type, expected_class):
    result = _create_mutation_from_string(mutation_type)
    assert isinstance(result, expected_class)


def test_create_mutation_from_string_invalid():
    with pytest.raises(ValueError, match="Unsupported mutation type: invalid"):
        _create_mutation_from_string("invalid")


# Tests for _convert_mutation_to_pygad_params
def test_convert_mutation_none():
    result = _convert_mutation_to_pygad_params(None)
    expected = {
        "mutation_type": None,
        "mutation_probability": None,
        "mutation_percent_genes": None,
        "mutation_num_genes": None,
        "mutation_by_replacement": None,
    }
    assert result == expected


@pytest.mark.parametrize(
    "mutation_string",
    ["random", "swap", "inversion", "scramble", "adaptive"],
)
def test_convert_mutation_string(mutation_string):
    result = _convert_mutation_to_pygad_params(mutation_string)
    assert result["mutation_type"] == mutation_string
    assert "mutation_probability" in result
    assert "mutation_percent_genes" in result
    assert "mutation_num_genes" in result
    assert "mutation_by_replacement" in result


@pytest.mark.parametrize(
    "mutation_class",
    [
        RandomMutation,
        SwapMutation,
        InversionMutation,
        ScrambleMutation,
        AdaptiveMutation,
    ],
)
def test_convert_mutation_class(mutation_class):
    result = _convert_mutation_to_pygad_params(mutation_class)
    assert result["mutation_type"] == mutation_class.mutation_type
    assert "mutation_probability" in result
    assert "mutation_percent_genes" in result
    assert "mutation_num_genes" in result
    assert "mutation_by_replacement" in result


def test_convert_mutation_instance():
    # Test RandomMutation instance
    mutation = RandomMutation(probability=0.2, by_replacement=True)
    result = _convert_mutation_to_pygad_params(mutation)
    assert result["mutation_type"] == "random"
    assert result["mutation_probability"] == 0.2
    assert result["mutation_by_replacement"] is True

    # Test SwapMutation instance
    mutation = SwapMutation()
    result = _convert_mutation_to_pygad_params(mutation)
    assert result["mutation_type"] == "swap"

    # Test AdaptiveMutation instance
    mutation = AdaptiveMutation(probability_bad=0.3, probability_good=0.1)
    result = _convert_mutation_to_pygad_params(mutation)
    assert result["mutation_type"] == "adaptive"
    assert result["mutation_probability"] == [0.3, 0.1]


def test_convert_mutation_custom_function():
    def custom_mutation(offspring, ga_instance):
        return offspring

    result = _convert_mutation_to_pygad_params(custom_mutation)
    assert result["mutation_type"] == custom_mutation


def test_convert_mutation_invalid_type():
    with pytest.raises(ValueError, match="Unsupported mutation type"):
        _convert_mutation_to_pygad_params(123)


# Tests for mutation dataclasses
def test_random_mutation_default():
    mutation = RandomMutation()
    result = mutation.to_pygad_params()
    assert result["mutation_type"] == "random"
    assert result["mutation_probability"] is None
    assert result["mutation_percent_genes"] == "default"
    assert result["mutation_num_genes"] is None
    assert result["mutation_by_replacement"] is False


def test_random_mutation_with_parameters():
    mutation = RandomMutation(
        probability=0.15, num_genes=5, percent_genes=20.0, by_replacement=True
    )
    result = mutation.to_pygad_params()
    assert result["mutation_type"] == "random"
    assert result["mutation_probability"] == 0.15
    assert result["mutation_percent_genes"] == 20.0
    assert result["mutation_num_genes"] == 5
    assert result["mutation_by_replacement"] is True


@pytest.mark.parametrize(
    "mutation_class, expected_type",
    [
        (SwapMutation, "swap"),
        (InversionMutation, "inversion"),
        (ScrambleMutation, "scramble"),
    ],
)
def test_simple_mutations(mutation_class, expected_type):
    mutation = mutation_class()
    result = mutation.to_pygad_params()
    assert result["mutation_type"] == expected_type
    assert result["mutation_probability"] is None
    assert result["mutation_percent_genes"] == "default"
    assert result["mutation_num_genes"] is None
    assert result["mutation_by_replacement"] is False


def test_adaptive_mutation_default():
    mutation = AdaptiveMutation()
    result = mutation.to_pygad_params()
    assert result["mutation_type"] == "adaptive"
    assert result["mutation_probability"] == [0.1, 0.05]  # Default values
    assert result["mutation_percent_genes"] == None
    assert result["mutation_num_genes"] is None
    assert result["mutation_by_replacement"] is False


def test_adaptive_mutation_with_probabilities():
    mutation = AdaptiveMutation(probability_bad=0.2, probability_good=0.08)
    result = mutation.to_pygad_params()
    assert result["mutation_type"] == "adaptive"
    assert result["mutation_probability"] == [0.2, 0.08]
    assert result["mutation_percent_genes"] == None
    assert result["mutation_num_genes"] is None
    assert result["mutation_by_replacement"] is False


def test_adaptive_mutation_with_num_genes():
    mutation = AdaptiveMutation(num_genes_bad=10, num_genes_good=5)
    result = mutation.to_pygad_params()
    assert result["mutation_type"] == "adaptive"
    assert result["mutation_probability"] is None
    assert result["mutation_num_genes"] == [10, 5]
    assert result["mutation_percent_genes"] == None
    assert result["mutation_by_replacement"] is False


def test_adaptive_mutation_with_percent_genes():
    mutation = AdaptiveMutation(percent_genes_bad=25.0, percent_genes_good=10.0)
    result = mutation.to_pygad_params()
    assert result["mutation_type"] == "adaptive"
    assert result["mutation_probability"] is None
    assert result["mutation_num_genes"] is None
    assert result["mutation_percent_genes"] == [25.0, 10.0]
    assert result["mutation_by_replacement"] is False


def test_mutation_type_class_variables():
    assert RandomMutation.mutation_type == "random"
    assert SwapMutation.mutation_type == "swap"
    assert InversionMutation.mutation_type == "inversion"
    assert ScrambleMutation.mutation_type == "scramble"
    assert AdaptiveMutation.mutation_type == "adaptive"
