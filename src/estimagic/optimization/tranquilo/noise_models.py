from typing import NamedTuple

import numpy as np


class NoiseModel(NamedTuple):
    intercept: float = None
    slope: float = None
    square: float = None


def evaluate_noise_model(fvals, noise_model):
    """Add noise to fvals according to noise_model."""

    if noise_model.distribution != "normal":
        raise NotImplementedError()

    if (noise_model.intercept, noise_model.slope, noise_model.square) == (
        None,
        None,
        None,
    ):
        raise ValueError("noise_model must have at least one non-None entry.")

    sigmas = np.zeros_like(fvals)

    if noise_model.intercept is not None:
        sigmas += noise_model.intercept

    if noise_model.slope is not None:
        sigmas += noise_model.slope * fvals

    if noise_model.square is not None:
        sigmas += noise_model.square * fvals**2

    return sigmas
