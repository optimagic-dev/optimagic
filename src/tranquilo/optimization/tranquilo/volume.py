"""Functions to calculate volumes of hyperspheres and hypercubes.

Hypercubes can be seen as hyperspheres when the distance from the center is calculated
in an infinity norm rather than a euclidean norm.

This is why we caracterize hypercubes by their radius (half the side length).

"""
import numpy as np
from scipy.special import gamma, loggamma


def get_radius_after_volume_scaling(radius, dim, scaling_factor):
    out = radius * scaling_factor ** (1 / dim)
    return out


def get_radius_of_sphere_with_volume_of_cube(cube_radius, dim, scaling_factor=None):
    log_radius = (
        loggamma(dim / 2 + 1) / dim
        - np.log(np.pi) / 2
        + np.log(2)
        + np.log(cube_radius)
    )
    if scaling_factor is not None:
        log_radius += np.log(scaling_factor) / dim
    out = np.exp(log_radius)
    return out


def get_radius_of_cube_with_volume_of_sphere(sphere_radius, dim, scaling_factor=None):
    log_radius = (
        np.log(np.pi) / 2
        + np.log(sphere_radius)
        - np.log(2)
        - loggamma(dim / 2 + 1) / dim
    )
    if scaling_factor is not None:
        log_radius += np.log(scaling_factor) / dim
    out = np.exp(log_radius)
    return out


def get_volume(radius, dim, shape):
    if shape == "sphere":
        out = _sphere_volume(radius, dim)
    elif shape == "cube":
        out = _cube_volume(radius, dim)
    else:
        raise ValueError(f"shape must be 'shpere' or 'cube', not: {shape}")
    return out


def get_radius(volume, dim, shape):
    if shape == "sphere":
        out = _sphere_radius(volume, dim)
    elif shape == "cube":
        out = _cube_radius(volume, dim)
    else:
        raise ValueError(f"shape must be 'shpere' or 'cube', not: {shape}")
    return out


def _sphere_volume(radius, dim):
    vol = np.pi ** (dim / 2) * radius**dim / gamma(dim / 2 + 1)
    return vol


def _cube_volume(radius, dim):
    vol = (radius * 2) ** dim
    return vol


def _sphere_radius(volume, dim):
    radius = ((volume * gamma(dim / 2 + 1)) / (np.pi ** (dim / 2))) ** (1 / dim)
    return radius


def _cube_radius(volume, dim):
    radius = 0.5 * volume ** (1 / dim)
    return radius
