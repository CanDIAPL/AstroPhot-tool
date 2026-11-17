"""Fixtures shared by all tests, the images themselves are defined in `fake_data`."""

import pytest

import fake_data


@pytest.fixture
def images():
    """
    Fresh `build_target` inputs for blank versions of the three cutouts in
    `fake_data.IMAGES` (unit variance, nothing masked). Safe to modify.
    """
    return fake_data.image_inputs()


@pytest.fixture(scope="session")
def dataset():
    """Noisy fake cutouts of the `fake_data` scene, see `fake_data.fake_dataset`."""
    return fake_data.fake_dataset(seed=42)
