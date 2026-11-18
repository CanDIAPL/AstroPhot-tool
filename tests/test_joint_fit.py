"""End to end test: fit fake cutouts spanning a factor of 60 in pixel scale."""

import json

import numpy as np
import pytest

from AstroPhot_tool import joint_multi_image
import fake_data


@pytest.fixture(scope="module")
def fit(dataset):
    """Both outputs of `joint_multi_image`: the results and the diagnostic images."""
    rng = np.random.default_rng(1)
    # Start 0.1 arcsec away from the true positions, everything else automatic
    init_pos = dataset["positions"] + rng.normal(size=(2, 2)) * 0.1 / 3600
    return joint_multi_image(
        cutout_name=dataset["cutout_name"],
        cutout=dataset["cutout"],
        variance=dataset["variance"],
        mask=dataset["mask"],
        psf=dataset["psf"],
        wcs=dataset["wcs"],
        model_names=fake_data.MODEL_NAMES,
        model_types=fake_data.MODEL_TYPES,
        init_pos=init_pos,
        init_params=[{}, {}],
    )


@pytest.fixture(scope="module")
def result(fit):
    return fit[0]


@pytest.fixture(scope="module")
def diagnostics(fit):
    return fit[1]


def test_fake_dataset(dataset):
    """The fake data has the requested layout and the sources are well detected."""
    for name, cutout, variance, clean in zip(
        dataset["cutout_name"], dataset["cutout"], dataset["variance"], dataset["clean"]
    ):
        shape = fake_data.IMAGES[name]["shape"]
        assert cutout.shape == variance.shape == clean.shape == shape
        assert np.all(variance > 0)
        # The noise is drawn from the variance map
        assert np.std((cutout - clean) / np.sqrt(variance)) == pytest.approx(1, abs=0.05)
        # Clearly detected, but far from noiseless
        assert 10 < np.max(clean / np.sqrt(variance)) < 200


def test_fit_converged(result):
    assert "success" in result["message"]
    # ~64000 pixels, so chi2/ndf should be within ~0.006 of 1
    assert result["chi2_ndf"] == pytest.approx(1, abs=0.03)
    assert set(result["models"]) == set(fake_data.MODEL_NAMES)
    json.dumps(result)  # plain python types only


def _check(entry, truth, rtol, nsigma=5):
    value, sigma = np.array(entry["value"]), np.array(entry["uncertainty"])
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)
    assert np.all(np.abs(value - truth) < nsigma * sigma), (value, truth, sigma)
    assert value == pytest.approx(truth, rel=rtol)


def test_fit_positions(result, dataset):
    cos_dec = np.cos(np.deg2rad(fake_data.DEC0))
    for name, (ra, dec) in zip(fake_data.MODEL_NAMES, dataset["positions"]):
        model = result["models"][name]
        offset = np.hypot((model["ra"] - ra) * cos_dec, model["dec"] - dec) * 3600
        assert offset < 0.02, f"{name} is {offset} arcsec from the truth"
        sigma = np.array(model["shared"]["center"]["uncertainty"])
        assert np.all(sigma > 0) and np.all(sigma < 0.02)


def test_fit_shared_parameters(result):
    """One set of structural parameters, recovered from all images together."""
    shared = result["models"]["galaxy"]["shared"]
    assert set(shared) == {"center", "q", "PA", "n", "Re"}
    for key, truth in fake_data.STRUCTURE[0].items():
        _check(shared[key], truth, rtol=0.1)
    assert set(result["models"]["star"]["shared"]) == {"center"}


def test_fit_fluxes(result):
    """An independent flux for every model in every image."""
    for name, fluxes in zip(fake_data.MODEL_NAMES, fake_data.FLUXES):
        per_image = result["models"][name]["per_image"]
        assert list(per_image) == list(fake_data.IMAGES)
        for key, truths in fluxes.items():
            for image_name, truth in truths.items():
                # The blended far UV fluxes are the least certain
                rtol = 0.5 if image_name == "GALEX-FUV" else 0.1
                _check(per_image[image_name][key], truth, rtol=rtol)


def test_fit_images(result, diagnostics, dataset):
    """A model and a noise scaled residual image on the pixel grid of every cutout."""
    assert list(diagnostics) == dataset["cutout_name"]  # names as given
    chi2, pixels = 0.0, 0
    for name, cutout, variance, clean in zip(
        dataset["cutout_name"], dataset["cutout"], dataset["variance"], dataset["clean"]
    ):
        model, residual = diagnostics[name]["model"], diagnostics[name]["residual"]
        assert set(diagnostics[name]) == {"model", "residual"}
        assert model.shape == residual.shape == cutout.shape
        np.testing.assert_allclose(residual, (cutout - model) / np.sqrt(variance), rtol=1e-9)
        # The fitted model is the noiseless scene: the same total light, and
        # the same image to within the noise of a pixel or so
        assert np.sum(model) == pytest.approx(np.sum(clean), rel=0.02)
        assert np.max(np.abs(model - clean) / np.sqrt(variance)) < 3
        # Nothing left but noise
        assert np.std(residual) == pytest.approx(1, abs=0.05)
        chi2 += np.sum(residual**2)
        pixels += residual.size
    # The residual images are the chi squared of the fit (14 parameters, ~64000 pixels)
    assert chi2 / (pixels - 14) == pytest.approx(result["chi2_ndf"], rel=1e-6)
