"""Tests of the command line adapter the docker image runs (container_entrypoint.py)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import fake_data

# The entrypoint is a script at the root of the repository, not part of the package
_spec = importlib.util.spec_from_file_location(
    "container_entrypoint", Path(__file__).parents[1] / "container_entrypoint.py"
)
entrypoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entrypoint)


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["container_entrypoint.py", *map(str, args)])
    entrypoint.main()


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Start in a known directory (restored afterwards, `main` changes it)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    return tmp_path


def test_fit_from_fits_files(workdir, dataset, monkeypatch):
    config = fake_data.write_fits_inputs(workdir / "input", dataset)
    # relative paths, as typed on a command line
    run(monkeypatch, "--config", "input/config.json", "--output-dir", "output/run1")

    output = workdir / "output" / "run1"
    image_files = [
        f"{stem}_{kind}.fits"
        for stem in ("HST_F814W", "ground_r", "GALEX_FUV")  # names made safe
        for kind in ("model", "residual")
    ]
    assert sorted(p.name for p in output.iterdir()) == sorted(
        ["config.used.json", "results.json", *image_files]
    )
    assert json.loads((output / "config.used.json").read_text()) == config
    results = json.loads((output / "results.json").read_text())
    assert "success" in results["message"]
    assert results["chi2_ndf"] == pytest.approx(1, abs=0.03)
    galaxy = results["models"]["galaxy"]
    for key, truth in fake_data.STRUCTURE[0].items():
        assert galaxy["shared"][key]["value"] == pytest.approx(truth, rel=0.1)
    assert list(galaxy["per_image"]) == dataset["cutout_name"]  # names as given

    # the model and residual images, where results.json says they are
    assert list(results["images"]) == dataset["cutout_name"]  # names as given
    chi2 = 0.0
    for name, cutout, variance, wcs in zip(
        dataset["cutout_name"], dataset["cutout"], dataset["variance"], dataset["wcs"]
    ):
        files = results["images"][name]
        assert set(files) == {"model", "residual"}
        model, model_header = fits.getdata(output / files["model"], header=True)
        residual, residual_header = fits.getdata(output / files["residual"], header=True)
        assert model.shape == residual.shape == cutout.shape
        assert (model_header["IMGTYPE"], residual_header["IMGTYPE"]) == ("model", "residual")
        assert model_header["CUTOUT"] == name
        # (the inputs were read from float32 files)
        np.testing.assert_allclose(
            residual, (cutout - model) / np.sqrt(variance), rtol=1e-4, atol=1e-4
        )
        chi2 += np.sum(residual**2)
        # on the pixel grid of the cutout
        for header in (model_header, residual_header):
            np.testing.assert_allclose(
                WCS(header).pixel_to_world_values(3.0, 7.0), wcs.pixel_to_world_values(3.0, 7.0)
            )
    pixels = sum(c.size for c in dataset["cutout"])
    assert chi2 / (pixels - 14) == pytest.approx(results["chi2_ndf"], rel=1e-6)

    # nothing is written beside the inputs or where the tool was started
    assert not list(workdir.glob("*.log")) and not list((workdir / "input").glob("*.log"))

    # an output directory is never reused
    monkeypatch.chdir(workdir)  # (the first run moved to a temporary directory)
    with pytest.raises(FileExistsError):
        run(monkeypatch, "--config", "input/config.json", "--output-dir", "output/run1")


def test_prepare_inputs(workdir, dataset):
    config = fake_data.write_fits_inputs(workdir / "input", dataset)
    kwargs = entrypoint.prepare_inputs(config, workdir / "input")
    assert kwargs["cutout_name"] == dataset["cutout_name"]
    for key in ("cutout", "variance", "psf"):
        for array, expected in zip(kwargs[key], dataset[key]):
            assert array.dtype == np.float64 and array.dtype.isnative
            np.testing.assert_allclose(array, expected, rtol=1e-6)
    for mask, expected in zip(kwargs["mask"], dataset["mask"]):
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, expected)
    for wcs, expected in zip(kwargs["wcs"], dataset["wcs"]):
        np.testing.assert_allclose(
            wcs.pixel_to_world_values(3.0, 7.0), expected.pixel_to_world_values(3.0, 7.0)
        )
    np.testing.assert_array_equal(kwargs["init_pos"], dataset["positions"])
    assert kwargs["model_types"] == fake_data.MODEL_TYPES
    assert kwargs["init_params"] == [{}, {}]


def test_bad_inputs_are_refused(workdir, dataset):
    config = fake_data.write_fits_inputs(workdir / "input", dataset)
    first = config["cutouts"][0]

    # names which collide once AstroPhot has made them safe
    clash = json.loads(json.dumps(config))
    clash["cutouts"][0]["name"], clash["cutouts"][1]["name"] = "r-band", "r_band"
    with pytest.raises(ValueError, match="unique"):
        entrypoint.prepare_inputs(clash, workdir / "input")

    # a variance map of the wrong shape (here: the PSF file)
    wrong = json.loads(json.dumps(config))
    wrong["cutouts"][0]["variance"] = first["psf"]
    with pytest.raises(ValueError, match="variance shape"):
        entrypoint.prepare_inputs(wrong, workdir / "input")

    # an image whose header has no sky coordinates (here: the variance extension)
    no_wcs = json.loads(json.dumps(config))
    no_wcs["cutouts"][0]["image"] = first["variance"]
    with pytest.raises(ValueError, match="No RA/Dec WCS"):
        entrypoint.prepare_inputs(no_wcs, workdir / "input")


def test_finite_or_none():
    results = {
        "chi2_ndf": float("nan"),
        "message": "fail",
        "models": {"a": {"value": [1.0, float("inf")], "uncertainty": None, "n": 3}},
    }
    cleaned, replaced = entrypoint.finite_or_none(results)
    assert replaced == 2
    assert cleaned == {
        "chi2_ndf": None,
        "message": "fail",
        "models": {"a": {"value": [1.0, None], "uncertainty": None, "n": 3}},
    }
    json.dumps(cleaned, allow_nan=False)  # now valid JSON
