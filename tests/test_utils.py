"""Unit tests for AstroPhot_tool.utils."""

import json

import astrophot as ap
import numpy as np
import pytest
import torch
from astropy.wcs import WCS
from caskade import GraphError

from AstroPhot_tool import utils
from fake_data import (
    DEC0,
    IMAGES,
    MODEL_NAMES,
    MODEL_TYPES,
    OFFSETS,
    POSITIONS,
    RA0,
    STRUCTURE,
    image_inputs,
    make_wcs,
    offset_to_radec,
)

# The cutouts come from the `images` fixture (conftest.py), which holds blank
# versions of the three images defined in `fake_data.IMAGES`.
PIXELSCALES = [spec["pixelscale"] for spec in IMAGES.values()]
NO_PSF = np.ones((1, 1))  # a single pixel PSF does not blur the image


def to_tensor(*values):
    return tuple(torch.as_tensor(np.asarray(v, dtype=np.float64)) for v in values)


def centroid(image):
    """Flux weighted (x, y) centroid of a numpy image."""
    y, x = np.indices(image.shape)
    return np.sum(x * image) / np.sum(image), np.sum(y * image) / np.sum(image)


###############################################################################
# safe_name
###############################################################################
@pytest.mark.parametrize(
    "name, expected",
    [
        ("r", "r"),
        ("HST_F814W", "HST_F814W"),
        ("GALEX-FUV", "GALEX_FUV"),
        ("HST/ACS F814W", "HST_ACS_F814W"),
        ("2MASS_J", "_2MASS_J"),
        ("lambda", "_lambda"),
    ],
)
def test_safe_name(name, expected):
    assert utils.safe_name(name) == expected
    ap.TargetImage(name=utils.safe_name(name), data=np.zeros((3, 3)))  # accepted by AstroPhot


###############################################################################
# world_to_plane
###############################################################################
@pytest.mark.parametrize("dec0", [0.0, 2.2345, -45.0, 89.5])
def test_world_to_plane_matches_astropy(dec0):
    # A TAN WCS with 1 arcsec pixels, the tangent point at pixel (0, 0), and
    # x increasing to the East has pixel coordinates equal to plane coordinates
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [RA0, dec0]
    wcs.wcs.crpix = [1, 1]
    wcs.wcs.cd = np.eye(2) / 3600
    x = np.array([0.0, 30.0, -200.0, 5.0, -600.0])
    y = np.array([0.0, -10.0, 150.0, 400.0, -300.0])
    ra, dec = wcs.pixel_to_world_values(x, y)
    px, py = utils.world_to_plane(ra, dec, RA0, dec0)
    np.testing.assert_allclose(px, x, atol=1e-7)
    np.testing.assert_allclose(py, y, atol=1e-7)
    # and agrees with AstroPhot's own projection
    ax, ay = ap.image.func.world_to_plane_gnomonic(*to_tensor(ra, dec, RA0, dec0))
    np.testing.assert_allclose(px, ax.numpy(), atol=1e-7)
    np.testing.assert_allclose(py, ay.numpy(), atol=1e-7)


###############################################################################
# normalize_wcs
###############################################################################
def test_normalize_wcs_same_reference():
    """Keeping the original reference point reproduces the original WCS."""
    wcs = make_wcs(RA0, DEC0, 0.8, (40, 60), rotation=0.4)
    out = utils.normalize_wcs(wcs, np.array([RA0, DEC0]))
    assert set(out) == {"crval", "crtan", "crpix", "CD"}
    np.testing.assert_allclose(out["crval"], [RA0, DEC0])
    np.testing.assert_allclose(out["crtan"], [0, 0])
    # 0-indexed (x, y), the center of a W=60, H=40 image
    np.testing.assert_allclose(out["crpix"], [29.5, 19.5], atol=1e-6)
    np.testing.assert_allclose(out["CD"], wcs.wcs.cd * 3600, rtol=1e-7, atol=1e-10)


def test_normalize_wcs_moves_reference_pixel():
    wcs = make_wcs(RA0, DEC0, 0.8, (40, 60))
    new_crval = np.array(wcs.pixel_to_world_values(10.0, 30.0))
    out = utils.normalize_wcs(wcs, new_crval)
    np.testing.assert_allclose(out["crpix"], [10.0, 30.0], atol=1e-6)
    np.testing.assert_allclose(out["CD"], [[-0.8, 0], [0, 0.8]], atol=1e-5)


@pytest.mark.parametrize("dec0", [0.0, 2.2345, -60.0, 89.0, -89.9])
@pytest.mark.parametrize(
    "pixelscale, shape", [(0.05, (400, 300)), (0.8, (150, 200)), (3.0, (100, 100))]
)
@pytest.mark.parametrize("rotation, flip_ra", [(0.0, True), (0.7, True), (-2.0, False)])
@pytest.mark.parametrize("offset", [(0.0, 0.0), (20.0, -15.0)])
def test_normalize_wcs_matches_astropy(dec0, pixelscale, shape, rotation, flip_ra, offset):
    """An AstroPhot image built from the output maps sky to pixels like astropy."""
    wcs = make_wcs(RA0, dec0, pixelscale, shape, rotation, flip_ra)
    new_crval = offset_to_radec(offset, dec0=dec0)  # tangent point away from the cutout center
    image = ap.TargetImage(data=np.zeros(shape), **utils.normalize_wcs(wcs, new_crval))

    H, W = shape
    x, y = np.meshgrid(np.linspace(0, W - 1, 5), np.linspace(0, H - 1, 5))
    ra, dec = wcs.pixel_to_world_values(x, y)
    i, j = image.world_to_pixel(*to_tensor(ra, dec))
    # AstroPhot index i runs along x (FITS axis 1) and j along y
    error = np.hypot(i.numpy() - x, j.numpy() - y)
    assert error.max() < 2e-3, "more than 0.002 pixels from astropy"
    # exactly the same tangent plane when the tangent point is unchanged
    if offset == (0.0, 0.0):
        assert error.max() < 1e-5
    # and back again
    ra2, dec2 = image.pixel_to_world(*to_tensor(x, y))
    sep = np.hypot((ra2.numpy() - ra) * np.cos(np.deg2rad(dec)), dec2.numpy() - dec) * 3600
    assert sep.max() < 2e-3 * pixelscale

    assert image.pixelscale.item() == pytest.approx(pixelscale, rel=1e-4)
    assert image.flip_ra_axis == flip_ra


def test_normalize_wcs_ra_wrap():
    """Cutouts straddling RA = 0 are handled."""
    wcs = make_wcs(359.9999, 30.0, 0.8, (100, 100), rotation=0.2)
    new_crval = np.array([0.0002, 30.0003])  # other side of the wrap
    image = ap.TargetImage(data=np.zeros((100, 100)), **utils.normalize_wcs(wcs, new_crval))
    x, y = np.meshgrid(np.linspace(0, 99, 5), np.linspace(0, 99, 5))
    ra, dec = wcs.pixel_to_world_values(x, y)
    assert ra.min() < 1 and ra.max() > 359  # the test does straddle the wrap
    i, j = image.world_to_pixel(*to_tensor(ra, dec))
    assert np.hypot(i.numpy() - x, j.numpy() - y).max() < 2e-3


###############################################################################
# build_target
###############################################################################
@pytest.fixture
def target(images):
    return utils.build_target(**images)


def test_build_target_structure(images, target):
    assert isinstance(target, ap.TargetImageList)
    assert [image.name for image in target.images] == ["HST_F814W", "ground_r", "GALEX_FUV"]
    for image, cutout, scale in zip(target.images, images["cutout"], PIXELSCALES):
        assert tuple(image.data.shape) == cutout.shape
        assert image.pixelscale.item() == pytest.approx(scale, rel=1e-4)
        assert image.has_psf
        assert image.psf.data.sum().item() == pytest.approx(1)


def test_build_target_common_tangent_plane(images, target):
    # The tangent point is the center of the first cutout
    H, W = images["cutout"][0].shape
    center = images["wcs"][0].pixel_to_world_values((W - 1) / 2, (H - 1) / 2)
    np.testing.assert_allclose(center, [RA0, DEC0], atol=1e-9)
    for image in target.images:
        np.testing.assert_allclose(image.crval.value.numpy(), center, atol=1e-12)
        np.testing.assert_allclose(image.crtan.value.numpy(), [0, 0])

    # A point on the sky has the same tangent plane coordinates in every image,
    # going through each cutout's own pixel grid
    for offset in [(0.0, 0.0), (1.3, -2.1), (-4.0, 3.0)]:
        ra, dec = offset_to_radec(offset)
        expected = utils.world_to_plane(ra, dec, RA0, DEC0)
        np.testing.assert_allclose(expected, offset, atol=1e-4)
        for image, wcs in zip(target.images, images["wcs"]):
            x, y = wcs.world_to_pixel_values(ra, dec)
            plane = image.pixel_to_plane(*to_tensor(x, y))
            np.testing.assert_allclose([p.item() for p in plane], expected, atol=1e-4)


def test_build_target_pixel_orientation(images):
    """Pixel [row, column] of the input array is where the WCS says it is."""
    rng = np.random.default_rng(0)
    marks = [(100, 20), (5, 40), (22, 3)]  # (row, column), well off the diagonal
    for cutout, variance, mask, (row, col) in zip(
        images["cutout"], images["variance"], images["mask"], marks
    ):
        cutout[:] = rng.normal(size=cutout.shape)
        cutout[row, col] = 100.0
        variance[:] = rng.uniform(1, 2, size=cutout.shape)
        mask[row, col + 1] = True
    target = utils.build_target(**images)
    for image, cutout, variance, mask, wcs, (row, col) in zip(
        target.images, images["cutout"], images["variance"], images["mask"], images["wcs"], marks
    ):
        np.testing.assert_array_equal(image.data.numpy(), cutout)
        np.testing.assert_allclose(image.variance.numpy(), variance)
        np.testing.assert_array_equal(image.mask.numpy(), mask)
        ra, dec = wcs.pixel_to_world_values(col, row)
        i, j = image.world_to_pixel(*to_tensor(ra, dec))
        np.testing.assert_allclose([i.item(), j.item()], [col, row], atol=1e-3)
        assert image.data[round(j.item()), round(i.item())].item() == 100.0


def test_build_target_psf_normalized(images):
    images["psf"] = [p * scale for p, scale in zip(images["psf"], [7.0, 1.0, 1e-3])]
    target = utils.build_target(**images)
    for image, psf in zip(target.images, images["psf"]):
        np.testing.assert_allclose(image.psf.data.numpy(), psf / psf.sum())


def test_build_target_bad_variance_is_masked(images):
    bad = {(3, 4): 0.0, (5, 6): np.nan, (7, 8): -1.0, (9, 10): np.inf}
    for pixel, value in bad.items():
        images["variance"][0][pixel] = value
    images["mask"][0][0, 0] = True
    images["variance"][1][2, 3] = np.nan
    given = [v.copy() for v in images["variance"]]
    target = utils.build_target(**images)
    for variance, original in zip(images["variance"], given):
        np.testing.assert_array_equal(variance, original)  # the caller's arrays are untouched
    mask = target.images[0].mask.numpy()
    assert mask.sum() == 5
    assert mask[0, 0] and all(mask[pixel] for pixel in bad)
    assert target.images[1].mask.numpy().sum() == 1 and target.images[1].mask[2, 3]
    assert not target.images[2].mask.any()
    for image in target.images:
        assert np.all(np.isfinite(image.weight.numpy()))


def test_build_target_names(images):
    images["cutout_name"] = ["HST/ACS F814W", "r", "GALEX-FUV"]
    target = utils.build_target(**images)
    assert [image.name for image in target.images] == ["HST_ACS_F814W", "r", "GALEX_FUV"]
    # names which are the same once made safe are refused by AstroPhot
    images["cutout_name"] = ["a-b", "a_b", "c"]
    with pytest.raises(GraphError):
        utils.build_target(**images)


@pytest.mark.parametrize("key", ["cutout_name", "variance", "mask", "psf", "wcs"])
def test_build_target_length_mismatch(images, key):
    images[key] = images[key][:2]
    with pytest.raises(ValueError):
        utils.build_target(**images)


###############################################################################
# build_model
###############################################################################
def two_model_scene(target, init_pos=POSITIONS):
    """The galaxy and star of `fake_data`, with unit galaxy brightness."""
    return utils.build_model(
        target,
        MODEL_NAMES,
        MODEL_TYPES,
        init_pos,
        [dict(STRUCTURE[0], Ie=1.0), dict(flux=5.0)],
    )


def test_build_model_structure(target):
    model = two_model_scene(target)
    assert model.name == "full_scene"
    assert model.target is target
    assert [m.name for m in model.models] == MODEL_NAMES
    for group, model_type in zip(model.models, MODEL_TYPES):
        assert group.target is target
        assert [m.name for m in group.models] == [
            f"{group.name}_{image.name}" for image in target.images
        ]
        for sub_model, image in zip(group.models, target.images):
            assert sub_model.model_type == model_type
            assert sub_model.target is image


def test_build_model_parameter_sharing(target):
    model = two_model_scene(target)
    galaxy, star = model.models
    for key in ["center", "q", "PA", "n", "Re"]:
        assert galaxy.models[0][key].dynamic
        for sub_model in galaxy.models[1:]:
            assert sub_model[key].pointer
            np.testing.assert_array_equal(
                sub_model[key].value.numpy(), galaxy.models[0][key].value.numpy()
            )
    for sub_model in galaxy.models:
        assert sub_model.Ie.dynamic
    for sub_model in star.models:
        assert sub_model.flux.dynamic
    assert all(m.center.pointer for m in star.models[1:])
    # The WCS of the images is not part of the fit
    for image in target.images:
        assert image.crval.static and image.crtan.static and image.CD.static
    # galaxy: center(2) q PA n Re + 3 Ie, star: center(2) + 3 flux
    assert len(model.build_params_array_names()) == 9 + 5

    # Changing a shared parameter changes every image, a flux only its own
    before = [d.numpy().copy() for d in galaxy().data]
    galaxy.models[0].q.value = 0.3
    after = [d.numpy().copy() for d in galaxy().data]
    assert all(not np.allclose(b, a) for b, a in zip(before, after))
    galaxy.models[1].Ie.value = 3.0
    final = [d.numpy().copy() for d in galaxy().data]
    # (the PSF convolution leaves round-off at ~1e-16 of the peak)
    np.testing.assert_allclose(final[0], after[0], atol=1e-12 * after[0].max())
    np.testing.assert_allclose(final[1], 3 * after[1], atol=1e-12 * after[1].max())
    np.testing.assert_allclose(final[2], after[2], atol=1e-12 * after[2].max())


def test_build_model_position(images, target):
    """Models land on the requested RA Dec in every image."""
    model = two_model_scene(target)
    for group, offset in zip(model.models, OFFSETS):
        np.testing.assert_allclose(group.models[0].center.value.numpy(), offset, atol=1e-4)
    # The star is a PSF at the star position, compare the centroid with astropy
    star = model.models[1]
    for sub_model, wcs, psf in zip(star.models, images["wcs"], images["psf"]):
        image = sub_model().data.numpy()
        expected = wcs.world_to_pixel_values(*POSITIONS[1])
        np.testing.assert_allclose(centroid(image), expected, atol=0.02)
        assert image.sum() == pytest.approx(5.0, rel=1e-4)  # flux is conserved
        assert image.max() <= 5.0 * psf.max() * (1 + 1e-6)


def test_build_model_position_precision(target):
    """Positions given as python floats are not truncated to float32."""
    reference = two_model_scene(target)
    model = two_model_scene(utils.build_target(**image_inputs()), init_pos=POSITIONS.tolist())
    for group, ref_group in zip(model.models, reference.models):
        np.testing.assert_allclose(
            group.models[0].center.value.numpy(),
            ref_group.models[0].center.value.numpy(),
            atol=1e-7,
        )
        assert group.models[0].center.value.dtype == torch.float64


def test_build_model_same_scene_in_every_image():
    """The same sky is drawn at every resolution: flux, position and shape agree."""
    # Measuring the shape of an object needs more room than the cutouts of
    # `fake_data.IMAGES` have, so this test has its own: 60 arcsec across, no
    # PSF, at three pixel scales and orientations (the last one also mirrored,
    # East to the right).
    wide = {
        "fine": dict(pixelscale=0.25, shape=(240, 240), rotation=0.3),
        "medium": dict(pixelscale=0.8, shape=(75, 75)),
        "coarse": dict(pixelscale=3.0, shape=(21, 21), rotation=-0.6, flip_ra=False),
    }
    inputs = image_inputs(specs=wide)
    target = utils.build_target(**inputs)
    sigma, q, PA, flux = 4.0, 0.5, 0.7, 10.0
    position = offset_to_radec((1.0, -2.0))
    model = utils.build_model(
        target,
        ["blob"],
        ["gaussian galaxy model"],
        position[None],
        [dict(sigma=sigma, q=q, PA=PA, flux=flux)],
    )
    totals, angles = [], []
    for image, w in zip(model().data, inputs["wcs"]):
        image = image.numpy()
        totals.append(image.sum())
        np.testing.assert_allclose(centroid(image), w.world_to_pixel_values(*position), atol=0.02)
        # Second moments on the sky: the same ellipse whatever the pixel grid
        y, x = np.indices(image.shape)
        ra, dec = w.pixel_to_world_values(x, y)
        east, north = utils.world_to_plane(ra, dec, *position)
        m_ee = np.sum(image * east**2) / image.sum()
        m_nn = np.sum(image * north**2) / image.sum()
        m_en = np.sum(image * east * north) / image.sum()
        major = 0.5 * (m_ee + m_nn) + np.hypot(0.5 * (m_ee - m_nn), m_en)
        minor = 0.5 * (m_ee + m_nn) - np.hypot(0.5 * (m_ee - m_nn), m_en)
        # pixel integration adds pixelscale^2 / 12 to each moment
        pix2 = np.abs(np.linalg.det(w.wcs.cd)) * 3600**2 / 12
        assert major - pix2 == pytest.approx(sigma**2, rel=0.02)
        assert minor - pix2 == pytest.approx((q * sigma) ** 2, rel=0.02)
        # Angle of the major axis from North towards East on the sky
        angles.append(0.5 * np.arctan2(2 * m_en, m_nn - m_ee))
    # The same orientation on the sky in every image. AstroPhot measures PA
    # counter-clockwise from +y on a tangent plane where +x is East, so the
    # major axis is at PA from North towards West.
    np.testing.assert_allclose(angles, -PA, atol=0.01)
    # The same total flux whatever the pixel size (a 144x range in pixel area).
    # The AstroPhot "flux" of a gaussian is the total flux only for q = 1.
    np.testing.assert_allclose(totals, flux * q, rtol=2e-3)


def test_build_model_psf_convolution(images, target):
    """Each image is convolved with its own PSF."""
    kwargs = dict(
        model_names=["blob"],
        model_types=["gaussian galaxy model"],
        init_pos=offset_to_radec((0.0, 0.0))[None],
        init_params=[dict(sigma=0.3, q=1.0, PA=0.0, flux=1.0)],
    )
    convolved = utils.build_model(target, **kwargs)().data
    no_psf = utils.build_target(**dict(images, psf=[NO_PSF] * 3))
    sharp = utils.build_model(no_psf, **kwargs)().data
    for image, raw, psf in zip(convolved[:2], sharp[:2], images["psf"][:2]):
        image, raw = image.numpy(), raw.numpy()
        assert image.max() < raw.max()
        assert image.sum() == pytest.approx(raw.sum(), rel=1e-3)
        # Gaussian (x) Gaussian: variances add (measured along x in pixels)
        y, x = np.indices(image.shape)
        var = lambda im: np.sum(im * (x - centroid(im)[0]) ** 2) / im.sum()
        yp, xp = np.indices(psf.shape)
        psf_var = np.sum(psf * (xp - psf.shape[1] // 2) ** 2)
        assert var(image) == pytest.approx(var(raw) + psf_var, rel=0.02)


def test_build_model_sky_is_independent(target):
    model = utils.build_model(
        target, ["sky"], ["plane sky model"], POSITIONS[:1], [dict(I0=1.0, delta=[0.0, 0.0])]
    )
    for sub_model in model.models[0].models:
        assert sub_model.I0.dynamic and sub_model.delta.dynamic


def test_build_model_names_and_lengths(target):
    model = utils.build_model(
        target, ["NGC 1234"], ["point model"], POSITIONS[:1], [dict(flux=1.0)]
    )
    assert model.models[0].name == "NGC_1234"
    # two models of the same name are refused by AstroPhot
    with pytest.raises(GraphError):
        utils.build_model(target, ["a", "a"], ["point model"] * 2, POSITIONS, [{}, {}])
    with pytest.raises(ValueError):
        utils.build_model(target, ["a", "b"], ["point model"], POSITIONS, [{}, {}])


###############################################################################
# initialize_fluxes
###############################################################################
GALAXY_IE, STAR_FLUX = [2.0, 800.0, 0.03], [30.0, 6000.0, 0.1]  # one per image


def scene_with_fluxes(target, galaxy_Ie=GALAXY_IE, star_flux=STAR_FLUX):
    model = two_model_scene(target)
    for sub_model, value in zip(model.models[0].models, galaxy_Ie):
        sub_model.Ie.value = value
    for sub_model, value in zip(model.models[1].models, star_flux):
        sub_model.flux.value = value
    return model


def target_with_data(data):
    # (tiny variance: every model is detected at high significance)
    return utils.build_target(**image_inputs(data, variance=1e-12))


def test_initialize_fluxes_recovers_linear_solution(target):
    data = [d.numpy() for d in scene_with_fluxes(target)().data]
    # Start from fluxes which are wrong by orders of magnitude
    model = scene_with_fluxes(target_with_data(data), [1e3, 1e-2, 5.0], [1e-3, 1e7, 40.0])
    utils.initialize_fluxes(model)
    for sub_model, value in zip(model.models[0].models, GALAXY_IE):
        assert sub_model.Ie.value.item() == pytest.approx(value, rel=1e-6)
    for sub_model, value in zip(model.models[1].models, STAR_FLUX):
        assert sub_model.flux.value.item() == pytest.approx(value, rel=1e-6)
    # structural parameters are untouched
    assert model.models[0].models[0].Re.value.item() == STRUCTURE[0]["Re"]


def test_initialize_fluxes_respects_mask_and_missing_model(target):
    truth = scene_with_fluxes(target, star_flux=[1e-30] * 3)  # no star in the data
    data = [d.numpy().copy() for d in truth().data]
    data[1][10:14, 20:24] = 1e6  # an artifact
    inputs = image_inputs(data, variance=1e-12)
    inputs["mask"][1][10:14, 20:24] = True
    model = two_model_scene(utils.build_target(**inputs))
    utils.initialize_fluxes(model)
    for sub_model, value in zip(model.models[0].models, GALAXY_IE):
        assert sub_model.Ie.value.item() == pytest.approx(value, rel=1e-3)
    for sub_model, psf in zip(model.models[1].models, inputs["psf"]):
        flux = sub_model.flux.value.item()
        # undetected: set to the 1 sigma uncertainty of a PSF flux
        assert flux == pytest.approx(1e-6 / np.sqrt(np.sum(psf**2)), rel=0.3)


###############################################################################
# fit_results
###############################################################################
def test_fit_results_layout(target):
    # variance 1e-12 and a tiny offset from the true model
    data = [d.numpy() + 1e-9 for d in scene_with_fluxes(target)().data]
    model = scene_with_fluxes(target_with_data(data))
    fitter = ap.fit.LM(model, verbose=0, max_iter=2).fit()
    cutout_names = ["HST/ACS F814W", "r", "GALEX-FUV"]  # as the user named them
    results = utils.fit_results(model, fitter, cutout_names, ["my galaxy", "star"])

    assert json.loads(json.dumps(results)) == results  # plain types only
    assert set(results) == {"message", "chi2_ndf", "iterations", "crval", "models"}
    np.testing.assert_allclose(results["crval"], [RA0, DEC0], atol=1e-9)
    assert 0 < results["chi2_ndf"] < 1.1  # the 1e-9 offset is 1e-3 sigma per pixel
    assert list(results["models"]) == ["my galaxy", "star"]

    galaxy = results["models"]["my galaxy"]
    assert galaxy["model_type"] == "sersic galaxy model"
    np.testing.assert_allclose([galaxy["ra"], galaxy["dec"]], POSITIONS[0], atol=1e-6)
    assert set(galaxy["shared"]) == {"center", "q", "PA", "n", "Re"}
    assert list(galaxy["per_image"]) == cutout_names
    for name, Ie in zip(cutout_names, GALAXY_IE):
        entry = galaxy["per_image"][name]["Ie"]
        assert set(entry) == {"value", "uncertainty", "units"}
        assert entry["value"] == pytest.approx(Ie, rel=0.05)
        assert entry["uncertainty"] > 0
        assert entry["units"] == "flux/arcsec^2"
    assert galaxy["shared"]["Re"]["units"] == "arcsec"
    assert len(galaxy["shared"]["center"]["value"]) == 2

    star = results["models"]["star"]
    np.testing.assert_allclose([star["ra"], star["dec"]], POSITIONS[1], atol=1e-6)
    assert set(star["shared"]) == {"center"}
    assert set(star["per_image"]["r"]) == {"flux"}


###############################################################################
# fit_images
###############################################################################
def test_fit_images(target):
    truth = [d.numpy() for d in scene_with_fluxes(target)().data]
    # Data which is 3 sigma above the true model everywhere, with one bad region
    inputs = image_inputs([t + 3 * 0.5 for t in truth], variance=0.25)
    inputs["mask"][1][10:14, 20:24] = True
    inputs["variance"][2][5, 7] = np.nan  # masked by `build_target`
    model = scene_with_fluxes(utils.build_target(**inputs))
    cutout_names = ["HST/ACS F814W", "r", "GALEX-FUV"]  # as the user named them
    images = utils.fit_images(model, cutout_names)

    assert list(images) == cutout_names
    for name, expected, spec in zip(cutout_names, truth, IMAGES.values()):
        assert set(images[name]) == {"model", "residual"}
        model_image, residual = images[name]["model"], images[name]["residual"]
        # numpy arrays in the orientation of the input cutout (rows, columns)
        assert model_image.shape == residual.shape == spec["shape"]
        assert model_image.flags.c_contiguous and model_image.dtype == np.float64
        np.testing.assert_allclose(model_image, expected, rtol=1e-12)
        np.testing.assert_allclose(residual[np.isfinite(residual)], 3.0, rtol=1e-9)

    # only masked pixels are NaN, whether masked by the user or by a bad variance
    masked = [np.isnan(images[name]["residual"]) for name in cutout_names]
    assert not masked[0].any()
    assert masked[1].sum() == 16 and masked[1][10:14, 20:24].all()
    assert masked[2].sum() == 1 and masked[2][5, 7]
    assert all(np.isfinite(images[name]["model"]).all() for name in cutout_names)
