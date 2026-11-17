"""
The fake cutouts and scene used by all of the tests.

The images are defined once, in `IMAGES`, and handed to the tests by the
fixtures in ``conftest.py``.
"""

import astrophot as ap
import numpy as np
from astropy.wcs import WCS

from AstroPhot_tool.utils import build_model, build_target


def make_wcs(ra, dec, pixelscale, shape, rotation=0.0, flip_ra=True):
    """
    TAN WCS for a cutout of the given (H, W) shape, centered on (ra, dec) in
    degrees. `pixelscale` in arcsec/pixel, `rotation` in radians. With
    `flip_ra` the image has the usual orientation (East to the left).
    """
    H, W = shape
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(W + 1) / 2, (H + 1) / 2]  # FITS 1-indexed image center
    c, s = np.cos(rotation), np.sin(rotation)
    wcs.wcs.cd = (
        np.array([[c, -s], [s, c]]) @ np.diag([-pixelscale if flip_ra else pixelscale, pixelscale])
    ) / 3600
    return wcs


def gaussian_psf(fwhm, size=None):
    """
    Unit sum circular Gaussian PSF image, `fwhm` in pixels. By default the stamp
    reaches 6 sigma, so the PSF is flat at the stamp edge and a point source
    keeps its flux (AstroPhot extrapolates linearly over the last half pixel).
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    if size is None:
        size = 2 * int(np.ceil(6 * sigma)) + 1
    y, x = np.mgrid[:size, :size] - size // 2
    psf = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return psf / psf.sum()


# A field observed at three very different resolutions. Each cutout has its own
# size, orientation, flux units and is centered on a slightly different point.
RA0, DEC0 = 150.1234, 2.2345
IMAGES = {  # typical of: HST, a ground based optical survey, a far UV survey
    "HST_F814W": dict(
        pixelscale=0.05,  # arcsec/pixel
        shape=(240, 240),  # (H, W), 12 arcsec
        rotation=np.deg2rad(20),
        psf_fwhm=0.1,  # arcsec
        offset=(0.0, 0.0),  # cutout center relative to (RA0, DEC0), arcsec
        sky_sigma=0.002,  # background noise per pixel, flux units of this image
        gain=2000.0,  # counts per flux unit, sets the source Poisson noise
    ),
    "ground_r": dict(
        pixelscale=0.8,
        shape=(56, 64),  # 45 x 51 arcsec
        rotation=0.0,
        psf_fwhm=1.2,
        offset=(0.3, -0.25),
        sky_sigma=15.0,
        gain=3.0,
    ),
    "GALEX-FUV": dict(
        pixelscale=3.0,
        shape=(41, 41),  # 123 arcsec
        rotation=np.deg2rad(-35),
        psf_fwhm=5.0,
        offset=(-1.1, 0.8),
        sky_sigma=0.0015,
        gain=4000.0,
    ),
}

# A galaxy with a star 4.3 arcsec away, the two are blended in the far UV.
MODEL_NAMES = ["galaxy", "star"]
MODEL_TYPES = ["sersic galaxy model", "point model"]
OFFSETS = np.array([[0.4, -0.3], [-3.5, 2.5]])  # East, North of (RA0, DEC0) in arcsec
STRUCTURE = [dict(q=0.6, PA=np.deg2rad(60), n=1.8, Re=1.2), {}]  # shared by all images
FLUXES = [  # independent in each image, in the flux units of that image
    {"Ie": {"HST_F814W": 2.0, "ground_r": 800.0, "GALEX-FUV": 0.03}},
    {"flux": {"HST_F814W": 30.0, "ground_r": 6000.0, "GALEX-FUV": 0.1}},
]


def offset_to_radec(offset, ra0=RA0, dec0=DEC0):
    """(East, North) offset in arcsec from (ra0, dec0) to (RA, Dec) in degrees."""
    offset = np.asarray(offset)
    return np.stack(
        [ra0 + offset[..., 0] / 3600 / np.cos(np.deg2rad(dec0)), dec0 + offset[..., 1] / 3600],
        axis=-1,
    )


POSITIONS = offset_to_radec(OFFSETS)  # (M, 2) true RA Dec of the models in degrees


def image_inputs(cutout=None, variance=1.0, specs=IMAGES):
    """
    The `build_target` inputs (``cutout_name``, ``cutout``, ``variance``,
    ``mask``, ``psf``, ``wcs``) for the cutouts described by `specs`.

    The cutouts are blank unless a list of arrays is given as `cutout`, with a
    uniform `variance` and nothing masked. A spec without a ``psf_fwhm`` gets a
    single pixel PSF (no blurring), ``rotation`` and ``offset`` default to zero
    and ``flip_ra`` to True (East to the left).
    """
    out = dict(cutout_name=[], cutout=[], variance=[], mask=[], psf=[], wcs=[])
    for i, (name, spec) in enumerate(specs.items()):
        shape, scale = spec["shape"], spec["pixelscale"]
        out["cutout_name"].append(name)
        out["cutout"].append(np.zeros(shape) if cutout is None else cutout[i])
        out["variance"].append(np.full(shape, variance))
        out["mask"].append(np.zeros(shape, dtype=bool))
        out["psf"].append(
            gaussian_psf(spec["psf_fwhm"] / scale) if "psf_fwhm" in spec else np.ones((1, 1))
        )
        out["wcs"].append(
            make_wcs(
                *offset_to_radec(spec.get("offset", (0.0, 0.0))),
                scale,
                shape,
                spec.get("rotation", 0.0),
                spec.get("flip_ra", True),
            )
        )
    return out


def fake_dataset(seed=42):
    """
    Use `build_model` to make noisy fake cutouts of the scene defined above.

    Returns a dict with the inputs for `joint_multi_image` (``cutout_name``,
    ``cutout``, ``variance``, ``mask``, ``psf``, ``wcs``), the noise free images
    (``clean``) and the true sky positions (``positions``, (M, 2) RA Dec).
    """
    rng = np.random.default_rng(seed)
    inputs = image_inputs()

    # The true scene, sampled onto every (blank) image
    model = build_model(build_target(**inputs), MODEL_NAMES, MODEL_TYPES, POSITIONS, STRUCTURE)
    for group, fluxes in zip(model.models, FLUXES):
        for sub_model, name in zip(group.models, IMAGES):
            for param, values in fluxes.items():
                sub_model[param].value = values[name]
    clean = [ap.backend.to_numpy(image) for image in model().data]

    # Background noise plus Poisson noise from the sources
    cutout, variance = [], []
    for image, spec in zip(clean, IMAGES.values()):
        var = spec["sky_sigma"] ** 2 + np.clip(image, 0, None) / spec["gain"]
        cutout.append(image + rng.normal(size=image.shape) * np.sqrt(var))
        variance.append(var)

    return dict(inputs, cutout=cutout, variance=variance, clean=clean, positions=POSITIONS)
