import astrophot as ap
import numpy as np
from astropy.wcs import WCS

from AstroPhot_tool.utils import (
    build_target,
    build_model,
    initialize_fluxes,
    fit_results,
    fit_images,
)


def joint_multi_image(
    cutout_name: list[str],  # N[str]
    cutout: list[np.ndarray],  # N[(H, W)]
    variance: list[np.ndarray],  # N[(H, W)]
    mask: list[np.ndarray],  # N[(H, W)]
    psf: list[np.ndarray],  # N[(Hp, Wp)]
    wcs: list[WCS],  # N[WCS]
    model_names: list[str],  # M[str]
    model_types: list[str],  # M[str]
    init_pos: np.ndarray,  # (M, 2)
    init_params: list[dict[str, np.ndarray]],  # M[dict[str, np.ndarray]]
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """
    Jointly fits multiple cutout images with a hierarchical model.

    The multiple cutout images are combined onto a common tangent plane. Then a
    series of models (indicated by `model_names` and `model_types`) are created
    which share structural parameters across all images, while allowing flux
    parameters to vary independently. The joint model is fit to all images
    simultaneously. This produces a single consistent set of model parameters
    for the desired object(s).

    Parameters
    ----------
    cutout_name : list[str]
        List of cutout image names (could be band, filter, or other identifier).
    cutout : list[np.ndarray]
        List of cutout image data arrays (may be different sizes).
    variance : list[np.ndarray]
        List of variance arrays corresponding to the cutout images (same sizes
        as corresponding cutout images).
    mask : list[np.ndarray]
        List of mask arrays corresponding to the cutout images (same sizes as
        corresponding cutout images), True for pixels to ignore.
    psf : list[np.ndarray]
        List of PSF arrays corresponding to the cutout images (may be different
        sizes). Each must have odd side lengths and be sampled on the pixel grid
        of its cutout.
    wcs : list[WCS]
        List of astropy WCS objects corresponding to the cutout images (must be
        in cutout image pixel coordinates).
    model_names : list[str]
        List of model names (any unique identifier).
    model_types : list[str]
        List of model types (see `astrophot.Model.List_Models(usable=True,
        types=True)` for all options).
    init_pos : np.ndarray
        Initial positions of the models (shape: M, 2) in Ra Dec coordinates.
    init_params : list[dict[str, np.ndarray]]
        Initial parameters for the models (may be empty dicts, anything not
        given is initialized automatically from the first cutout). Fluxes are
        always initialized from the data, see
        `AstroPhot_tool.utils.initialize_fluxes`.

    Returns
    -------
    results : dict
        Fitted parameters with uncertainties and fit statistics, see
        `AstroPhot_tool.utils.fit_results` for the layout. Plain python types
        only, so it can be written as JSON.
    images : dict[str, dict[str, np.ndarray]]
        Diagnostic images of the fit. For each cutout name, ``"model"`` is the
        fitted scene on the pixel grid of that cutout (in its flux units) and
        ``"residual"`` is ``(data - model) * sqrt(weight)``, the residual in
        units of the noise, with NaN in masked pixels. See
        `AstroPhot_tool.utils.fit_images`.
    """
    # Make target image list
    target = build_target(cutout_name, cutout, variance, mask, psf, wcs)

    # Make models
    model = build_model(target, model_names, model_types, init_pos, init_params)

    # Fit the model
    model.initialize()
    initialize_fluxes(model)
    fitter = ap.fit.LM(model, verbose=0).fit()

    # Return the fit result and the images of the fitted model
    return fit_results(model, fitter, cutout_name, model_names), fit_images(model, cutout_name)
