import keyword
import re

import astrophot as ap
import numpy as np
from astropy.wcs import WCS
from scipy.optimize import nnls

RAD_TO_ARCSEC = np.rad2deg(1.0) * 3600.0

# Parameters which are fit independently in every image, everything else is
# shared (linked) across images. "delta" is the gradient of a plane sky model.
ASTROPHOT_FLUX_PARAMS = {"I0", "I", "Ie", "I_R", "Ib", "flux", "weights", "delta"}

# Subset of the flux parameters which are a single multiplicative amplitude, a
# model is exactly linear in these.
ASTROPHOT_AMPLITUDE_PARAMS = {"I0", "Ie", "Ib", "flux"}


def safe_name(name: str) -> str:
    """
    Convert an arbitrary label into a name AstroPhot will accept.

    AstroPhot (caskade) requires names to be valid Python identifiers, which
    rules out common labels such as "GALEX-FUV", "2MASS_J" or "HST/ACS F814W".
    Any character that is not a letter, digit or underscore is replaced by an
    underscore, and a leading underscore is added where still needed.
    """
    name = re.sub(r"\W", "_", str(name))
    if not name.isidentifier() or keyword.iskeyword(name):
        name = "_" + name
    return name


def world_to_plane(ra, dec, ra0, dec0):
    """
    Gnomonic (TAN) projection of (RA, Dec) onto the plane tangent at (ra0, dec0).

    All inputs in degrees, returns (x, y) in arcsec with x increasing to the
    East and y increasing to the North.
    """
    ra, dec, ra0, dec0 = (np.deg2rad(v) for v in (ra, dec, ra0, dec0))
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    x = np.cos(dec) * np.sin(ra - ra0)
    y = np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)
    return x * RAD_TO_ARCSEC / cosc, y * RAD_TO_ARCSEC / cosc


def normalize_wcs(
    wcs: WCS, new_crval: np.ndarray, pixel_scale_step: float = 10.0
) -> dict[str, np.ndarray]:
    """
    Determine tangent plane WCS parameters for the given new reference coordinate.

    Parameters
    ----------
    wcs : WCS
        Original WCS object.
    new_crval : np.ndarray
        New reference coordinate (RA, Dec) in degrees.
    pixel_scale_step : float, optional
        Step size in pixels for finite difference calculation, by default 10.0

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing normalized WCS parameters: crval (deg), crtan
        (arcsec), crpix (0-indexed pixel, x then y), and CD matrix (arcsec/pixel).
    """
    orig_center_x, orig_center_y = wcs.world_to_pixel_values(new_crval[0], new_crval[1])
    # Calculate the Jacobian (CD Matrix) via Finite Differences. The stepped
    # points are projected onto the new tangent plane, so the result is in the
    # coordinates AstroPhot works in and holds at any Dec (including the poles).
    step = pixel_scale_step * np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    ra, dec = wcs.pixel_to_world_values(orig_center_x + step[:, 0], orig_center_y + step[:, 1])
    x, y = world_to_plane(ra, dec, new_crval[0], new_crval[1])

    cd1_1 = 0.5 * (x[1] - x[0]) / pixel_scale_step
    cd2_1 = 0.5 * (y[1] - y[0]) / pixel_scale_step
    cd1_2 = 0.5 * (x[3] - x[2]) / pixel_scale_step
    cd2_2 = 0.5 * (y[3] - y[2]) / pixel_scale_step

    return {
        "crval": np.array(new_crval, dtype=np.float64),
        "crtan": np.array([0.0, 0.0]),
        "crpix": np.array([orig_center_x, orig_center_y], dtype=np.float64),
        "CD": np.array([[cd1_1, cd1_2], [cd2_1, cd2_2]]),
    }


def build_target(
    cutout_name: list[str],  # N[str]
    cutout: list[np.ndarray],  # N[(H, W)]
    variance: list[np.ndarray],  # N[(H, W)]
    mask: list[np.ndarray],  # N[(H, W)]
    psf: list[np.ndarray],  # N[(Hp, Wp)]
    wcs: list[WCS],  # N[WCS]
) -> ap.TargetImageList:
    """
    Build a target image list with consistent WCS parameters.

    Chooses a single crval reference coordinate (Ra Dec) and recomputes WCS
    parameters for all cutout images. This puts them all on the same tangent
    plane for simultaneous fitting. Then builds an AstroPhot target image for
    each cutout and combines them into a target image list.

    Parameters
    ----------
    cutout_name : list[str]
        List of cutout image names (could be band, filter, or other identifier).
    cutout : list[np.ndarray]
        List of cutout image data arrays (may be different sizes).
    variance : list[np.ndarray]
        List of variance arrays corresponding to the cutout images (same sizes
        as corresponding cutout images). Pixels whose variance is not
        finite and positive are masked.
    mask : list[np.ndarray]
        List of mask arrays corresponding to the cutout images (same sizes as
        corresponding cutout images), True for pixels to ignore.
    psf : list[np.ndarray]
        List of PSF arrays corresponding to the cutout images (may be different
        sizes). Each must have odd side lengths and be sampled on the pixel grid
        of its cutout, they are normalized to unit sum.
    wcs : list[WCS]
        List of astropy WCS objects corresponding to the cutout images (must be
        in cutout image pixel coordinates).

    Returns
    -------
    ap.TargetImageList
        Target image list with consistent WCS parameters. The images are in the
        same order as the inputs, with names passed through `safe_name`.
    """
    target_images = []
    H1, W1 = cutout[0].shape
    # Center of the first cutout (0-indexed pixel centers), as plain (RA, Dec) in degrees
    new_crval = np.array(wcs[0].pixel_to_world_values((W1 - 1) / 2, (H1 - 1) / 2))
    for c, v, m, p, w, n in zip(cutout, variance, mask, psf, wcs, cutout_name, strict=True):
        bad = ~(np.isfinite(v) & (v > 0))
        v = np.where(bad, 1.0, v)
        m = m | bad
        p = p / np.sum(p)
        n = safe_name(n)
        target_images.append(
            ap.TargetImage(name=n, data=c, variance=v, mask=m, psf=p, **normalize_wcs(w, new_crval))
        )
    return ap.TargetImageList(target_images)


def build_model(
    target: ap.TargetImageList,
    model_names: list[str],  # M[str]
    model_types: list[str],  # M[str]
    init_pos: np.ndarray,  # (M, 2)
    init_params: list[dict[str, np.ndarray]],  # M[dict[str, np.ndarray]]
) -> ap.Model:
    """
    Builds a hierarchical model of the provided target image list.

    Creates a model for each item in `model_names`, each model spans all of the
    target images. All structural parameters are linked across images, while
    flux parameters are allowed to vary independently. The final total model is
    given the name "full_scene".

    The result is nested as ``full_scene.models[m].models[n]`` for model ``m``
    in image ``n``, with the sub-model named "<model name>_<image name>".

    Parameters
    ----------
    target : ap.TargetImageList
        Target image list to build the model for (likely from
        `AstroPhot_tool.utils.build_target`).
    model_names : list[str]
        List of model names (any unique identifier).
    model_types : list[str]
        List of model types (see `astrophot.Model.List_Models(usable=True,
        types=True)` for all options).
    init_pos : np.ndarray
        Initial positions of the models (shape: M, 2) in Ra Dec coordinates.
    init_params : list[dict[str, np.ndarray]]
        Initial parameters for the models (may be empty dicts). The same values
        are used in every image, so it is rarely useful to provide a flux
        parameter here. Lengths are in arcsec and angles in radians, defined on
        the tangent plane.

    Returns
    -------
    ap.Model
        Hierarchical model of the target image list.
    """
    models = []
    # Explicit float64, otherwise positions given as python floats become
    # float32 which is only good to ~0.05 arcsec in RA.
    init_pos = ap.backend.as_array(
        np.asarray(init_pos, dtype=np.float64), dtype=ap.config.DTYPE, device=ap.config.DEVICE
    )
    for name, m_type, pos, params in zip(
        model_names, model_types, init_pos, init_params, strict=True
    ):
        name = safe_name(name)
        sub_model = []
        for i, tar in enumerate(target.images):
            if i == 0:
                # Use first image to find tangent plane location
                plane_pos = tar.world_to_plane(pos[0], pos[1])
            sub_model.append(
                ap.Model(
                    name=name + "_" + tar.name,
                    model_type=m_type,
                    center=plane_pos,
                    target=tar,
                    **params,
                )
            )
            if i != 0:  # Link structure params to first model
                # Only the model's own parameters, `all_params` would also
                # include the WCS parameters of the target image.
                for param_name in sub_model[-1].parameter_specs:
                    if param_name not in ASTROPHOT_FLUX_PARAMS:
                        sub_model[-1][param_name].to_pointer(sub_model[0][param_name])
        models.append(
            ap.Model(name=name, model_type="group model", target=target, models=sub_model)
        )

    return ap.Model(name="full_scene", model_type="group model", target=target, models=models)


def initialize_fluxes(model: ap.Model) -> None:
    """
    Set the flux of every model in every image by linear least squares.

    Every model built by `build_model` covers the whole of each cutout, so the
    per-model AstroPhot initialization assigns all of the light in a cutout to
    each model. For a scene with more than one model that is a poor start, and
    since fluxes are fit in log space a blended model can be driven to zero flux
    and stay there. Here the structural parameters are held at their current
    values and the fluxes in each image are solved for jointly, which is a
    linear problem (solved with non-negative least squares).

    Only parameters in `ASTROPHOT_AMPLITUDE_PARAMS` are updated, other models
    (e.g. spline profiles) are held fixed and subtracted. A flux which comes
    out below its own 1 sigma uncertainty (an undetected model) is set to that
    uncertainty, so the fit never starts at or next to zero flux. Any flux
    values given in `init_params` are overwritten.

    Parameters
    ----------
    model : ap.Model
        Model from `build_model`, with all parameters initialized (i.e. after
        calling ``model.initialize()``). Modified in place.
    """
    for n, tar in enumerate(model.target.images):
        weight = ap.backend.to_numpy(tar.weight) * ~ap.backend.to_numpy(tar.mask)
        sqrt_w = np.sqrt(weight).flatten()
        residual = ap.backend.to_numpy(tar.data).flatten()
        columns, amplitudes = [], []
        for group in model.models:
            sub_model = group.models[n]
            image = ap.backend.to_numpy(sub_model().data).flatten()
            amplitude = [
                sub_model[k]
                for k in sub_model.parameter_specs
                if k in ASTROPHOT_AMPLITUDE_PARAMS and sub_model[k].dynamic
            ]
            usable = len(amplitude) == 1 and amplitude[0].value.numel() == 1
            if usable:
                unit_image = image / amplitude[0].value.item()
                usable = np.all(np.isfinite(unit_image)) and np.any(unit_image * sqrt_w != 0)
            if usable:
                columns.append(unit_image * sqrt_w)
                amplitudes.append(amplitude[0])
            else:
                residual = residual - image
        if len(columns) == 0:
            continue
        A = np.stack(columns, axis=1)
        solution, _ = nnls(A, residual * sqrt_w)
        sigma = 1 / np.sqrt(np.sum(A**2, axis=0))
        for param, value, floor in zip(amplitudes, solution, sigma):
            param.value = max(value, floor)


def _param_entry(param) -> dict:
    uncertainty = param.uncertainty
    return {
        "value": ap.backend.to_numpy(param.value).tolist(),
        "uncertainty": None if uncertainty is None else ap.backend.to_numpy(uncertainty).tolist(),
        "units": param.units,
    }


def fit_results(
    model: ap.Model,
    fitter,
    cutout_name: list[str],  # N[str]
    model_names: list[str],  # M[str]
) -> dict:
    """
    Collect the outcome of a fit into a plain (JSON serializable) dictionary.

    Parameters
    ----------
    model : ap.Model
        Fitted model from `build_model`.
    fitter : ap.fit.LM
        The optimizer after calling ``fit()``.
    cutout_name : list[str]
        Cutout names as given by the user, used as keys in the output.
    model_names : list[str]
        Model names as given by the user, used as keys in the output.

    Returns
    -------
    dict
        - ``message``: convergence message from the optimizer.
        - ``chi2_ndf``: chi squared per degree of freedom over all images.
        - ``iterations``: number of optimizer iterations.
        - ``crval``: [RA, Dec] in degrees of the common tangent point.
        - ``models``: for each model name a dict with ``model_type``, the
          fitted position ``ra`` and ``dec`` in degrees, ``shared`` (parameters
          common to all images) and ``per_image`` (parameters fit separately in
          each cutout, keyed by cutout name). Every parameter is a dict of
          ``value``, ``uncertainty`` (1 sigma, None if unavailable) and
          ``units``. Fluxes are in the units of the corresponding cutout,
          lengths in arcsec and angles in radians on the tangent plane.
    """
    first = model.target.images[0]
    results = {
        "message": fitter.message,
        "chi2_ndf": float(fitter.chi2_ndf()),
        "iterations": len(fitter.loss_history) - 1,
        "crval": ap.backend.to_numpy(first.crval.value).tolist(),
        "models": {},
    }
    for group, name in zip(model.models, model_names, strict=True):
        ref = group.models[0]
        ra, dec = first.plane_to_world(*ref.center.value)
        entry = {
            "model_type": ref.model_type,
            "ra": ra.item(),
            "dec": dec.item(),
            "shared": {},
            "per_image": {},
        }
        for key in ref.parameter_specs:
            if key not in ASTROPHOT_FLUX_PARAMS:
                entry["shared"][key] = _param_entry(ref[key])
        for sub_model, image_name in zip(group.models, cutout_name, strict=True):
            entry["per_image"][image_name] = {
                key: _param_entry(sub_model[key])
                for key in sub_model.parameter_specs
                if key in ASTROPHOT_FLUX_PARAMS
            }
        results["models"][name] = entry
    return results


def fit_images(model: ap.Model, cutout_name: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """
    Sample the model on the pixel grid of every cutout, as diagnostic images.

    Parameters
    ----------
    model : ap.Model
        Model from `build_model`, at the parameter values to draw (normally
        after fitting).
    cutout_name : list[str]
        Cutout names as given by the user, used as keys in the output.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        For each cutout name a dict of two arrays with the shape of that cutout:

        - ``model``: the full scene (all models summed, PSF convolved) in the
          flux units of the cutout.
        - ``residual``: ``(data - model) * sqrt(weight)``, i.e. the residual in
          units of the noise (weight = 1 / variance). The sum of its squares is
          the chi squared of the fit. Masked pixels are NaN.
    """
    images = {}
    samples = model().images
    for name, tar, sample in zip(cutout_name, model.target.images, samples, strict=True):
        # contiguous arrays of our own, not views of AstroPhot's (transposed) tensors
        model_image = np.array(ap.backend.to_numpy(sample.data), dtype=np.float64, order="C")
        data, weight = ap.backend.to_numpy(tar.data), ap.backend.to_numpy(tar.weight)
        residual = (data - model_image) * np.sqrt(weight)
        residual[ap.backend.to_numpy(tar.mask)] = np.nan
        images[name] = {"model": model_image, "residual": residual}
    return images
