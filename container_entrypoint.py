#!/usr/bin/env python3
"""FITS + JSON command-line adapter for the current AstroPhot-tool API.

Example:
    python container_entrypoint.py --config /input/config.json --output-dir /output
"""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path


def read_fits(spec, base_dir):
    """Read one explicit FITS HDU; its header is also the image's WCS source."""
    import numpy as np
    from astropy.io import fits

    path = Path(spec["path"])
    if not path.is_absolute():
        path = base_dir / path
    # hdu can be an integer (0 = primary) or an extension name, e.g. "SCI".
    with fits.open(path, memmap=False) as hdus:
        hdu = hdus[spec.get("hdu", 0)]
        if hdu.data is None:
            raise ValueError(f"No image data in selected HDU: {path}")
        # FITS arrays may be big-endian. Copy to native-endian float64 so the
        # NumPy arrays can be passed to AstroPhot/PyTorch without byte-order errors.
        array = np.array(hdu.data, dtype=np.float64, copy=True)
        header = hdu.header.copy()
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"Expected a nonempty 2D image: {path}")
    return array, header


def check_names(label, entries):
    from AstroPhot_tool.utils import safe_name

    names = [entry["name"] for entry in entries]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"Each {label} needs a nonempty name")
    # AstroPhot sanitizes punctuation: "r-band" and "r_band" would collide.
    if len({safe_name(name) for name in names}) != len(names):
        raise ValueError(
            f"{label} names must be unique! (note special characters are sanitized to underscores)"
        )


def prepare_inputs(config, base_dir):
    """Turn transport-friendly paths/JSON values into the library's real inputs."""
    import numpy as np
    from astropy.wcs import WCS

    cutouts = config["cutouts"]
    models = config["models"]
    check_names("cutout", cutouts)
    check_names("model", models)

    args = {key: [] for key in ("cutout_name", "cutout", "variance", "mask", "psf", "wcs")}
    for entry in cutouts:
        image, header = read_fits(entry["image"], base_dir)
        variance, _ = read_fits(entry["variance"], base_dir)
        mask_pixels, _ = read_fits(entry["mask"], base_dir)
        mask = mask_pixels != 0
        psf, _ = read_fits(entry["psf"], base_dir)
        for label, array in (("variance", variance), ("mask", mask)):
            if array.shape != image.shape:
                raise ValueError(
                    f"{label} shape {array.shape} does not match image shape "
                    f"{image.shape}: {entry['name']}"
                )
        wcs = WCS(header)
        if not wcs.array_shape == image.shape:
            raise ValueError(f"WCS array shape does not match image shape: {entry['name']}")
        # Without sky axes astropy silently falls back to pixel coordinates
        if not wcs.has_celestial:
            raise ValueError(f"No RA/Dec WCS in the image header: {entry['name']}")
        values = (entry["name"], image, variance, mask, psf, wcs)
        for key, value in zip(args, values, strict=True):
            args[key].append(value)

    positions = np.asarray([[m["ra"], m["dec"]] for m in models], dtype=np.float64)
    args.update(
        model_names=[m["name"] for m in models],
        model_types=[m["type"] for m in models],
        init_pos=positions,
        init_params=[m.get("params", {}) for m in models],
    )
    return args


# What each diagnostic image is, written into its FITS header
IMAGE_KINDS = {
    "model": "Fitted model of the cutout, in the flux units of the cutout",
    "residual": "(data - model) * sqrt(weight), masked pixels are NaN",
}


def write_images(images, wcs, output_dir):
    """
    Write the model and residual image of every cutout as FITS files.

    Each file is a single image with the WCS of its cutout, named
    ``<cutout name>_model.fits`` and ``<cutout name>_residual.fits`` (with the
    name made safe, as AstroPhot uses it). Returns the file names for each
    cutout name, relative to the output directory.
    """
    from astropy.io import fits

    from AstroPhot_tool.utils import safe_name

    files = {}
    for (name, cutout_images), cutout_wcs in zip(images.items(), wcs, strict=True):
        files[name] = {}
        for kind, description in IMAGE_KINDS.items():
            # relax: also keep distortion terms of the input WCS (e.g. SIP)
            header = cutout_wcs.to_header(relax=True)
            header["CUTOUT"] = (name, "Cutout name")
            header["IMGTYPE"] = (kind, "AstroPhot-tool output image")
            header["COMMENT"] = description
            file_name = f"{safe_name(name)}_{kind}.fits"
            fits.PrimaryHDU(cutout_images[kind], header=header).writeto(output_dir / file_name)
            files[name][kind] = file_name
    return files


def finite_or_none(value):
    """
    Replace NaN and infinite numbers by None (JSON null), at any depth.

    JSON has no NaN, and a failed fit can leave one in a value or uncertainty.
    Returns the cleaned copy and the number of values replaced.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None, 1
    if isinstance(value, dict):
        cleaned = {key: finite_or_none(item) for key, item in value.items()}
        return {k: v for k, (v, _) in cleaned.items()}, sum(n for _, n in cleaned.values())
    if isinstance(value, (list, tuple)):
        cleaned = [finite_or_none(item) for item in value]
        return [v for v, _ in cleaned], sum(n for _, n in cleaned)
    return value, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="JSON config containing cutouts and models"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Writable directory on persistent storage"
    )
    cli = parser.parse_args()

    # Absolute paths, since the working directory changes below
    config_file = cli.config.resolve()
    output_dir = cli.output_dir.resolve()
    with config_file.open(encoding="utf-8") as stream:
        config = json.load(stream)

    # AstroPhot creates AstroPhot.log in the working directory when it is
    # imported, and fails if it can not. Work from a fresh temporary directory
    # so this does not depend on where the container was started.
    os.chdir(tempfile.mkdtemp(prefix="astrophot-"))

    kwargs = prepare_inputs(config, config_file.parent)

    from AstroPhot_tool import joint_multi_image

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    config_path = output_dir / "config.used.json"
    if result_path.exists() or config_path.exists():
        raise FileExistsError(
            "Use a new output directory for each run; result/config already exists"
        )

    with config_path.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")

    results, images = joint_multi_image(**kwargs)
    # Where each image was saved goes in the results, for the next stage
    results["images"] = write_images(images, kwargs["wcs"], output_dir)

    # Never lose a finished fit to a single NaN: write those as null
    results, replaced = finite_or_none(results)
    if replaced:
        print(f"Warning: {replaced} non-finite result value(s) written as null", file=sys.stderr)
    result_json = json.dumps(results, indent=2, allow_nan=False)
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(result_json + "\n")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        print(f"AstroPhot input/output error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
