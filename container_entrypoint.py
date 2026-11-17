#!/usr/bin/env python3
"""FITS + JSON command-line adapter for the current AstroPhot-tool API.

Example:
    python container_entrypoint.py --config /input/config.json --output-dir /output
"""

import argparse
import json
import sys
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
        wcs = WCS(header)
        if not wcs.array_shape == image.shape:
            raise ValueError(f"WCS array shape does not match image shape: {entry['name']}")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="JSON config containing cutouts and models"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Writable directory on persistent storage"
    )
    cli = parser.parse_args()

    with cli.config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    config_dir = cli.config.resolve().parent
    kwargs = prepare_inputs(config, config_dir)

    from AstroPhot_tool import joint_multi_image

    cli.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cli.output_dir / "results.json"
    config_path = cli.output_dir / "config.used.json"
    if result_path.exists() or config_path.exists():
        raise FileExistsError(
            "Use a new output directory for each run; result/config already exists"
        )

    with config_path.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")

    results = joint_multi_image(**kwargs)

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
