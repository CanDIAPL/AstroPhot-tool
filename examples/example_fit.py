"""
Example joint fit, makes the figure and results shown in the README.

A fake galaxy with a nearby star is observed by three very different
instruments (see ``tests/fake_data.py``), fit with `joint_multi_image`, and the
model and residual images it returns are drawn next to the data.

    python examples/example_fit.py

writes ``example_fit.png`` and ``example_results.json`` beside this script.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm, LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE.parent / "tests")]

import fake_data  # noqa: E402
from AstroPhot_tool import joint_multi_image  # noqa: E402

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
# brightness: one hue, light to dark. residuals: two hues around a neutral gray
BRIGHTNESS = LinearSegmentedColormap.from_list("brightness", [SURFACE, INK])
RESIDUAL = LinearSegmentedColormap.from_list("residual", ["#2a78d6", "#f0efec", "#e34948"])


def compact_json(results):
    """The results as JSON text with one parameter per line."""
    lines = ["{"]
    for key in ("message", "chi2_ndf", "iterations", "crval"):
        lines.append(f'  "{key}": {json.dumps(results[key])},')
    lines.append('  "models": {')
    models = list(results["models"].items())
    for m, (name, fit) in enumerate(models):
        lines.append(f'    "{name}": {{')
        for key in ("model_type", "ra", "dec"):
            lines.append(f'      "{key}": {json.dumps(fit[key])},')
        lines.append('      "shared": {')
        shared = [f'        "{k}": {json.dumps(v)}' for k, v in fit["shared"].items()]
        lines.append(",\n".join(shared))
        lines.append("      },")
        lines.append('      "per_image": {')
        images = [f'        "{k}": {json.dumps(v)}' for k, v in fit["per_image"].items()]
        lines.append(",\n".join(images))
        lines.append("      }")
        lines.append("    }" + ("," if m < len(models) - 1 else ""))
    lines += ["  }", "}"]
    text = "\n".join(lines)
    assert json.loads(text) == results  # same content, only the layout differs
    return text


def figure(dataset, images, results):
    """Data, fitted model and residual of every cutout, from the outputs of the fit."""
    names = dataset["cutout_name"]
    scale_bars = [2, 10, 30]  # arcsec, one per image
    # one row per image, as tall as its cutout, and a thin row for the color bar
    heights = [c.shape[0] / c.shape[1] for c in dataset["cutout"]] + [0.035]
    fig = plt.figure(figsize=(9.4, 9.6), constrained_layout=True, facecolor=SURFACE)
    grid = fig.add_gridspec(len(names) + 1, 3, height_ratios=heights)
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(len(names))])
    for row, (name, spec) in enumerate(fake_data.IMAGES.items()):
        data, model = dataset["cutout"][row], images[name]["model"]
        H, W = data.shape
        extent = [-0.5, W - 0.5, -0.5, H - 0.5]
        sky = spec["sky_sigma"]
        # the same stretch for the data and the model of one image
        norm = AsinhNorm(linear_width=3 * sky, vmin=-3 * sky, vmax=model.max())
        kwargs = dict(origin="lower", extent=extent, interpolation="nearest")
        axes[row, 0].imshow(data, cmap=BRIGHTNESS, norm=norm, **kwargs)
        axes[row, 1].imshow(model, cmap=BRIGHTNESS, norm=norm, **kwargs)
        # already in units of the noise: (data - model) * sqrt(weight)
        residual = axes[row, 2].imshow(
            images[name]["residual"], cmap=RESIDUAL, vmin=-4, vmax=4, **kwargs
        )

        scale = spec["pixelscale"]
        axes[row, 0].set_ylabel(
            f'{name}\n{scale}"/pixel, {W * scale:.0f}" x {H * scale:.0f}"', color=INK, fontsize=10
        )
        # scale bar, lower left of the data panel
        length = scale_bars[row] / scale
        x0, y0 = 0.06 * W, 0.07 * H
        axes[row, 0].plot([x0, x0 + length], [y0, y0], color=INK, lw=2, solid_capstyle="butt")
        axes[row, 0].text(
            x0 + length / 2, y0 + 0.03 * H, f'{scale_bars[row]}"', color=INK, fontsize=9, ha="center"
        )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#c3c2b7")

    # name the two objects once, where they are well separated
    for label, (ra, dec), offset in zip(results["models"], dataset["positions"], [(0, -32), (0, 14)]):
        x, y = dataset["wcs"][0].world_to_pixel_values(ra, dec)
        axes[0, 0].annotate(
            label, (x, y), xytext=offset, textcoords="offset points", color=INK, fontsize=9, ha="center"
        )

    for ax, title in zip(axes[0], ["data", "fitted model", "residual"]):
        ax.set_title(title, color=INK, fontsize=11)
    bar = fig.colorbar(residual, cax=fig.add_subplot(grid[-1, 2]), orientation="horizontal")
    bar.set_label("(data - model) / sigma", color=MUTED, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=9)
    bar.outline.set_edgecolor("#c3c2b7")
    return fig


if __name__ == "__main__":
    dataset = fake_data.fake_dataset(seed=42)
    # start 0.1 arcsec away from the true positions, everything else automatic
    init_pos = dataset["positions"] + np.random.default_rng(1).normal(size=(2, 2)) * 0.1 / 3600
    results, images = joint_multi_image(
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
    (HERE / "example_results.json").write_text(compact_json(results) + "\n")
    fig = figure(dataset, images, results)
    fig.savefig(HERE / "example_fit.png", dpi=110, facecolor=fig.get_facecolor())
    print(f"chi2/ndf = {results['chi2_ndf']:.4f}, written to {HERE}")
