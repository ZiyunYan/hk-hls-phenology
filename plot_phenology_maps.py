#!/usr/bin/env python3
"""Map median SOS, EOS, and season length from the saved phenology cubes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent
PHENO = ROOT / "data" / "phenology"
META = ROOT / "data" / "meta.json"
FIG = ROOT / "figures"
HEIGHT = WIDTH = 1830
MIN_YEARS = 3


def load_cube(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = PHENO / f"phenology_sos_eos_{name}.nc"
    with Dataset(path) as src:
        row = np.array(src.variables["row"][:], np.int32)
        col = np.array(src.variables["col"][:], np.int32)
        sos = np.array(src.variables["sos"][:], np.float32)
        eos = np.array(src.variables["eos"][:], np.float32)
    return row, col, sos, eos


def median_maps(row, col, sos, eos) -> dict[str, np.ndarray]:
    both = np.isfinite(sos) & np.isfinite(eos)
    n_valid = both.sum(axis=1).astype(np.float32)
    keep = n_valid >= MIN_YEARS
    season = np.where(both, eos - sos, np.nan)
    grids = {
        "sos": np.full((HEIGHT, WIDTH), np.nan, np.float32),
        "eos": np.full((HEIGHT, WIDTH), np.nan, np.float32),
        "length": np.full((HEIGHT, WIDTH), np.nan, np.float32),
        "n_valid": np.full((HEIGHT, WIDTH), np.nan, np.float32),
    }
    grids["n_valid"][row, col] = n_valid
    grids["sos"][row[keep], col[keep]] = np.nanmedian(sos[keep], axis=1)
    grids["eos"][row[keep], col[keep]] = np.nanmedian(eos[keep], axis=1)
    grids["length"][row[keep], col[keep]] = np.nanmedian(season[keep], axis=1)
    return grids


def extent() -> list[float]:
    bounds = json.loads(META.read_text())["bounds_lonlat"]
    return [bounds["west"], bounds["east"], bounds["south"], bounds["north"]]


def _draw(ax, grid, bounds, cmap, vmin, vmax, title, label, norm=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    show = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("#e6e6e6")
    norm = norm or Normalize(vmin, vmax)
    im = ax.imshow(
        show,
        origin="upper",
        extent=bounds,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=8)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return im


def plot_hk(grids) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bounds = extent()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.4))
    panels = [
        (axes[0, 0], grids["sos"], "viridis", 70, 140, "Median SOS", "Day of year"),
        (axes[0, 1], grids["eos"], "cividis", 300, 360, "Median EOS", "Day of year"),
        (axes[1, 0], grids["length"], "YlGn", 180, 280, "Median season length", "Days"),
        (axes[1, 1], grids["n_valid"], "Blues", 0, 11, "Years with SOS and EOS", "Years, 2015–2025"),
    ]
    for ax, grid, cmap, vmin, vmax, title, label in panels:
        _draw(ax, grid, bounds, cmap, vmin, vmax, title, label)
    fig.suptitle(
        "Hong Kong HLS 49QHE, local 6-band gap fill, EVI\n"
        f"Zhang logistic after Savitzky–Golay (window 9). "
        f"Timing maps keep pixels with at least {MIN_YEARS} valid seasons.",
        fontsize=12,
    )
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "phenology_maps_hk_evi.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_compare(maps: dict[str, dict[str, np.ndarray]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    bounds = extent()
    diff = maps["usa_evi"]["sos"] - maps["hk_evi"]["sos"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.4))
    panels = [
        (axes[0, 0], maps["hk_evi"]["sos"], "Median SOS, Hong Kong fill, EVI"),
        (axes[0, 1], maps["hk_evi2"]["sos"], "Median SOS, Hong Kong fill, EVI2"),
        (axes[1, 0], maps["usa_evi"]["sos"], "Median SOS, USA fill, EVI"),
    ]
    for ax, grid, title in panels:
        _draw(ax, grid, bounds, "viridis", 60, 210, title, "Day of year")
    _draw(
        axes[1, 1],
        diff,
        bounds,
        "RdBu_r",
        None,
        None,
        "USA minus Hong Kong SOS, EVI",
        "Days",
        norm=TwoSlopeNorm(vcenter=0, vmin=-40, vmax=120),
    )
    fig.suptitle(
        "Start of season, 2015–2025 median\n"
        f"Pixels with fewer than {MIN_YEARS} valid seasons are blank. "
        "Positive difference means the USA fill places SOS later.",
        fontsize=12,
    )
    fig.tight_layout()
    path = FIG / "phenology_maps_sos_compare.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    names = ("hk_evi", "hk_evi2", "usa_evi")
    maps = {}
    for name in names:
        row, col, sos, eos = load_cube(name)
        maps[name] = median_maps(row, col, sos, eos)
        shown = int(np.isfinite(maps[name]["sos"]).sum())
        print(f"{name} mapped pixels (>={MIN_YEARS} years): {shown}")
    hk = plot_hk(maps["hk_evi"])
    compare = plot_compare(maps)
    print(f"wrote {hk}")
    print(f"wrote {compare}")


if __name__ == "__main__":
    main()
