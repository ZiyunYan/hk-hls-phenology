#!/usr/bin/env python3
"""Median maps for the 15% and 50% amplitude-threshold dates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data/phenology/phenology_threshold_sl122_evi.nc"
META = ROOT / "data/meta.json"
FIG = ROOT / "figures"
HEIGHT = WIDTH = 1830
MIN_YEARS = 3


def load():
    with Dataset(SRC) as f:
        row = np.array(f.variables["row"][:], np.int32)
        col = np.array(f.variables["col"][:], np.int32)
        smooth = f.getncattr("smooth")
        dates = {
            name: np.array(f.variables[name][:], np.float32)
            for name in ("sos15", "sos50", "eos50", "eos15")
        }
    return row, col, dates, smooth


def grids(row, col, sos, eos):
    both = np.isfinite(sos) & np.isfinite(eos)
    n_valid = both.sum(axis=1).astype(np.float32)
    keep = n_valid >= MIN_YEARS
    season = np.where(both, eos - sos, np.nan)
    out = {
        "sos": np.full((HEIGHT, WIDTH), np.nan, np.float32),
        "eos": np.full((HEIGHT, WIDTH), np.nan, np.float32),
        "length": np.full((HEIGHT, WIDTH), np.nan, np.float32),
        "n_valid": np.full((HEIGHT, WIDTH), np.nan, np.float32),
    }
    out["n_valid"][row, col] = n_valid
    out["sos"][row[keep], col[keep]] = np.nanmedian(sos[keep], axis=1)
    out["eos"][row[keep], col[keep]] = np.nanmedian(eos[keep], axis=1)
    out["length"][row[keep], col[keep]] = np.nanmedian(season[keep], axis=1)
    return out


def draw(path: Path, mapped: dict, smooth: str, title: str, scales: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    bounds = json.loads(META.read_text())["bounds_lonlat"]
    extent = [bounds["west"], bounds["east"], bounds["south"], bounds["north"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.4))
    panels = list(zip(axes.ravel(), ("sos", "eos", "length", "n_valid"), scales))
    for ax, key, (cmap, vmin, vmax, panel, label) in panels:
        show = np.ma.masked_invalid(mapped[key])
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("#e6e6e6")
        im = ax.imshow(
            show, origin="upper", extent=extent, cmap=cmap_obj,
            norm=Normalize(vmin, vmax), interpolation="nearest", aspect="equal",
        )
        ax.set_title(panel, fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.tick_params(labelsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    shown = int(np.isfinite(mapped["sos"]).sum())
    med_sos = float(np.nanmedian(mapped["sos"]))
    med_eos = float(np.nanmedian(mapped["eos"]))
    med_len = float(np.nanmedian(mapped["length"]))
    fig.suptitle(
        f"{title}\n{smooth}. Pixels with at least {MIN_YEARS} valid seasons: {shown:,}. "
        f"Median SOS {med_sos:.0f}, EOS {med_eos:.0f}, length {med_len:.0f} days.",
        fontsize=12,
    )
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}  n={shown} sos={med_sos:.1f} eos={med_eos:.1f} length={med_len:.1f}", flush=True)


def main() -> None:
    row, col, dates, smooth = load()
    g15 = grids(row, col, dates["sos15"], dates["eos15"])
    g50 = grids(row, col, dates["sos50"], dates["eos50"])
    draw(
        FIG / "phenology_maps_threshold15_sl122_evi.png",
        g15,
        smooth,
        "Hong Kong, 122-day fill, EVI at 15% of the seasonal amplitude",
        [
            ("viridis", 60, 140, "Median green-up, 15%", "Day of year"),
            ("cividis", 300, 360, "Median dormancy, 15%", "Day of year"),
            ("YlGn", 180, 290, "Median season length, 15%", "Days"),
            ("Blues", 0, 11, "Years with both dates", "Years, 2015–2025"),
        ],
    )
    draw(
        FIG / "phenology_maps_threshold50_sl122_evi.png",
        g50,
        smooth,
        "Hong Kong, 122-day fill, EVI at 50% of the seasonal amplitude",
        [
            ("viridis", 90, 170, "Median midpoint of green-up, 50%", "Day of year"),
            ("cividis", 240, 340, "Median midpoint of senescence, 50%", "Day of year"),
            ("YlGn", 100, 220, "Median season length, 50%", "Days"),
            ("Blues", 0, 11, "Years with both dates", "Years, 2015–2025"),
        ],
    )


if __name__ == "__main__":
    main()
