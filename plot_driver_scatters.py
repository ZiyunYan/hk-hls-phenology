#!/usr/bin/env python3
"""Scatter plots for the fixed-season correlations that clear |r| = 0.6.

The pairs are the Hong Kong median date against the mean of the HKO stations,
the same series behind territory_r in driver_lag_correlations.csv. Sliding
windows are left out: they move with the date.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import linregress, pearsonr

from analyze_driver_grids import (
    CLIM,
    es_kpa,
    station_matrix,
)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
YEARLY = FIG / "phenology_yearly_sos15_eos50.csv"
FONT = Path("/tmp/NotoSansSC-Regular.otf")


def season_mean(series: np.ndarray, days: pd.DatetimeIndex, years: np.ndarray, months: tuple[int, int, int]) -> np.ndarray:
    out = []
    for year in years:
        part = series[(days.year == int(year)) & days.month.isin(months)]
        if np.isfinite(part).mean() < 0.8:
            out.append(np.nan)
        else:
            out.append(float(np.nanmean(part)))
    return np.asarray(out, np.float64)


def territory_climate(years: np.ndarray) -> dict[str, np.ndarray]:
    days = pd.date_range("2014-01-01", "2025-12-31", freq="D")
    stations = pd.read_csv(CLIM / "stations.csv")
    temp_codes = stations.loc[stations["temp"] == 1, "code"].tolist()
    temp, temp_kept = station_matrix(temp_codes, "TEMP", days)
    dew_codes = [code for code in temp_kept if int(stations.set_index("code").loc[code, "dew"]) == 1]
    dew, dew_kept = station_matrix(dew_codes, "DEW", days)
    lookup = {code: i for i, code in enumerate(temp_kept)}
    dew_lookup = {code: i for i, code in enumerate(dew_kept)}
    vpd_rows = []
    for code in dew_kept:
        if code not in lookup:
            continue
        t = temp[lookup[code]]
        d = dew[dew_lookup[code]]
        both = np.isfinite(t) & np.isfinite(d)
        vpd = np.full(t.shape, np.nan)
        vpd[both] = np.clip(es_kpa(t[both]) - es_kpa(d[both]), 0, None)
        vpd_rows.append(vpd)
    return {
        "temp_jja": season_mean(np.nanmean(temp, axis=0), days, years, (6, 7, 8)),
        "temp_son": season_mean(np.nanmean(temp, axis=0), days, years, (9, 10, 11)),
        "vpd_mam": season_mean(np.nanmean(np.vstack(vpd_rows), axis=0), days, years, (3, 4, 5)),
    }


def fit_text(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    slope, intercept, r, p, _ = linregress(x[mask], y[mask])
    return float(slope), float(intercept), float(r), float(p)


def draw(frame: pd.DataFrame, path: Path) -> None:
    font = font_manager.FontProperties(fname=str(FONT)) if FONT.is_file() else font_manager.FontProperties(family="DejaVu Sans")
    panels = [
        ("vpd_mam", "sos", "3–5 月水汽压差与返青", "3–5 月水汽压差 (kPa)", "返青（日序）"),
        ("temp_son", "eos", "9–11 月气温与休眠", "9–11 月气温 (°C)", "休眠（日序）"),
        ("temp_jja", "eos", "6–8 月气温与休眠", "6–8 月气温 (°C)", "休眠（日序）"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), dpi=150)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.82, bottom=0.16, wspace=0.28)
    years = frame["year"].to_numpy()
    for ax, (xname, yname, title, xlabel, ylabel) in zip(axes, panels):
        x = frame[xname].to_numpy(np.float64)
        y = frame[yname].to_numpy(np.float64)
        late = years >= 2016
        slope, intercept, r_all, p_all = fit_text(x, y)
        slope2, intercept2, r_late, p_late = fit_text(x[late], y[late])
        order = np.argsort(x)
        xs = np.linspace(np.nanmin(x) - 0.02 * np.ptp(x), np.nanmax(x) + 0.02 * np.ptp(x), 50)
        ax.plot(xs, intercept + slope * xs, color="#1f4e79", lw=1.6, label="2015–2025")
        ax.plot(xs, intercept2 + slope2 * xs, color="#1f4e79", lw=1.2, ls="--", label="去掉 2015")
        highlight = (years == 2018) & (yname == "eos")
        ax.scatter(x[~highlight], y[~highlight], s=36, color="#1f4e79", zorder=3)
        if highlight.any():
            ax.scatter(x[highlight], y[highlight], s=46, color="#d62728", zorder=4)
        ax.margins(x=0.14, y=0.16)
        offsets = {
            "vpd_mam": {
                2016: (-2, 8), 2019: (6, -11), 2015: (6, 5), 2017: (7, 4),
                2024: (6, -11), 2020: (-26, 7), 2023: (7, -11), 2022: (6, 5),
                2025: (-26, 6), 2018: (6, -11), 2021: (6, 5),
            },
            "temp_son": {
                2018: (7, 4), 2021: (-28, 2), 2016: (-28, -10), 2020: (7, 6),
                2019: (-30, 8), 2017: (7, 8), 2023: (7, -12), 2025: (-28, -6),
                2015: (7, -12), 2022: (-30, 4), 2024: (6, -11),
            },
            "temp_jja": {
                2018: (7, 4), 2025: (-28, 6), 2021: (-28, -11), 2017: (7, 7),
                2016: (-28, -8), 2022: (-30, 3), 2024: (7, 5), 2019: (-28, 8),
                2015: (7, -12), 2023: (0, -13), 2020: (4, -12),
            },
        }[xname]
        for year, xv, yv in zip(years, x, y):
            color = "#d62728" if year == 2018 and yname == "eos" else "#333333"
            ax.annotate(
                str(int(year)), (xv, yv),
                textcoords="offset points", xytext=offsets[int(year)],
                fontsize=7.5, color=color, fontproperties=font, zorder=5,
            )
        ax.set_title(title, fontproperties=font, fontsize=12, loc="left")
        ax.set_xlabel(xlabel, fontproperties=font, fontsize=10)
        ax.set_ylabel(ylabel, fontproperties=font, fontsize=10)
        ax.tick_params(labelsize=8)
        note = f"2015–2025   r = {r_all:+.2f}   p = {p_all:.3f}\n去掉 2015   r = {r_late:+.2f}   p = {p_late:.3f}"
        # The VPD panel is empty in the upper right; the temperature panels are empty in the upper left.
        note_at = (0.97, 0.96, "right") if xname == "vpd_mam" else (0.04, 0.96, "left")
        ax.text(
            note_at[0], note_at[1], note, transform=ax.transAxes, va="top", ha=note_at[2],
            fontsize=8.5, fontproperties=font, color="#222222",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.92},
        )
        # Confirm the correlation matches pearsonr, which is what the table used.
        ax.margins(x=0.12, y=0.18)
        mask = np.isfinite(x) & np.isfinite(y)
        r_check, _ = pearsonr(x[mask], y[mask])
        if abs(r_check - r_all) > 1e-6:
            raise RuntimeError(f"{title} fit r {r_all} != pearson {r_check}")
    fig.suptitle(
        "全港中位日期与站点平均气候。点是年份。红点是 2018 年的休眠。",
        fontproperties=font, fontsize=12, x=0.06, ha="left",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, prop=font, bbox_to_anchor=(0.5, 0.01))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    yearly = pd.read_csv(YEARLY)
    climate = territory_climate(yearly["year"].to_numpy())
    frame = yearly[["year", "sos_median", "eos_median"]].rename(columns={"sos_median": "sos", "eos_median": "eos"})
    for name, values in climate.items():
        frame[name] = values
    table = FIG / "driver_scatter_points.csv"
    frame.to_csv(table, index=False)
    out = FIG / "driver_scatter_high_r.png"
    draw(frame, out)
    published = pd.read_csv(FIG / "driver_lag_correlations.csv")
    published = published[(published["grid_m"] == 2000) & (published["version"] == "2015-2025")]
    checks = [
        ("sos_mean", "vpd", "MAM", "vpd_mam", "sos"),
        ("eos_mean", "temp", "SON", "temp_son", "eos"),
        ("eos_mean", "temp", "JJA", "temp_jja", "eos"),
    ]
    for response, driver, window, xname, yname in checks:
        hit = published[(published["response"] == response) & (published["driver"] == driver) & (published["window"] == window)]
        r, _ = pearsonr(frame[xname], frame[yname])
        stored = float(hit["territory_r"].iloc[0])
        print(f"{response} {driver} {window}: plot {r:+.3f}  table {stored:+.3f}")
        if abs(r - stored) > 0.02:
            raise SystemExit(f"correlation drifted for {response} {driver} {window}")
    print(f"wrote {out}")
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
