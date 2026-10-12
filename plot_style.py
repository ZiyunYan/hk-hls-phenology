"""Shared matplotlib styling for HK phenology figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from numpy.typing import NDArray

_FONT_DIR = Path("/work/home/ziyun2026/miniconda3/envs/timesfm/fonts")
ARIAL = _FONT_DIR / "arial.ttf"
ARIAL_BOLD = _FONT_DIR / "arialbd.ttf"


def use_arial() -> None:
    for path in (ARIAL, ARIAL_BOLD, _FONT_DIR / "ariali.ttf", _FONT_DIR / "arialbi.ttf"):
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def arial(size: float = 10) -> font_manager.FontProperties:
    use_arial()
    return font_manager.FontProperties(fname=str(ARIAL), size=size)


def annotate_years(ax, years, xs, ys, *, size: float = 7.5, colors=None, forbidden=()) -> None:
    """Label each year beside its point. Nearby points get a staggered stack."""
    del forbidden
    font = arial(size)
    fig = ax.figure
    fig.canvas.draw()
    disp = np.array([ax.transData.transform((float(x), float(y))) for x, y in zip(xs, ys)])
    parent = list(range(len(years)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            if np.hypot(*(disp[i] - disp[j])) < 42:
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(years)):
        groups.setdefault(find(i), []).append(i)
    for members in groups.values():
        members = sorted(members, key=lambda i: disp[i, 1])
        span = (len(members) - 1) / 2
        for k, i in enumerate(members):
            color = "#333333" if colors is None else colors[i]
            ax.annotate(
                str(int(years[i])),
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=(8, (k - span) * 12),
                fontproperties=font,
                color=color,
                zorder=5,
                annotation_clip=False,
            )


def year_stats_table(
    fig,
    years: NDArray[np.integer],
    means: NDArray[np.floating],
    stds: NDArray[np.floating],
    *,
    rect: tuple[float, float, float, float] = (0.08, 0.02, 0.9, 0.14),
    extra_rows: list[tuple[str, NDArray[np.floating]]] | None = None,
) -> None:
    """Mean/std table under the plot (no overlap with boxplots)."""
    ax_t = fig.add_axes(rect)
    ax_t.axis("off")
    rows: list[list[str]] = [
        [f"{m:.1f}" for m in means],
        [f"{s:.1f}" for s in stds],
    ]
    row_labels = ["Mean", "Std"]
    if extra_rows:
        for label, vals in extra_rows:
            rows.append([f"{v:.1f}" for v in vals])
            row_labels.append(label)
    table = ax_t.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=[str(int(y)) for y in years],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight="bold")
        cell.set_linewidth(0.4)
    table.scale(1.0, 1.35)
