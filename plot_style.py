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


def annotate_years(ax, years, xs, ys, *, size: float = 7.5, colors=None, forbidden=(), inside: bool = False) -> None:
    """Label each year beside its point. Nearby points get a staggered stack."""
    del forbidden
    font = arial(size)
    fig = ax.figure
    fig.canvas.draw()
    disp = np.array([ax.transData.transform((float(x), float(y))) for x, y in zip(xs, ys)])
    if inside:
        _layout_year_labels(ax, years, xs, ys, disp, font, fig)
        return
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0
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
    box = ax.get_window_extent()
    point_to_px = fig.dpi / 72.0
    for members in groups.values():
        members = sorted(members, key=lambda i: disp[i, 1])
        span = (len(members) - 1) / 2
        if inside and len(members) > 1:
            gap = 15 * point_to_px
            cy = float(np.mean(disp[members, 1]))
            cx = float(np.mean(disp[members, 0]))
            height = (len(members) - 1) * gap
            shift = 0.0
            if cy + height / 2 > box.y1 - 6:
                shift -= cy + height / 2 - (box.y1 - 6)
            if cy - height / 2 + shift < box.y0 + 6:
                shift += box.y0 + 6 - (cy - height / 2 + shift)
            place_left = cx > box.x0 + 0.62 * box.width
            for k, i in enumerate(members):
                label_y = cy - height / 2 + k * gap + shift
                dx = -9 if place_left else 8
                dy = (label_y - disp[i, 1]) * 72.0 / fig.dpi
                _year_annotation(
                    ax, years[i], xs[i], ys[i], dx, dy,
                    ha="right" if place_left else "left",
                    font=font,
                    color="#333333" if colors is None else colors[i],
                    leader=abs(dy) > 8,
                )
            continue
        for k, i in enumerate(members):
            color = "#333333" if colors is None else colors[i]
            dx, dy, ha = 8, (k - span) * 12, "left"
            if inside:
                xfrac = (float(xs[i]) - xmin) / xspan
                yfrac = (float(ys[i]) - ymin) / yspan
                if xfrac > 0.78:
                    dx, ha = -8, "right"
                if yfrac > 0.9:
                    dy = min(dy, -2)
                if yfrac < 0.1:
                    dy = max(dy, 2)
            _year_annotation(ax, years[i], xs[i], ys[i], dx, dy, ha=ha, font=font, color=color, leader=False)


def _layout_year_labels(ax, years, xs, ys, disp, font, fig) -> None:
    """Put each year just beside its point. No leader lines."""
    import matplotlib.patheffects as pe

    box = ax.get_window_extent()
    n = len(years)
    pos = disp.copy()
    ha = ["left"] * n
    for i in range(n):
        if disp[i, 0] > box.x0 + 0.78 * box.width:
            pos[i, 0] = disp[i, 0] - 8
            ha[i] = "right"
        else:
            pos[i, 0] = disp[i, 0] + 8
        pos[i, 1] = disp[i, 1] + 5
    tw, th = 16.0, 11.0
    base = pos.copy()
    for _ in range(50):
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                if abs(dx) >= tw or abs(dy) >= th:
                    continue
                sign = 1.0 if dy > 0 else -1.0
                if dy == 0:
                    sign = 1.0 if i < j else -1.0
                pos[i, 1] += sign * 1.6
                pos[j, 1] -= sign * 1.6
            pos[i, 1] += 0.08 * (base[i, 1] - pos[i, 1])
            pos[i, 1] = float(np.clip(pos[i, 1], base[i, 1] - 16, base[i, 1] + 16))
            pos[i, 1] = float(np.clip(pos[i, 1], box.y0 + 8, box.y1 - 8))
    scale = 72.0 / fig.dpi
    stroke = [pe.withStroke(linewidth=2.4, foreground="white")]
    for i in range(n):
        dx = (pos[i, 0] - disp[i, 0]) * scale
        dy = (pos[i, 1] - disp[i, 1]) * scale
        ax.annotate(
            str(int(years[i]))[-2:],
            (float(xs[i]), float(ys[i])),
            textcoords="offset points",
            xytext=(dx, dy),
            fontproperties=font,
            color="#1a1a1a",
            ha=ha[i],
            va="center",
            zorder=6,
            annotation_clip=False,
            path_effects=stroke,
        )


def _year_annotation(ax, year, x, y, dx, dy, *, ha, font, color, leader: bool) -> None:
    ax.annotate(
        str(int(year)),
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        fontproperties=font,
        color=color,
        ha=ha,
        va="center",
        zorder=5,
        annotation_clip=False,
        arrowprops={"arrowstyle": "-", "color": "#9a9a9a", "lw": 0.6, "shrinkA": 3, "shrinkB": 1} if leader else None,
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
