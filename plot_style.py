"""Shared matplotlib styling for HK phenology figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def use_arial() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
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
