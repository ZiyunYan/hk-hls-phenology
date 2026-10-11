#!/usr/bin/env python3
"""Draw the locked Hong Kong HLS pipeline and print the script for each step.

The figure is the record of the run that produced the published dates:
122-day fill, Savitzky–Golay window 15, SOS at 15% of green-up, EOS at 50%
of senescence, then station-climate correlations on 2/5/10 km cells.

This script does not download, train, or fill. It only documents that order.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "figures" / "pipeline_flowchart.png"
FONT_CANDIDATES = (
    Path("/tmp/NotoSansSC-Regular.otf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ROOT / "figures" / "NotoSansSC-Regular.otf",
)

# Each step names the Python (or shell) entry that actually does the work.
STEPS = (
    {
        "id": "1",
        "name": "数据下载",
        "scripts": (
            "本地 HLS 景：data/granules/*.npz、granules.json、meta.json",
            "规划署土地利用栅格（2018、2024），由 composite / apply_veg_mask 读取",
            "download_hko.py",
        ),
    },
    {
        "id": "2",
        "name": "三天合成与植被掩膜",
        "scripts": ("composite_veg_cube.py", "apply_veg_mask.py"),
    },
    {
        "id": "3a",
        "name": "模型训练",
        "scripts": ("train_imputator_window_ablation.sh", "imputator_ssl/run.py"),
    },
    {
        "id": "3b",
        "name": "填补前清洗",
        "scripts": ("bolton_clean.py", "apply_bolton_clean.py"),
    },
    {
        "id": "4",
        "name": "缺口填补",
        "scripts": ("compare_impute_windows.py", "impute_fast.py"),
    },
    {
        "id": "5",
        "name": "物候提取",
        "scripts": ("phenology_threshold.py", "plot_threshold_maps.py"),
    },
    {
        "id": "6",
        "name": "气候分析",
        "scripts": ("analyze_driver_grids.py",),
    },
)


def load_font() -> font_manager.FontProperties:
    for path in FONT_CANDIDATES:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path))
    raise SystemExit(
        "需要中文字体才能画这张图。把 NotoSansSC-Regular.otf 放到 /tmp 后再运行。"
    )


def box(ax, x, y, w, h, face, edge, title, lines, font, title_size=12.5, body_size=9.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.08",
        linewidth=1.3, facecolor=face, edgecolor=edge,
        mutation_aspect=0.6, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.16, y + h - 0.28, title,
        fontproperties=font, fontsize=title_size, color=edge,
        fontweight="bold", va="top", ha="left", zorder=3,
    )
    ax.text(
        x + 0.16, y + h - 0.58, "\n".join(lines),
        fontproperties=font, fontsize=body_size, color="#1c1c1c",
        va="top", ha="left", linespacing=1.35, zorder=3,
    )


def arrow(ax, x1, y1, x2, y2, color="#4a4a4a"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.35, color=color, zorder=1,
        shrinkA=0, shrinkB=0,
    ))


def band(ax, x, y, w, h, face):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.1",
        linewidth=0, facecolor=face, zorder=0,
    ))


def render(path: Path, font: font_manager.FontProperties) -> None:
    fig_w, fig_h = 16.4, 23.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.4, 23.25, "香港 HLS 物候流程",
        fontproperties=font, fontsize=22, color="#1a1a1a", fontweight="bold", va="top",
    )
    ax.text(
        0.4, 22.72,
        "瓦片 49QHE  ·  2015–2025  ·  30 米  ·  正式结果：sl122 填补，SG15，返青 15%，休眠 50%",
        fontproperties=font, fontsize=11, color="#444444", va="top",
    )

    # --- 1. downloads ---
    band(ax, 0.28, 20.15, 15.85, 2.35, "#f4f7fb")
    ax.text(0.45, 22.28, "1  数据下载", fontproperties=font, fontsize=13, color="#24598a", fontweight="bold", va="top")
    box(
        ax, 0.45, 20.32, 4.95, 1.82, "#e8f1fb", "#24598a",
        "HLS 影像",
        [
            "L30 与 S30，瓦片 49QHE",
            "清单约 1141 景，云量不超过 90%",
            "合成只用其中的 2015–2025",
            "granules/*.npz",
            "另有 granules.json、meta.json",
        ],
        font, body_size=8.8,
    )
    box(
        ax, 5.6, 20.32, 4.95, 1.82, "#e8f1fb", "#24598a",
        "土地利用",
        [
            "规划署 LUM 2018 与 2024",
            "原始约 10 米",
            "投到 HLS 30 米格网",
            "每个像元取众数类别",
            "耕地、城市用来做掩膜",
        ],
        font, body_size=8.8,
    )
    box(
        ax, 10.75, 20.32, 5.15, 1.82, "#e8f1fb", "#24598a",
        "天文台气候   download_hko.py",
        [
            "站点表来自天文台网页",
            "日值气温、雨量、风速、露点",
            "climate/daily_*_ALL.csv",
            "climate/stations.csv",
            "这一步的结果留到第 6 步再用",
        ],
        font, body_size=8.8,
    )

    # --- 2. cube and mask ---
    band(ax, 0.28, 16.55, 15.85, 3.35, "#f3faf8")
    ax.text(0.45, 19.68, "2  合成与掩膜", fontproperties=font, fontsize=13, color="#1d6b5c", fontweight="bold", va="top")
    box(
        ax, 0.45, 16.72, 7.55, 2.78, "#e7f6f2", "#1d6b5c",
        "三天合成   composite_veg_cube.py",
        [
            "按景去掉云、云阴影、卷云、水、雪",
            "六个光学波段相对香港均值做 |z| ≥ 4 过滤",
            "波段：Blue、Green、Red、NIR、SWIR1、SWIR2",
            "反射率 × 10000 存成 int16，0 表示缺失",
            "每 3 天取有效像元的中值，一年 122 步",
            "2015–2025 共 11 年、1342 个时间步",
            "网格 1830 × 1830，30 米，UTM 49N",
            "输出  data/phenology/veg_cube.nc",
        ],
        font,
    )
    box(
        ax, 8.2, 16.72, 7.7, 2.78, "#e7f6f2", "#1d6b5c",
        "稳定自然植被   apply_veg_mask.py",
        [
            "去掉耕地（土地利用代码 61）",
            "保留林地、灌木、草地、红树林（71–74）",
            "2018 和 2024 两年都必须是自然植被",
            "两年都不能是城市",
            "不满足的像元整条时间序列置 0",
            "约 75 万个像元进入后面的清洗和填补",
            "掩膜写回 veg_cube.nc 的 vegetation",
        ],
        font,
    )

    # --- 3. train and clean, side by side ---
    band(ax, 0.28, 12.15, 7.75, 4.15, "#f8f4fb")
    band(ax, 8.2, 12.15, 7.93, 4.15, "#fff8ef")
    ax.text(0.45, 16.08, "3a  模型训练", fontproperties=font, fontsize=13, color="#5e348e", fontweight="bold", va="top")
    ax.text(8.38, 16.08, "3b  填补前清洗", fontproperties=font, fontsize=13, color="#a86b12", fontweight="bold", va="top")
    box(
        ax, 0.45, 12.32, 7.4, 3.55, "#f3eaf8", "#5e348e",
        "三个窗口   run.py",
        [
            "训练像元 imputator_hk/train.nc",
            "约 3.75 万像元；留出 test.nc 约 0.47 万",
            "启动脚本 train_imputator_window_ablation.sh",
            "Transformer：256 维，8 头，6 层，前馈 1024",
            "随机掩掉 80% 时间步，用 MSE 重建",
            "每条序列丢掉误差最大的 5 步",
            "二阶差分平滑，权重 1",
            "seq_len = 122、244 或 366（约 1、2、3 年）",
            "权重目录 imputator_ssl/checkpoints/",
            "sl122 一年一窗；sl244 两年一窗",
            "sl366 用三年上下文，写出中间一年",
        ],
        font, body_size=8.7,
    )
    box(
        ax, 8.38, 12.32, 7.55, 3.55, "#fff4e5", "#a86b12",
        "亮异常与 EVI 尖峰   bolton_clean.py",
        [
            "入口 apply_bolton_clean.py",
            "对象是合成后的观测，此时还没有填补",
            "亮异常沿用 Bolton 等（2020）",
            "蓝光相对前后晴空日抬升则删",
            "若红光抬得更多，当成植被变化，保留",
            "EVI 尖峰：前后晴空日间隔不超过 90 天",
            "相对线性插值偏离超过 0.1，向上向下都删",
            "当天六个波段一起改回缺失（0）",
            "不做雪填补、样条，也不做地形校正",
            "输出  veg_cube_bolton.nc",
        ],
        font, body_size=8.7,
    )

    # --- 4. fill ---
    band(ax, 0.28, 8.85, 15.85, 3.05, "#f3faf3")
    ax.text(0.45, 11.68, "4  缺口填补", fontproperties=font, fontsize=13, color="#246b32", fontweight="bold", va="top")
    box(
        ax, 0.45, 9.02, 15.5, 2.48, "#e8f6ea", "#246b32",
        "三个模型各填一份   impute_fast.py",
        [
            "先用 compare_impute_windows.py 在留出步上比 122、244、366 的误差",
            "输入是洗过的立方体 veg_cube_bolton.nc，再加上面对应的 checkpoint",
            "只改写原来缺失的时间步（DN = 0）；晴空观测原样保留",
            "sl122：每年 122 步单独前向，11 年共 11 次",
            "sl244：每两年一块，最后一年（第 1220 步之后）改用 sl122",
            "sl366：每次看 366 步，首尾年对齐序列两端，中间年取窗口正中的 122 步",
            "输出 veg_filled_sl122.nc、veg_filled_sl244.nc、veg_filled_sl366.nc",
            "正式物候用 sl122。submit_impute_windows.sh 末尾的 SG9 曲率日期只是对照，不作为结果",
        ],
        font, body_size=9.0,
    )

    # --- 5. phenology ---
    band(ax, 0.28, 5.15, 15.85, 3.45, "#fbf8ef")
    ax.text(0.45, 8.38, "5  物候提取", fontproperties=font, fontsize=13, color="#8a6a10", fontweight="bold", va="top")
    box(
        ax, 0.45, 5.32, 15.5, 2.88, "#fff8e8", "#8a6a10",
        "阈值法   phenology_threshold.py",
        [
            "输入 veg_filled_sl122.nc，指数用 Huete EVI，按日历年分开",
            "每年用 Savitzky–Golay 平滑：窗口 15（约 45 天），多项式 2 阶",
            "SG9（约 27 天）太窄，上升段会多次穿过 15%；双 logistic 拟合不稳定，两条都不用",
            "在平滑曲线的峰值处把一年切开。返青取上升段第一次到达振幅 15% 的日期",
            "休眠取下降段第一次落到剩余振幅 50% 的日期。日期在相邻 3 天样本之间线性插值",
            "振幅小于 0.04、有效步少于 12、或季长短于 30 天的年份丢掉",
            "同时记下上升 50% 和下降 15%，正式生长季只用返青 15% 配休眠 50%",
            "输出 phenology_threshold_sl122_evi.nc。空间图由 plot_threshold_maps.py 绘制",
            "至少 3 个有效年的像元约 75.0 万。中位数：返青第 95 天，休眠第 302 天，季长 212 天",
        ],
        font, body_size=8.9,
    )

    # --- 6. analysis ---
    band(ax, 0.28, 0.95, 15.85, 3.95, "#f4f6fc")
    ax.text(0.45, 4.68, "6  气候与物候", fontproperties=font, fontsize=13, color="#303f9f", fontweight="bold", va="top")
    box(
        ax, 0.45, 1.12, 15.5, 3.38, "#eef1fb", "#303f9f",
        "格网滞后   analyze_driver_grids.py",
        [
            "物候仍是 30 米像元，先放进 2、5、10 公里格网，格内算均值和标准差",
            "气温、雨量、风速、露点从站点反距离权重插到格心，不插到每个 30 米像元",
            "站距大约是气温 2.8 公里、水汽压差 3.6 公里、雨量 4.0 公里、风速 5.0 公里",
            "水汽压差 = es(气温) − es(露点)，Tetens 公式，单位 kPa，先在站点上算再插值",
            "主结果用固定季节：3–5 月对返青，6–8 月和 9–11 月对休眠",
            "另算了紧贴物候日期的 30、60、90 天。窗口会跟着日期一起挪，休眠提前就落在更暖的时段，不作为主结果",
            "两版年份：2015–2025，以及去掉 Sentinel-2 质量较差的 2015",
            "5 公里和 10 公里上的全港相关与 2 公里几乎一样",
            "图 figures/driver_lag_correlations.png、phenology_yearly_sos15_eos50.png",
            "表 figures/driver_lag_correlations.csv、phenology_yearly_sos15_eos50.csv",
        ],
        font, body_size=8.9,
    )

    ax.text(
        0.45, 0.72,
        "早期曲率法 phenology_hplm.py、phenology_from_cube.py 只用来对照取点位置，不进入正式日期。立方体和权重不进 Git，图和脚本在仓库里。",
        fontproperties=font, fontsize=8.6, color="#555555", va="top",
    )

    # arrows
    arrow(ax, 2.9, 20.32, 4.2, 19.50)
    arrow(ax, 8.05, 20.32, 12.0, 19.50)
    arrow(ax, 4.2, 16.72, 4.15, 15.87)
    arrow(ax, 12.0, 16.72, 12.15, 15.87)
    arrow(ax, 4.15, 12.32, 6.4, 11.50)
    arrow(ax, 12.15, 12.32, 10.2, 11.50)
    arrow(ax, 8.2, 9.02, 8.2, 8.20)
    arrow(ax, 8.2, 5.32, 8.2, 4.50)
    # climate joins the analysis from the download box
    ax.plot([15.55, 15.95, 15.95], [20.32, 20.32, 4.55], color="#24598a", lw=1.15, zorder=1)
    arrow(ax, 15.95, 4.55, 15.55, 4.55, color="#24598a")
    ax.text(15.62, 12.4, "气\n候", fontproperties=font, fontsize=8.5, color="#24598a", va="center", ha="left")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_steps() -> None:
    print("锁定流程。清洗在填补之前；正式物候不用双 logistic，也不用 SG9。")
    for step in STEPS:
        print(f"\n{step['id']}  {step['name']}")
        for script in step["scripts"]:
            print(f"    {script}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--print", dest="do_print", action="store_true")
    args = parser.parse_args()
    if args.do_print:
        print_steps()
    font = load_font()
    render(args.out, font)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
