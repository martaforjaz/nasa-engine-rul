"""Vector paper figures informed by Chen Liu's figures4papers design guide."""
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parent
BLUE, PLUM, INK, GRAY = "#0F4D92", "#915883", "#252c36", "#767676"
PALE = "#eaf0f7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .8, "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "legend.frameon": False, "svg.fonttype": "none", "svg.hashsalt": "nasa-rul",
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def save(fig, name):
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    for fmt, meta in [("svg", {"Date": None}), ("pdf", {"CreationDate": None})]:
        path = folder / f"{name}.{fmt}"
        fig.savefig(path, metadata=meta)
        if fmt == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_bytes(("\n".join(line.rstrip() for line in lines) + "\n").encode("utf-8"))
    if preview := os.environ.get("FIGURE_PREVIEW"):
        fig.savefig(Path(preview) / f"{name}.png", dpi=150)
    plt.close(fig)


def badge(ax, x, label):
    ax.add_patch(FancyBboxPatch((x, 2.08), 2.26, .38, boxstyle="round,pad=0,rounding_size=.19",
                                linewidth=0, facecolor=BLUE))
    ax.text(x+1.13, 2.27, label, ha="center", va="center", color="white", weight="bold")


def turbine(ax, x):
    ax.add_patch(Polygon([(x+.33, .83), (x+1.4, .9), (x+1.92, 1.04),
                         (x+1.92, 1.45), (x+1.4, 1.62), (x+.33, 1.7)],
                         closed=True, facecolor=PALE, edgecolor=BLUE, lw=1.2))
    ax.add_patch(Circle((x+.7, 1.27), .43, facecolor="white", edgecolor=BLUE, lw=1.2))
    for angle in np.linspace(0, 2*np.pi, 9, endpoint=False):
        dx, dy = np.cos(angle), np.sin(angle)
        ax.plot([x+.7+.105*dx, x+.7+.34*dx-.07*dy],
                [1.27+.105*dy, 1.27+.34*dy+.07*dx], color=BLUE, lw=1.5)
    ax.add_patch(Circle((x+.7, 1.27), .09, facecolor=PLUM, edgecolor="none"))


def telemetry(ax, x):
    ax.add_patch(FancyBboxPatch((x+.27, .84), 1.72, .86, boxstyle="round,pad=.03",
                                facecolor=PALE, edgecolor="#c1cfdf", lw=.8))
    time = np.linspace(x+.39, x+1.86, 21)
    signal = np.sin(np.arange(21)*1.6)*.12 + np.linspace(-.06, .06, 21)
    ax.plot(time, 1.37+signal, color=BLUE, lw=1.3)
    ax.plot(time, 1.14-signal*.63, color=PLUM, lw=1.1)
    ax.plot([x+1.26, x+1.26], [.88, 1.66], ls=":", color=GRAY, lw=.8)


def tree(ax, x):
    nodes = [(x+1.13, 1.72), (x+.68, 1.35), (x+1.58, 1.35),
             (x+.42, .96), (x+.94, .96), (x+1.32, .96), (x+1.84, .96)]
    for parent, child in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
        px, py = nodes[parent]
        cx, cy = nodes[child]
        ax.plot([px, cx], [py, cy], color="#8a9cb3", lw=1)
    for i, node in enumerate(nodes):
        ax.add_patch(Circle(node, .095, facecolor=BLUE if i < 3 else PLUM, edgecolor="white", lw=.8))


def endpoint(ax, x):
    ax.plot([x+.36, x+1.43], [1.2, 1.2], color=BLUE, lw=2)
    ax.plot([x+1.43, x+1.93], [1.2, 1.2], ls="--", color=PLUM, lw=2)
    ax.scatter([x+1.43, x+1.93], [1.2, 1.2], s=[35, 24], c=[BLUE, PLUM], zorder=3)
    ax.text(x+1.68, 1.42, "RUL", ha="center", color=PLUM, weight="bold")


def pipeline():
    fig, ax = plt.subplots(figsize=(8.6, 2.25))
    fig.subplots_adjust(left=.015, right=.985, bottom=.07, top=.98)
    ax.set(xlim=(0, 10), ylim=(.15, 2.55))
    ax.set_axis_off()
    stages = [
        ("I   Engine telemetry", "C-MAPSS FD001", turbine),
        ("II   Feature window", "Trailing 20 cycles", telemetry),
        ("III   Fit models", "Training engines", tree),
        ("IV   Evaluate", "Held-out endpoints", endpoint),
    ]
    for i, (heading, note, icon) in enumerate(stages):
        x = .07 + 2.52*i
        badge(ax, x, heading)
        icon(ax, x)
        ax.text(x+1.13, .49, note, ha="center", va="center", color=GRAY)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x+2.09, 1.23), (x+2.48, 1.23),
                                         arrowstyle="-|>", mutation_scale=11, color=BLUE, lw=1))
    save(fig, "pipeline")


def results():
    metrics = json.loads((ROOT / "evidence/metrics.json").read_text(encoding="utf-8"))
    with (ROOT / "evidence/predictions.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    actual = np.array([float(r["actual_rul"]) for r in rows])
    predicted = np.array([float(r["sensor_model"]) for r in rows])
    fig, (summary, parity, error) = plt.subplots(1, 3, figsize=(8.6, 3.05))
    for panel, ax in zip("abc", (summary, parity, error)):
        ax.text(-.18, 1.08, panel, transform=ax.transAxes, weight="bold", fontsize=13)
    for i, (key, color) in enumerate([("age_baseline", "#c4cbd5"), ("sensor_model", BLUE)]):
        summary.bar(np.arange(2)+(i-.5)*.32,
                    [metrics[key]["mae"], metrics[key]["rmse"]], .32,
                    color=color, edgecolor=INK, linewidth=.65)
    summary.set(xticks=[0, 1], xticklabels=["MAE", "RMSE"], ylabel="Error (cycles)", ylim=(0, 47))
    summary.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#c4cbd5", BLUE)],
                   ["Age baseline", "Sensor model"], fontsize=8, loc="upper left", borderaxespad=.15)
    limit = max(160, int(np.ceil(max(actual.max(), predicted.max()) / 40) * 40))
    parity.plot([0, limit], [0, limit], color=GRAY, lw=.9, ls="--", zorder=0)
    parity.scatter(actual, predicted, s=16, color=BLUE, alpha=.70, linewidths=.25, edgecolors="white")
    parity.set(xlim=(0, limit), ylim=(0, limit), xlabel="Actual RUL (cycles)",
               ylabel="Predicted RUL (cycles)", aspect="equal")
    rng = np.random.default_rng(42)
    for i, mask in enumerate([actual < 50, (actual >= 50) & (actual < 100), actual >= 100]):
        absolute = np.abs(predicted[mask]-actual[mask])
        error.scatter(i+rng.uniform(-.15, .15, len(absolute)), absolute,
                      s=13, color=BLUE, alpha=.48, linewidths=0)
        error.hlines(absolute.mean(), i-.23, i+.23, color=PLUM, linewidth=2.1)
    error.set(xticks=[0, 1, 2], xticklabels=["0–49", "50–99", "100+"],
              xlabel="Actual RUL (cycles)", ylabel="Absolute error (cycles)", ylim=(0, None))
    error.legend(handles=[Line2D([0], [0], color=PLUM, lw=2, label="Mean")],
                 fontsize=8, loc="upper left", borderaxespad=.1)
    fig.subplots_adjust(left=.075, right=.988, bottom=.22, top=.89, wspace=.50)
    save(fig, "results")


if __name__ == "__main__":
    pipeline()
    results()
