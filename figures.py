"""Regenerate the two README figures from evaluation outputs."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
NAVY, PLUM, GRAY = "#102b59", "#915883", "#506078"


def text(x, y, content, size=18, color=NAVY, bold=False):
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{700 if bold else 400}">{content}</text>')


def save(title, parts, height):
    (ROOT / "figures").mkdir(exist_ok=True)
    with (ROOT / "figures" / f"{title}.svg").open("w", encoding="utf-8", newline="\n") as file:
        file.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
                   f'viewBox="0 0 960 {height}" font-family="Arial,sans-serif" role="img">'
                   f'<title>{title}</title><rect width="960" height="{height}" fill="white"/>'
                   + "".join(parts) + "</svg>\n")


metrics = json.loads((ROOT / "evidence/metrics.json").read_text(encoding="utf-8"))
stages = [
    ("I. Ingest", "NASA C-MAPSS FD001", "Simulated engine cycles"),
    ("II. Prepare", "20-cycle trailing features", "No future readings"),
    ("III. Fit", f'{metrics["train_engines"]} training engines', "Age baseline + boosting"),
    ("IV. Evaluate", f'{metrics["test_engines"]} test engines', "One prediction per engine"),
]
parts = [text(28, 50, "From engine telemetry to remaining-life estimates", 30, bold=True)]
for i, (heading, line, detail) in enumerate(stages):
    x = 28 + i * 234
    parts.extend([f'<rect x="{x}" y="86" width="211" height="43" rx="21" fill="{NAVY}"/>',
                  text(x+14, 114, heading, color="white", bold=True),
                  text(x+7, 166, line, 17, bold=True), text(x+7, 193, detail, 16, GRAY)])
    if i < 3:
        parts.append(text(x+212, 168, "→", 19, PLUM, True))
save("pipeline", parts, 230)

with (ROOT / "evidence/predictions.csv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))
parts = [text(28, 50, "Sensor features improved the FD001 test error", 30, bold=True),
         text(28, 80, "Mean absolute error in cycles · 100 held-out engine endpoints", 17, GRAY)]
scores = [(name, metrics[key]["mae"]) for name, key in
          [("Age baseline", "age_baseline"), ("Sensor model", "sensor_model")]]
for i, (name, value) in enumerate(scores):
    y = 114 + i * 49
    color = (NAVY, PLUM)[i]
    parts += [text(28, y+19, name, bold=True),
              f'<rect x="240" y="{y}" width="{value*16:.2f}" height="24" rx="3" fill="{color}"/>',
              text(252+value*16, y+19, f"{value:.2f}", color=color, bold=True)]
parts += [text(28, 238, "Sensor-model error by actual RUL", 22, bold=True)]
for i, (low, high, label) in enumerate([(0, 50, "0–49 cycles"), (50, 100, "50–99 cycles"), (100, float("inf"), "100+ cycles")]):
    group = [row for row in rows if low <= int(row["actual_rul"]) < high]
    error = mean(abs(int(r["actual_rul"]) - float(r["sensor_model"])) for r in group)
    x = 30 + i * 310
    parts += [text(x, 282, label, 17, GRAY), text(x, 316, f"{error:.2f} MAE · n={len(group)}", 20, PLUM, True)]
save("results", parts, 350)
