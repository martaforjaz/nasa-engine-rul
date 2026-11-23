"""Evaluate C-MAPSS FD001 at the final observed cycle of each test engine."""
import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

ROOT = Path(__file__).resolve().parent
SOURCE = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"


def read(archive, filename):
    member, = [entry for entry in archive.namelist() if Path(entry).name == filename]
    return np.loadtxt(BytesIO(archive.read(member)))


def trailing_features(rows):
    features, labels, endpoints = [], [], []
    for unit in np.unique(rows[:, 0]):
        series = rows[rows[:, 0] == unit]
        if not np.array_equal(series[:, 1], np.arange(1, len(series) + 1)):
            raise ValueError(f"Unexpected cycle order for engine {unit}")
        for i, row in enumerate(series):
            window = series[max(0, i - 19):i + 1, 2:]
            features.append(np.concatenate(([row[1]], row[2:], window.mean(axis=0), window.std(axis=0))))
            labels.append(len(series) - i - 1)
        endpoints.append(len(features) - 1)
    return np.asarray(features), np.asarray(labels), np.asarray(endpoints)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ROOT / "data/CMAPSSData.zip")
    args = parser.parse_args()
    if not args.zip.exists():
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE, timeout=90) as stream:
            args.zip.write_bytes(stream.read())
    content = args.zip.read_bytes()
    with ZipFile(BytesIO(content)) as archive:
        train = read(archive, "train_FD001.txt")
        test = read(archive, "test_FD001.txt")
        truth = read(archive, "RUL_FD001.txt").ravel()
    train_x, train_y, train_ends = trailing_features(train)
    test_x, _, test_ends = trailing_features(test)
    if len(test_ends) != len(truth):
        raise ValueError("Test engines and supplied labels do not align.")
    # The age baseline uses the median run-to-failure cycle in training.
    baseline = np.maximum(0, np.median(train_x[train_ends, 0]) - test_x[test_ends, 0])
    model = HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15,
                                          random_state=42, early_stopping=False)
    model.fit(train_x, train_y)
    predicted = np.maximum(0, model.predict(test_x[test_ends]))
    metrics = {name: {"mae": round(mean_absolute_error(truth, result), 4),
                      "rmse": round(root_mean_squared_error(truth, result), 4)}
               for name, result in [("age_baseline", baseline), ("sensor_model", predicted)]}
    metrics.update(train_engines=len(train_ends), test_engines=len(test_ends),
                   archive_sha256=sha256(content).hexdigest())
    out = ROOT / "evidence"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["engine", "observed_cycles", "actual_rul", "age_baseline", "sensor_model"])
        for index, (end, actual, age, estimate) in enumerate(zip(test_ends, truth, baseline, predicted), 1):
            writer.writerow([index, int(test_x[end, 0]), int(actual), round(age, 5), round(estimate, 5)])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
