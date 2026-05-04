import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_report(path: Path) -> dict[str, Any]:
    return dict(_read_json(path))


def load_training_runs(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError("Training summary JSON must be a list of run records.")
    return [dict(item) for item in data]


def build_class_distribution_df(dataset_report: dict[str, Any]) -> pd.DataFrame:
    class_distribution = dataset_report.get("class_distribution", {})
    rows = [
        {"class": class_name, "count": int(stats.get("count", 0)), "pct": float(stats.get("pct", 0.0))}
        for class_name, stats in class_distribution.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["class", "count", "pct"])
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def build_labels_per_image_df(dataset_report: dict[str, Any]) -> pd.DataFrame:
    distribution = dataset_report.get("labels_per_image", {}).get("distribution", {})
    rows = [
        {
            "labels_per_image": int(num_labels),
            "count": int(stats.get("count", 0)),
            "pct": float(stats.get("pct", 0.0)),
        }
        for num_labels, stats in distribution.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["labels_per_image", "count", "pct"])
    return pd.DataFrame(rows).sort_values("labels_per_image", ascending=True)


def build_top_pairs_df(dataset_report: dict[str, Any], limit: int = 12) -> pd.DataFrame:
    top_pairs = dataset_report.get("cooccurrence", {}).get("top_pairs", [])
    rows = []
    for item in top_pairs[:limit]:
        pair = item.get("pair", [])
        pair_label = " + ".join(pair) if isinstance(pair, list) else str(pair)
        rows.append(
            {
                "pair": pair_label,
                "count": int(item.get("count", 0)),
                "pct": float(item.get("pct", 0.0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["pair", "count", "pct"])
    return pd.DataFrame(rows)


def build_runs_df(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "timestamp": run.get("timestamp"),
                "macro_f1": float(run.get("final_macro_f1", 0.0)),
                "micro_f1": float(run.get("final_micro_f1", 0.0)),
                "epochs": int(run.get("epochs", 0)),
                "lr": float(run.get("learning_rate", 0.0)),
                "weight_decay": float(run.get("weight_decay", 0.0)),
                "dropout": run.get("dropout"),
                "batch_size": run.get("batch_size"),
                "has_thresholds": bool(run.get("best_thresholds")),
                "val_test_csv_path": run.get("val_test_csv_path"),
                "val_setup": "external_val_test" if run.get("val_test_csv_path") else "train_val_split_only",
                "num_trainable_params": run.get("num_trainable_params"),
                "num_frozen_params": run.get("num_frozen_params"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
        df["run_index"] = df.index + 1
    return df


def get_best_run(runs: list[dict[str, Any]], metric: str = "final_macro_f1") -> dict[str, Any] | None:
    if not runs:
        return None
    return max(runs, key=lambda x: float(x.get(metric, 0.0)))


def get_latest_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    return max(runs, key=lambda x: str(x.get("timestamp", "")))


def build_loss_df(run: dict[str, Any]) -> pd.DataFrame:
    train_losses = run.get("train_losses", []) or []
    val_losses = run.get("val_losses", []) or []
    n = min(len(train_losses), len(val_losses))
    rows = []
    for idx in range(n):
        epoch = idx + 1
        rows.append({"epoch": epoch, "split": "train", "loss": float(train_losses[idx])})
        rows.append({"epoch": epoch, "split": "val", "loss": float(val_losses[idx])})
    if not rows:
        return pd.DataFrame(columns=["epoch", "split", "loss"])
    return pd.DataFrame(rows)


def build_per_class_df(run: dict[str, Any]) -> pd.DataFrame:
    per_class_metrics = run.get("per_class_metrics", []) or []
    rows = []
    for metric in per_class_metrics:
        rows.append(
            {
                "class": metric.get("class"),
                "precision": float(metric.get("precision", 0.0)),
                "recall": float(metric.get("recall", 0.0)),
                "f1": float(metric.get("f1", 0.0)),
                "support": int(metric.get("support", 0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["class", "precision", "recall", "f1", "support"])
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


def build_thresholds_df(run: dict[str, Any]) -> pd.DataFrame:
    thresholds = run.get("best_thresholds", {}) or {}
    rows = [{"class": class_name, "threshold": float(value)} for class_name, value in thresholds.items()]
    if not rows:
        return pd.DataFrame(columns=["class", "threshold"])
    return pd.DataFrame(rows).sort_values("threshold", ascending=False)
