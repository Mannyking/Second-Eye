from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from utils.report_loader import (
    build_class_distribution_df,
    build_labels_per_image_df,
    build_loss_df,
    build_per_class_df,
    build_runs_df,
    build_thresholds_df,
    build_top_pairs_df,
    get_best_run,
    get_latest_run,
    load_dataset_report,
    load_training_runs,
)


st.set_page_config(page_title="Inventory Reports", page_icon=":bar_chart:", layout="wide")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-start: #f5f7ff;
                --bg-end: #edf7f2;
                --ink: #1f2937;
                --muted: #4b5563;
                --accent: #0f766e;
            }
            .stApp {
                background: radial-gradient(circle at 0% 0%, var(--bg-start), var(--bg-end) 65%);
                color: var(--ink);
            }
            .report-chip {
                display: inline-block;
                border: 1px solid rgba(15, 118, 110, 0.25);
                border-radius: 999px;
                padding: 0.2rem 0.65rem;
                margin-right: 0.4rem;
                margin-bottom: 0.5rem;
                background: rgba(15, 118, 110, 0.06);
                color: var(--accent);
                font-size: 0.85rem;
                font-weight: 600;
            }
            .report-subtle {
                color: var(--muted);
                font-size: 0.95rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def _load_all_reports(base_dir: Path) -> tuple[dict, list[dict]]:
    reports_dir = base_dir / "reports"
    dataset_report = load_dataset_report(reports_dir / "dataset_report_bag.json")
    training_runs = load_training_runs(reports_dir / "training_summary.json")
    return dataset_report, training_runs


def _format_ts(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return "N/A"
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def _render_dataset_tab(dataset_report: dict) -> None:
    st.subheader("Dataset Profile")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{int(dataset_report.get('rows', 0)):,}")
    col2.metric("Classes", str(dataset_report.get("num_classes", "N/A")))
    col3.metric("Missing Images", str(dataset_report.get("missing_images", "N/A")))
    col4.metric("Corrupt Images", str(dataset_report.get("corrupt_images", "N/A")))

    classes = dataset_report.get("classes", [])
    if classes:
        chips = "".join(f"<span class='report-chip'>{c}</span>" for c in classes)
        st.markdown(chips, unsafe_allow_html=True)

    class_df = build_class_distribution_df(dataset_report)
    labels_df = build_labels_per_image_df(dataset_report)
    top_pairs_df = build_top_pairs_df(dataset_report)

    left, right = st.columns(2)
    with left:
        st.markdown("**Class Distribution**")
        if class_df.empty:
            st.info("No class distribution data available.")
        else:
            chart = (
                alt.Chart(class_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    x=alt.X("count:Q", title="Count"),
                    y=alt.Y("class:N", sort="-x", title="Class"),
                    tooltip=["class", "count", alt.Tooltip("pct:Q", format=".2%")],
                    color=alt.value("#0f766e"),
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

    with right:
        st.markdown("**Labels Per Image**")
        if labels_df.empty:
            st.info("No labels-per-image distribution found.")
        else:
            chart = (
                alt.Chart(labels_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("labels_per_image:O", title="Labels per image"),
                    y=alt.Y("count:Q", title="Count"),
                    tooltip=["labels_per_image", "count", alt.Tooltip("pct:Q", format=".2%")],
                    color=alt.value("#1d4ed8"),
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown("**Top Co-occurring Pairs**")
    if top_pairs_df.empty:
        st.info("No co-occurrence data available.")
    else:
        chart = (
            alt.Chart(top_pairs_df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("pair:N", sort="-x", title="Pair"),
                tooltip=["pair", "count", alt.Tooltip("pct:Q", format=".2%")],
                color=alt.value("#be185d"),
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)


def _render_training_tab(training_runs: list[dict]) -> None:
    st.subheader("Training Run History")
    runs_df = build_runs_df(training_runs)
    best_run = get_best_run(training_runs, metric="final_macro_f1")
    latest_run = get_latest_run(training_runs)

    total_runs = len(training_runs)
    best_macro = float(best_run.get("final_macro_f1", 0.0)) if best_run else 0.0
    best_micro = float(best_run.get("final_micro_f1", 0.0)) if best_run else 0.0
    runs_with_thresholds = sum(1 for run in training_runs if run.get("best_thresholds"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", str(total_runs))
    col2.metric("Best Macro F1", f"{best_macro:.4f}")
    col3.metric("Best Micro F1", f"{best_micro:.4f}")
    col4.metric("Runs with Thresholds", f"{runs_with_thresholds}/{total_runs}")

    if latest_run:
        latest_text = (
            f"Latest run: `{latest_run.get('timestamp', 'N/A')}` "
            f"| macro={float(latest_run.get('final_macro_f1', 0.0)):.4f} "
            f"| micro={float(latest_run.get('final_micro_f1', 0.0)):.4f}"
        )
        st.markdown(f"<p class='report-subtle'>{latest_text}</p>", unsafe_allow_html=True)

    if runs_df.empty:
        st.warning("No training runs found.")
        return

    plot_df = runs_df.melt(
        id_vars=["run_index", "timestamp"],
        value_vars=["macro_f1", "micro_f1"],
        var_name="metric",
        value_name="score",
    )

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("run_index:Q", title="Run #"),
            y=alt.Y("score:Q", title="F1 Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "metric:N",
                title="Metric",
                scale=alt.Scale(domain=["macro_f1", "micro_f1"], range=["#0f766e", "#1d4ed8"]),
            ),
            tooltip=[
                alt.Tooltip("run_index:Q", title="Run"),
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("score:Q", title="Score", format=".4f"),
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)

    display_df = runs_df.copy()
    display_df["timestamp"] = display_df["timestamp"].apply(_format_ts)
    st.dataframe(
        display_df[
            [
                "run_index",
                "timestamp",
                "macro_f1",
                "micro_f1",
                "epochs",
                "lr",
                "weight_decay",
                "dropout",
                "batch_size",
                "has_thresholds",
                "val_setup",
                "val_test_csv_path",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Config Impact Analysis**")
    impact_df = runs_df.copy()
    impact_df["dropout_label"] = impact_df["dropout"].apply(
        lambda x: "None" if pd.isna(x) else f"{float(x):.2f}"
    )
    impact_df["thresholds_label"] = impact_df["has_thresholds"].map({True: "thresholds_on", False: "thresholds_off"})

    grouped_configs = (
        impact_df.groupby(
            ["epochs", "lr", "weight_decay", "dropout_label", "thresholds_label", "val_setup"],
            dropna=False,
        )
        .agg(
            runs=("run_index", "count"),
            avg_macro_f1=("macro_f1", "mean"),
            max_macro_f1=("macro_f1", "max"),
            avg_micro_f1=("micro_f1", "mean"),
        )
        .reset_index()
        .sort_values(["max_macro_f1", "avg_macro_f1"], ascending=False)
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Learning Rate vs Macro F1**")
        scatter = (
            alt.Chart(impact_df)
            .mark_circle(size=90, opacity=0.85)
            .encode(
                x=alt.X("lr:Q", title="Learning Rate", scale=alt.Scale(type="log")),
                y=alt.Y("macro_f1:Q", title="Macro F1", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("dropout_label:N", title="Dropout"),
                shape=alt.Shape("thresholds_label:N", title="Thresholds"),
                tooltip=[
                    "run_index",
                    alt.Tooltip("timestamp:T", title="Timestamp"),
                    alt.Tooltip("macro_f1:Q", format=".4f"),
                    alt.Tooltip("micro_f1:Q", format=".4f"),
                    "epochs",
                    "weight_decay",
                    "dropout_label",
                    "thresholds_label",
                    "val_setup",
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(scatter, use_container_width=True)

    with right:
        st.markdown("**Average Macro F1 by Config Component**")
        component = st.selectbox(
            "Compare by",
            options=["epochs", "dropout_label", "thresholds_label", "val_setup", "weight_decay"],
            key="impact_component",
        )
        component_df = (
            impact_df.groupby(component, dropna=False)
            .agg(avg_macro_f1=("macro_f1", "mean"), runs=("run_index", "count"))
            .reset_index()
            .sort_values("avg_macro_f1", ascending=False)
        )
        bar = (
            alt.Chart(component_df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("avg_macro_f1:Q", title="Avg Macro F1", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y(f"{component}:N", sort="-x", title=component),
                tooltip=[component, alt.Tooltip("avg_macro_f1:Q", format=".4f"), "runs"],
                color=alt.value("#0f766e"),
            )
            .properties(height=320)
        )
        st.altair_chart(bar, use_container_width=True)

    st.markdown("**Top Config Bundles by Score**")
    st.dataframe(
        grouped_configs.head(12),
        use_container_width=True,
        hide_index=True,
        column_config={
            "avg_macro_f1": st.column_config.NumberColumn(format="%.4f"),
            "max_macro_f1": st.column_config.NumberColumn(format="%.4f"),
            "avg_micro_f1": st.column_config.NumberColumn(format="%.4f"),
        },
    )


def _render_run_drilldown_tab(training_runs: list[dict]) -> None:
    st.subheader("Run Drilldown")
    if not training_runs:
        st.info("No runs available.")
        return

    run_labels = [
        f"{run.get('timestamp', 'N/A')} | macro={float(run.get('final_macro_f1', 0.0)):.4f} | micro={float(run.get('final_micro_f1', 0.0)):.4f}"
        for run in training_runs
    ]
    selected_label = st.selectbox("Select run", options=run_labels, index=len(run_labels) - 1)
    run = training_runs[run_labels.index(selected_label)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Epochs", str(run.get("epochs", "N/A")))
    c2.metric("LR", str(run.get("learning_rate", "N/A")))
    c3.metric("Weight Decay", str(run.get("weight_decay", "N/A")))
    c4.metric("Dropout", str(run.get("dropout", "None")))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Macro F1", f"{float(run.get('final_macro_f1', 0.0)):.4f}")
    c6.metric("Micro F1", f"{float(run.get('final_micro_f1', 0.0)):.4f}")
    c7.metric("Train Size", str(run.get("train_size", "N/A")))
    c8.metric("Val Size", str(run.get("val_size", "N/A")))

    loss_df = build_loss_df(run)
    per_class_df = build_per_class_df(run)
    thresholds_df = build_thresholds_df(run)

    left, right = st.columns(2)
    with left:
        st.markdown("**Loss Curves**")
        if loss_df.empty:
            st.info("No loss data available for this run.")
        else:
            loss_chart = (
                alt.Chart(loss_df)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X("epoch:Q", title="Epoch"),
                    y=alt.Y("loss:Q", title="Loss"),
                    color=alt.Color(
                        "split:N",
                        title="Split",
                        scale=alt.Scale(domain=["train", "val"], range=["#0f766e", "#f59e0b"]),
                    ),
                    tooltip=["epoch", "split", alt.Tooltip("loss:Q", format=".4f")],
                )
                .properties(height=320)
            )
            st.altair_chart(loss_chart, use_container_width=True)

    with right:
        st.markdown("**Per-Class F1**")
        if per_class_df.empty:
            st.info("No per-class metrics available.")
        else:
            f1_chart = (
                alt.Chart(per_class_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    x=alt.X("f1:Q", title="F1", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("class:N", sort="-x", title="Class"),
                    tooltip=["class", alt.Tooltip("f1:Q", format=".4f"), "support"],
                    color=alt.value("#1d4ed8"),
                )
                .properties(height=320)
            )
            st.altair_chart(f1_chart, use_container_width=True)

    if not per_class_df.empty:
        st.markdown("**Per-Class Metrics Table**")
        st.dataframe(
            per_class_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "precision": st.column_config.NumberColumn(format="%.4f"),
                "recall": st.column_config.NumberColumn(format="%.4f"),
                "f1": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    st.markdown("**Best Thresholds**")
    if thresholds_df.empty:
        st.info("No tuned thresholds recorded for this run.")
    else:
        st.dataframe(
            thresholds_df,
            use_container_width=True,
            hide_index=True,
            column_config={"threshold": st.column_config.NumberColumn(format="%.2f")},
        )


def main() -> None:
    _inject_styles()

    project_root = Path(__file__).resolve().parents[1]
    dataset_report_path = project_root / "reports" / "dataset_report_bag.json"
    training_summary_path = project_root / "reports" / "training_summary.json"

    st.title("Inventory Reports")
    st.caption("Visual summary of dataset quality and model training progression.")

    missing_files = [
        str(path.name)
        for path in [dataset_report_path, training_summary_path]
        if not path.exists()
    ]
    if missing_files:
        st.error(f"Missing report files: {', '.join(missing_files)}")
        return

    dataset_report, training_runs = _load_all_reports(project_root)
    tabs = st.tabs(["Dataset", "Training", "Run Drilldown"])

    with tabs[0]:
        _render_dataset_tab(dataset_report)
    with tabs[1]:
        _render_training_tab(training_runs)
    with tabs[2]:
        _render_run_drilldown_tab(training_runs)


if __name__ == "__main__":
    main()
