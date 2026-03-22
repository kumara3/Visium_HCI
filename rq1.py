import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Tumor Atlas Study Prototype",
    layout="wide"
)


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
@st.cache_data
def load_data():
    preprocess_dir = Path("Preprocess_data")
    spots = pd.read_parquet(preprocess_dir / "spots.parquet")
    img = Image.open(preprocess_dir / "tissue_hires_image.png")
    return spots, img


@st.cache_data
def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


spots, img = load_data()
spots = spots.copy()

W, H = img.size


# ------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------
def init_session_state():
    defaults = {
        "task_active": False,
        "task_name": None,
        "task_condition": None,
        "task_start_time": None,
        "task_elapsed_seconds": 0.0,
        "interaction_count": 0,
        "param_change_count": 0,
        "selection_count": 0,
        "last_selection_ids": [],
        "study_logs": [],
        "submitted_tasks": 0,
        "participant_id": "P001",
        "saved_rois": [],
        "roi_counter": 0,

    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def increment_interaction(kind: str, amount: int = 1):
    st.session_state.interaction_count += amount


def increment_param_change():
    st.session_state.param_change_count += 1
    increment_interaction("param_change", 1)


def start_task(task_name: str, condition: str):
    st.session_state.task_active = True
    st.session_state.task_name = task_name
    st.session_state.task_condition = condition
    st.session_state.task_start_time = time.time()
    st.session_state.task_elapsed_seconds = 0.0
    st.session_state.interaction_count = 0
    st.session_state.param_change_count = 0
    st.session_state.selection_count = 0
    st.session_state.last_selection_ids = []


def stop_task():
    if st.session_state.task_active and st.session_state.task_start_time is not None:
        st.session_state.task_elapsed_seconds = time.time() - st.session_state.task_start_time
    st.session_state.task_active = False


def current_elapsed_time() -> float:
    if st.session_state.task_active and st.session_state.task_start_time is not None:
        return time.time() - st.session_state.task_start_time
    return st.session_state.task_elapsed_seconds


def compute_region_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "n_spots": 0,
            "mean_tumor_probability": np.nan,
            "mean_tumor_gradient": np.nan,
            "mean_uncertainty": np.nan,
            "heterogeneity_score": np.nan,
            "dominant_region": "None",
            "tumor_fraction_estimate": np.nan,
        }

    region_counts = df["region"].value_counts()
    dominant_region = region_counts.idxmax() if not region_counts.empty else "None"

    # Simple heterogeneity proxy:
    # combine std of tumor probability + diversity of region labels
    prob_std = df["tumor_probability"].std(ddof=0) if len(df) > 1 else 0.0
    region_entropy = 0.0
    probs = region_counts / region_counts.sum()
    if len(probs) > 1:
        region_entropy = -(probs * np.log2(probs + 1e-12)).sum()

    heterogeneity_score = 0.6 * prob_std + 0.4 * region_entropy

    # Tumor fraction estimate: proportion above tumor threshold surrogate
    tumor_fraction_estimate = (df["region"] == "tumor_core").mean()

    return {
        "n_spots": int(len(df)),
        "mean_tumor_probability": float(df["tumor_probability"].mean()),
        "mean_tumor_gradient": float(df["tumor_gradient"].mean()),
        "mean_uncertainty": float(df["uncertainty"].mean()) if "uncertainty" in df.columns else np.nan,
        "heterogeneity_score": float(heterogeneity_score),
        "dominant_region": dominant_region,
        "tumor_fraction_estimate": float(tumor_fraction_estimate),
    }

def save_current_roi(spots_df: pd.DataFrame, image: Image.Image):
    selected_ids = st.session_state.last_selection_ids
    if not selected_ids:
        return False

    roi_df = spots_df[spots_df["spot_id"].isin(selected_ids)].copy()
    if roi_df.empty:
        return False

    roi_stats = compute_region_stats(roi_df)

    x_min, x_max = roi_df["x"].min(), roi_df["x"].max()
    y_min, y_max = roi_df["y"].min(), roi_df["y"].max()
    patch = safe_crop_image(image, x_min, y_min, x_max, y_max, pad=50)

    st.session_state.roi_counter += 1
    roi_name = f"ROI {st.session_state.roi_counter}"

    roi_record = {
        "roi_name": roi_name,
        "spot_ids": list(map(str, roi_df["spot_id"].tolist())),
        "n_spots": len(roi_df),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "dominant_region": roi_stats["dominant_region"],
        "mean_tumor_probability": roi_stats["mean_tumor_probability"],
        "mean_tumor_gradient": roi_stats["mean_tumor_gradient"],
        "mean_uncertainty": roi_stats["mean_uncertainty"],
        "heterogeneity_score": roi_stats["heterogeneity_score"],
        "tumor_fraction_estimate": roi_stats["tumor_fraction_estimate"],
        "roi_df": roi_df,
        "patch": patch,
    }

    st.session_state.saved_rois.append(roi_record)
    return True


def clear_saved_rois():
    st.session_state.saved_rois = []
    st.session_state.roi_counter = 0


def build_roi_comparison_table(saved_rois: list) -> pd.DataFrame:
    rows = []
    for roi in saved_rois:
        rows.append({
            "ROI": roi["roi_name"],
            "Spots": roi["n_spots"],
            "Dominant region": roi["dominant_region"],
            "Mean tumor prob": roi["mean_tumor_probability"],
            "Mean gradient": roi["mean_tumor_gradient"],
            "Mean uncertainty": roi["mean_uncertainty"],
            "Heterogeneity": roi["heterogeneity_score"],
            "Tumor fraction est": roi["tumor_fraction_estimate"],
        })
    return pd.DataFrame(rows)

def safe_crop_image(image: Image.Image, x_min, y_min, x_max, y_max, pad=40):
    x0 = max(int(x_min - pad), 0)
    y0 = max(int(y_min - pad), 0)
    x1 = min(int(x_max + pad), image.size[0])
    y1 = min(int(y_max + pad), image.size[1])
    if x1 <= x0 or y1 <= y0:
        return None
    return image.crop((x0, y0, x1, y1))


def assign_regions(df: pd.DataFrame, tumor_threshold, normal_threshold, gradient_threshold):
    out = df.copy()
    out["region"] = "other"

    out.loc[out["tumor_probability"] > tumor_threshold, "region"] = "tumor_core"

    adjacent_normal = (
        (out["tumor_probability"] < normal_threshold) &
        (out["tumor_gradient"] > gradient_threshold / 2)
    )
    out.loc[adjacent_normal, "region"] = "adjacent_normal"

    invasion = (
        (out["tumor_gradient"] > gradient_threshold) &
        (out["tumor_probability"].between(normal_threshold, tumor_threshold))
    )
    out.loc[invasion, "region"] = "invasion_front"

    return out


def compute_candidates(df: pd.DataFrame, top_k=8):
    scored = df.copy()

    # Hotspot score: tumor probability + proliferation signal - uncertainty penalty
    scored["hotspot_score"] = (
        0.60 * scored["tumor_probability"] +
        0.25 * scored.get("proliferation_index", 0) +
        0.15 * scored.get("immune_score", 0) -
        0.20 * scored.get("uncertainty", 0)
    )

    # Heterogeneity candidate score: high gradient + uncertainty + moderate tumor mix
    mid_prob = 1.0 - np.abs(scored["tumor_probability"] - 0.5) * 2
    scored["heterogeneity_score_local"] = (
        0.55 * scored["tumor_gradient"] +
        0.25 * mid_prob +
        0.20 * scored.get("uncertainty", 0)
    )

    hotspot_candidates = scored.nlargest(top_k, "hotspot_score")[
        ["spot_id", "x", "y", "hotspot_score", "tumor_probability", "tumor_gradient", "uncertainty", "region"]
    ].copy()

    hetero_candidates = scored.nlargest(top_k, "heterogeneity_score_local")[
        ["spot_id", "x", "y", "heterogeneity_score_local", "tumor_probability", "tumor_gradient", "uncertainty", "region"]
    ].copy()

    return hotspot_candidates, hetero_candidates


# ------------------------------------------------------------
# Title and study controls
# ------------------------------------------------------------
st.title("Tumor Atlas HCI Study Prototype")
st.caption(
    "Supports RQ1 with baseline vs interactive condition, ROI summaries, task timing, "
    "selection logging, and confidence capture."
)

with st.sidebar:
    st.header("Study Setup")

    participant_id = st.text_input("Participant ID", value=st.session_state.participant_id)
    st.session_state.participant_id = participant_id

    interface_mode = st.radio(
        "Condition",
        ["Baseline", "Tumor Atlas"],
        help="Baseline = simpler viewer. Tumor Atlas = richer interactive condition."
    )

    task_name = st.selectbox(
        "Task Prompt",
        [
            "Find the highest tumor hotspot",
            "Find one spatially heterogeneous region",
            "Find hotspot + heterogeneous region"
        ]
    )

    st.markdown("---")
    st.header("Region Parameters")

    tumor_threshold = st.slider(
        "Tumor core probability threshold",
        0.0, 1.0, 0.7,
        on_change=increment_param_change
    )

    normal_threshold = st.slider(
        "Normal probability threshold",
        0.0, 1.0, 0.3,
        on_change=increment_param_change
    )

    gradient_threshold = st.slider(
        "Invasion front gradient threshold",
        0.0, 1.0, 0.25,
        on_change=increment_param_change
    )

    k_neighbors = st.slider(
        "Spatial neighbors",
        3, 20, 6,
        on_change=increment_param_change
    )

    st.markdown("---")

    if not st.session_state.task_active:
        if st.button("Start Task", type="primary"):
            start_task(task_name, interface_mode)
    else:
        if st.button("Stop Task"):
            stop_task()

    st.write(f"Task active: **{st.session_state.task_active}**")
    st.write(f"Elapsed time: **{current_elapsed_time():.1f} sec**")
    st.write(f"Interactions: **{st.session_state.interaction_count}**")
    st.write(f"Parameter changes: **{st.session_state.param_change_count}**")
    st.write(f"Selections: **{st.session_state.selection_count}**")


# ------------------------------------------------------------
# Neighbor graph and gradient computation
# ------------------------------------------------------------
coords = spots[["x", "y"]].values

nbrs = NearestNeighbors(
    n_neighbors=k_neighbors
).fit(coords)

distances, indices = nbrs.kneighbors(coords)

prob = spots["tumor_probability"].values
gradients = []

for i in range(len(spots)):
    neigh = indices[i]
    diff = np.abs(prob[i] - prob[neigh])
    gradients.append(diff.mean())

spots["tumor_gradient"] = gradients
spots = assign_regions(spots, tumor_threshold, normal_threshold, gradient_threshold)

hotspot_candidates, hetero_candidates = compute_candidates(spots, top_k=8)


# ------------------------------------------------------------
# Interface configuration by condition
# ------------------------------------------------------------
if interface_mode == "Baseline":
    color_choices = ["tumor_probability"]
    default_color = "tumor_probability"
    allow_candidate_panels = False
    allow_roi_cards = False
    allow_detail_stats = False
    help_text = "Baseline viewer: simple heatmap-like point display with minimal support."
else:
    color_choices = [
        "region",
        "tumor_probability",
        "tumor_gradient",
        "immune_score",
        "proliferation_index",
        "uncertainty",
        "cell_type"
    ]
    default_color = "region"
    allow_candidate_panels = True
    allow_roi_cards = True
    allow_detail_stats = True
    help_text = "Tumor Atlas: richer exploratory condition with ROI summaries, candidate regions, and linked details."

st.info(help_text)


# ------------------------------------------------------------
# Top-level layout
# ------------------------------------------------------------
left_col, right_col = st.columns([2.2, 1.0], gap="large")


# ------------------------------------------------------------
# Main plot
# ------------------------------------------------------------
with left_col:
    st.subheader("Spatial Viewer")

    color_option = st.selectbox(
        "Color spots by",
        color_choices,
        index=color_choices.index(default_color) if default_color in color_choices else 0,
        on_change=increment_param_change
    )

    hover_fields = [
        "spot_id",
        "tumor_probability",
        "tumor_gradient",
        "uncertainty"
    ]
    for optional_col in ["immune_score", "proliferation_index", "cell_type", "region"]:
        if optional_col in spots.columns and optional_col not in hover_fields:
            hover_fields.append(optional_col)

    fig = px.scatter(
        spots,
        x="x",
        y="y",
        color=color_option,
        hover_data=hover_fields,
        custom_data=["spot_id", "x", "y", "region", "tumor_probability", "tumor_gradient", "uncertainty"],
        render_mode="webgl"
    )

    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        images=[
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=W,
                sizey=H,
                sizing="stretch",
                layer="below"
            )
        ],
        dragmode="lasso" if interface_mode == "Tumor Atlas" else "zoom",
        width=900,
        height=800,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    # Candidate overlays only for Tumor Atlas
    if interface_mode == "Tumor Atlas":
        fig.add_scatter(
            x=hotspot_candidates["x"],
            y=hotspot_candidates["y"],
            mode="markers",
            name="Hotspot candidates",
            marker=dict(symbol="x", size=11, line=dict(width=1)),
            customdata=hotspot_candidates[["spot_id"]].values
        )
        fig.add_scatter(
            x=hetero_candidates["x"],
            y=hetero_candidates["y"],
            mode="markers",
            name="Heterogeneity candidates",
            marker=dict(symbol="diamond-open", size=12, line=dict(width=1)),
            customdata=hetero_candidates[["spot_id"]].values
        )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key="main_plot",
        on_select="rerun",
        selection_mode=["points", "box", "lasso"]
    )

    selected_points = []
    selected_ids = []

    if event is not None and hasattr(event, "selection") and event.selection:
        selected_points = event.selection.get("points", [])
        if selected_points:
            st.session_state.selection_count += 1
            increment_interaction("selection", 1)

            for p in selected_points:
                custom = p.get("customdata", [])
                if custom and len(custom) >= 1:
                    selected_ids.append(custom[0])

    # Deduplicate but preserve order
    seen = set()
    selected_ids = [x for x in selected_ids if not (x in seen or seen.add(x))]
    st.session_state.last_selection_ids = selected_ids

    st.caption(
        "Use box/lasso/point selection to define an ROI. In the Tumor Atlas condition, "
        "the right panel summarizes the selected region."
    )


# ------------------------------------------------------------
# Right panel: candidate support + ROI cards + linked summaries
# ------------------------------------------------------------
with right_col:
    st.subheader("Study Support Panel")

    st.markdown("**Task prompt**")
    st.write(task_name)

    st.markdown("**Current condition**")
    st.write(interface_mode)

    st.markdown("**Live metrics**")
    st.write({
        "elapsed_seconds": round(current_elapsed_time(), 1),
        "interaction_count": st.session_state.interaction_count,
        "parameter_changes": st.session_state.param_change_count,
        "selection_count": st.session_state.selection_count,
    })

    if allow_candidate_panels:
        with st.expander("Candidate hotspot regions", expanded=True):
            st.dataframe(
                hotspot_candidates.rename(columns={
                    "spot_id": "spot",
                    "hotspot_score": "score",
                    "tumor_probability": "tumor_prob",
                    "tumor_gradient": "gradient"
                }),
                use_container_width=True,
                hide_index=True
            )

        with st.expander("Candidate heterogeneous regions", expanded=True):
            st.dataframe(
                hetero_candidates.rename(columns={
                    "spot_id": "spot",
                    "heterogeneity_score_local": "score",
                    "tumor_probability": "tumor_prob",
                    "tumor_gradient": "gradient"
                }),
                use_container_width=True,
                hide_index=True
            )

    selected_df = spots[spots["spot_id"].isin(st.session_state.last_selection_ids)].copy()

    if allow_roi_cards:
        st.markdown("---")
        st.subheader("ROI Cards")

        if selected_df.empty:
            st.warning("No ROI selected yet. Select points on the viewer.")
        else:
            roi_stats = compute_region_stats(selected_df)

            c1, c2 = st.columns(2)
            c1.metric("ROI spots", roi_stats["n_spots"])
            c2.metric("Dominant region", roi_stats["dominant_region"])

            c3, c4 = st.columns(2)
            c3.metric("Mean tumor probability", f"{roi_stats['mean_tumor_probability']:.3f}")
            c4.metric("Mean tumor gradient", f"{roi_stats['mean_tumor_gradient']:.3f}")

            c5, c6 = st.columns(2)
            c5.metric("Mean uncertainty", f"{roi_stats['mean_uncertainty']:.3f}")
            c6.metric("Tumor fraction est.", f"{roi_stats['tumor_fraction_estimate']:.2%}")

            st.metric("Heterogeneity score", f"{roi_stats['heterogeneity_score']:.3f}")

            # Uncertainty interpretation card
            if roi_stats["mean_uncertainty"] >= 0.45:
                st.error("High uncertainty ROI: interpret with caution.")
            elif roi_stats["mean_uncertainty"] >= 0.25:
                st.warning("Moderate uncertainty ROI.")
            else:
                st.success("Low uncertainty ROI.")

            # Simple textual interpretation
            if roi_stats["dominant_region"] == "tumor_core":
                st.markdown("**Interpretation:** ROI is primarily tumor-dominant.")
            elif roi_stats["dominant_region"] == "invasion_front":
                st.markdown("**Interpretation:** ROI suggests a likely boundary / invasive-front region.")
            elif roi_stats["dominant_region"] == "adjacent_normal":
                st.markdown("**Interpretation:** ROI appears mainly adjacent normal tissue.")
            else:
                st.markdown("**Interpretation:** ROI is mixed or less clearly classified.")

    if allow_detail_stats and not selected_df.empty:
        st.markdown("---")
        st.subheader("Linked ROI Details")

        st.markdown("**Selected spots preview**")
        st.dataframe(
            selected_df[
                ["spot_id", "x", "y", "region", "tumor_probability", "tumor_gradient", "uncertainty"]
            ].sort_values("tumor_probability", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("**ROI distributions**")
        stat_view = st.selectbox(
            "Distribution metric",
            ["tumor_probability", "tumor_gradient", "uncertainty"],
            key="roi_distribution_metric"
        )

        hist_fig = px.histogram(
            selected_df,
            x=stat_view,
            nbins=20,
            title=f"ROI distribution: {stat_view}"
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        x_min, x_max = selected_df["x"].min(), selected_df["x"].max()
        y_min, y_max = selected_df["y"].min(), selected_df["y"].max()
        patch = safe_crop_image(img, x_min, y_min, x_max, y_max, pad=50)

        if patch is not None:
            st.markdown("**Zoomed tissue patch for selected ROI**")
            st.image(patch, use_container_width=True)

    st.markdown("---")
    st.subheader("Task Response")

    confidence = st.slider(
        "How confident are you in your selected answer?",
        1, 7, 4,
        help="1 = very unsure, 7 = very confident"
    )

    selected_answer_type = st.multiselect(
        "What did you select?",
        ["Hotspot region", "Heterogeneous region"]
    )

    usability_note = st.text_area(
        "Qualitative note on usability / what helped or made the task harder",
        placeholder="Example: ROI cards made it easier to compare local regions..."
    )

    submit_disabled = (
        len(selected_answer_type) == 0 or
        len(st.session_state.last_selection_ids) == 0
    )

    if st.button("Submit task result", disabled=submit_disabled, type="primary"):
        elapsed = current_elapsed_time()
        selected_df = spots[spots["spot_id"].isin(st.session_state.last_selection_ids)].copy()
        roi_stats = compute_region_stats(selected_df)

        record = {
            "participant_id": st.session_state.participant_id,
            "task_index": st.session_state.submitted_tasks + 1,
            "task_name": st.session_state.task_name or task_name,
            "condition": st.session_state.task_condition or interface_mode,
            "elapsed_seconds": round(elapsed, 3),
            "interaction_count": st.session_state.interaction_count,
            "parameter_changes": st.session_state.param_change_count,
            "selection_count": st.session_state.selection_count,
            "n_selected_spots": len(selected_df),
            "selected_spot_ids": "|".join(map(str, st.session_state.last_selection_ids)),
            "selected_answer_type": "|".join(selected_answer_type),
            "confidence_1_7": confidence,
            "roi_dominant_region": roi_stats["dominant_region"],
            "roi_mean_tumor_probability": roi_stats["mean_tumor_probability"],
            "roi_mean_tumor_gradient": roi_stats["mean_tumor_gradient"],
            "roi_mean_uncertainty": roi_stats["mean_uncertainty"],
            "roi_heterogeneity_score": roi_stats["heterogeneity_score"],
            "roi_tumor_fraction_estimate": roi_stats["tumor_fraction_estimate"],
            "qualitative_note": usability_note,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        st.session_state.study_logs.append(record)
        st.session_state.submitted_tasks += 1
        stop_task()
        st.success("Task result saved.")


# ------------------------------------------------------------
# Bottom summary area
# ------------------------------------------------------------
st.markdown("---")

summary_col1, summary_col2, summary_col3 = st.columns([1.0, 1.2, 1.2])

with summary_col1:
    st.subheader("Region Counts")
    st.dataframe(
        spots["region"].value_counts().rename_axis("region").reset_index(name="count"),
        use_container_width=True,
        hide_index=True
    )

with summary_col2:
    st.subheader("Global Candidate Summary")
    candidate_summary = pd.DataFrame({
        "candidate_type": ["hotspot", "heterogeneous"],
        "top_score_mean": [
            hotspot_candidates["hotspot_score"].mean(),
            hetero_candidates["heterogeneity_score_local"].mean(),
        ],
        "top_score_max": [
            hotspot_candidates["hotspot_score"].max(),
            hetero_candidates["heterogeneity_score_local"].max(),
        ]
    })
    st.dataframe(candidate_summary, use_container_width=True, hide_index=True)

with summary_col3:
    st.subheader("Study Logs")
    if st.session_state.study_logs:
        log_df = pd.DataFrame(st.session_state.study_logs)
        st.dataframe(log_df, use_container_width=True)

        st.download_button(
            label="Download study logs as CSV",
            data=df_to_csv(log_df),
            file_name=f"{st.session_state.participant_id}_tumor_atlas_logs.csv",
            mime="text/csv"
        )
    else:
        st.info("No task submissions yet.")