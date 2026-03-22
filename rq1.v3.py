import time
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
    page_title="Tumor Atlas HCI Study Prototype",
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
spots["spot_id"] = spots["spot_id"].astype(str)

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

        # RQ2
        "rq2_condition": "ROI card + uncertainty overlay",

        # RQ4
        "predicted_boundary_ids": [],
        "edited_boundary_ids": [],
        "boundary_edit_mode": False,
        "boundary_edit_start_time": None,
        "boundary_edit_elapsed_seconds": 0.0,
        "boundary_edit_count": 0,
        "boundary_undo_count": 0,
        "boundary_before_stats": None,
        "boundary_after_stats": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def increment_interaction(amount: int = 1):
    st.session_state.interaction_count += amount


def increment_param_change():
    st.session_state.param_change_count += 1
    increment_interaction(1)


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

    prob_std = df["tumor_probability"].std(ddof=0) if len(df) > 1 else 0.0
    probs = region_counts / region_counts.sum()
    region_entropy = float(-(probs * np.log2(probs + 1e-12)).sum()) if len(probs) > 1 else 0.0
    heterogeneity_score = 0.6 * prob_std + 0.4 * region_entropy

    tumor_fraction_estimate = float((df["region"] == "tumor_core").mean())

    return {
        "n_spots": int(len(df)),
        "mean_tumor_probability": float(df["tumor_probability"].mean()),
        "mean_tumor_gradient": float(df["tumor_gradient"].mean()),
        "mean_uncertainty": float(df["uncertainty"].mean()) if "uncertainty" in df.columns else np.nan,
        "heterogeneity_score": float(heterogeneity_score),
        "dominant_region": dominant_region,
        "tumor_fraction_estimate": tumor_fraction_estimate,
    }


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

    proliferation = scored["proliferation_index"] if "proliferation_index" in scored.columns else 0
    immune = scored["immune_score"] if "immune_score" in scored.columns else 0
    uncertainty = scored["uncertainty"] if "uncertainty" in scored.columns else 0

    scored["hotspot_score"] = (
        0.60 * scored["tumor_probability"] +
        0.25 * proliferation +
        0.15 * immune -
        0.20 * uncertainty
    )

    mid_prob = 1.0 - np.abs(scored["tumor_probability"] - 0.5) * 2
    scored["heterogeneity_score_local"] = (
        0.55 * scored["tumor_gradient"] +
        0.25 * mid_prob +
        0.20 * uncertainty
    )

    hotspot_candidates = scored.nlargest(top_k, "hotspot_score")[
        ["spot_id", "x", "y", "hotspot_score", "tumor_probability", "tumor_gradient", "uncertainty", "region"]
    ].copy()

    hetero_candidates = scored.nlargest(top_k, "heterogeneity_score_local")[
        ["spot_id", "x", "y", "heterogeneity_score_local", "tumor_probability", "tumor_gradient", "uncertainty", "region"]
    ].copy()

    return hotspot_candidates, hetero_candidates


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
        "roi_label": "unlabeled",
    }

    st.session_state.saved_rois.append(roi_record)
    return True


def clear_saved_rois():
    st.session_state.saved_rois = []
    st.session_state.roi_counter = 0


def delete_roi_by_name(roi_name: str):
    st.session_state.saved_rois = [
        roi for roi in st.session_state.saved_rois
        if roi["roi_name"] != roi_name
    ]


def rename_roi(old_name: str, new_name: str):
    new_name = new_name.strip()
    if not new_name:
        return
    for roi in st.session_state.saved_rois:
        if roi["roi_name"] == old_name:
            roi["roi_name"] = new_name
            break


def relabel_roi(roi_name: str, roi_label: str):
    for roi in st.session_state.saved_rois:
        if roi["roi_name"] == roi_name:
            roi["roi_label"] = roi_label
            break


def build_roi_comparison_table(saved_rois: list) -> pd.DataFrame:
    rows = []
    for roi in saved_rois:
        rows.append({
            "ROI": roi["roi_name"],
            "Label": roi.get("roi_label", "unlabeled"),
            "Spots": roi["n_spots"],
            "Dominant region": roi["dominant_region"],
            "Mean tumor prob": roi["mean_tumor_probability"],
            "Mean gradient": roi["mean_tumor_gradient"],
            "Mean uncertainty": roi["mean_uncertainty"],
            "Heterogeneity": roi["heterogeneity_score"],
            "Tumor fraction est": roi["tumor_fraction_estimate"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# RQ2 helpers
# ------------------------------------------------------------
def infer_boundary_class_from_roi(df: pd.DataFrame) -> str:
    if df.empty:
        return "unknown"

    region_counts = df["region"].value_counts()
    if region_counts.empty:
        return "unknown"

    if "invasion_front" in region_counts and region_counts["invasion_front"] == region_counts.max():
        return "invasion_front"
    if "adjacent_normal" in region_counts and region_counts["adjacent_normal"] == region_counts.max():
        return "adjacent_normal"

    dominant = region_counts.idxmax()
    if dominant == "invasion_front":
        return "invasion_front"
    if dominant == "adjacent_normal":
        return "adjacent_normal"
    return "unknown"


def compute_rq2_ground_truth(selected_df: pd.DataFrame) -> dict:
    if selected_df.empty:
        return {
            "gt_tumor_fraction": np.nan,
            "gt_boundary_class": "unknown",
        }

    if "gt_tumor_fraction" in selected_df.columns:
        gt_tumor_fraction = float(selected_df["gt_tumor_fraction"].mean())
    else:
        gt_tumor_fraction = float((selected_df["region"] == "tumor_core").mean())

    if "gt_boundary_class" in selected_df.columns:
        gt_boundary_class = selected_df["gt_boundary_class"].mode().iloc[0]
    else:
        gt_boundary_class = infer_boundary_class_from_roi(selected_df)

    return {
        "gt_tumor_fraction": gt_tumor_fraction,
        "gt_boundary_class": gt_boundary_class,
    }


def score_rq2_response(
    selected_df: pd.DataFrame,
    user_tumor_fraction: float,
    user_boundary_class: str,
    confidence_1_7: int
) -> dict:
    gt = compute_rq2_ground_truth(selected_df)

    tumor_fraction_error = (
        abs(user_tumor_fraction - gt["gt_tumor_fraction"])
        if not np.isnan(gt["gt_tumor_fraction"]) else np.nan
    )

    boundary_correct = int(user_boundary_class == gt["gt_boundary_class"]) if gt["gt_boundary_class"] != "unknown" else 0

    confidence_norm = (confidence_1_7 - 1) / 6.0

    if np.isnan(tumor_fraction_error):
        tumor_fraction_correctness = np.nan
    else:
        tumor_fraction_correctness = max(0.0, 1.0 - tumor_fraction_error)

    if np.isnan(tumor_fraction_correctness):
        calibration_gap = abs(confidence_norm - boundary_correct)
    else:
        combined_correctness = 0.5 * boundary_correct + 0.5 * tumor_fraction_correctness
        calibration_gap = abs(confidence_norm - combined_correctness)

    return {
        "gt_tumor_fraction": gt["gt_tumor_fraction"],
        "gt_boundary_class": gt["gt_boundary_class"],
        "user_tumor_fraction": user_tumor_fraction,
        "user_boundary_class": user_boundary_class,
        "tumor_fraction_error": tumor_fraction_error,
        "boundary_correct": boundary_correct,
        "confidence_norm": confidence_norm,
        "calibration_gap": calibration_gap,
    }


# ------------------------------------------------------------
# RQ4 helpers
# ------------------------------------------------------------
def initialize_predicted_boundary(selected_df: pd.DataFrame):
    if selected_df.empty:
        return False

    st.session_state.predicted_boundary_ids = list(map(str, selected_df["spot_id"].tolist()))
    st.session_state.edited_boundary_ids = list(map(str, selected_df["spot_id"].tolist()))

    before_df = selected_df.copy()
    st.session_state.boundary_before_stats = compute_region_stats(before_df)
    st.session_state.boundary_after_stats = compute_region_stats(before_df)

    st.session_state.boundary_edit_mode = True
    st.session_state.boundary_edit_start_time = time.time()
    st.session_state.boundary_edit_elapsed_seconds = 0.0
    st.session_state.boundary_edit_count = 0
    st.session_state.boundary_undo_count = 0
    return True


def current_boundary_edit_elapsed() -> float:
    if st.session_state.boundary_edit_mode and st.session_state.boundary_edit_start_time is not None:
        return time.time() - st.session_state.boundary_edit_start_time
    return st.session_state.boundary_edit_elapsed_seconds


def get_boundary_df(full_df: pd.DataFrame, id_list: list[str]) -> pd.DataFrame:
    if not id_list:
        return full_df.iloc[0:0].copy()
    id_set = set(map(str, id_list))
    return full_df[full_df["spot_id"].isin(id_set)].copy()


def apply_boundary_edit(full_df: pd.DataFrame, added_ids: list[str], removed_ids: list[str]):
    edited = set(map(str, st.session_state.edited_boundary_ids))
    before_ids = set(edited)

    for sid in map(str, added_ids):
        edited.add(sid)

    for sid in map(str, removed_ids):
        edited.discard(sid)

    st.session_state.edited_boundary_ids = list(edited)
    st.session_state.boundary_edit_count += 1

    after_df = get_boundary_df(full_df, st.session_state.edited_boundary_ids)
    st.session_state.boundary_after_stats = compute_region_stats(after_df)

    return {
        "before_count": len(before_ids),
        "after_count": len(edited),
        "added_count": len(set(map(str, added_ids))),
        "removed_count": len(set(map(str, removed_ids))),
    }


def undo_boundary_to_predicted(full_df: pd.DataFrame):
    st.session_state.edited_boundary_ids = list(st.session_state.predicted_boundary_ids)
    st.session_state.boundary_undo_count += 1
    after_df = get_boundary_df(full_df, st.session_state.edited_boundary_ids)
    st.session_state.boundary_after_stats = compute_region_stats(after_df)


def finalize_boundary_edit(full_df: pd.DataFrame) -> dict:
    st.session_state.boundary_edit_elapsed_seconds = current_boundary_edit_elapsed()

    before_ids = set(map(str, st.session_state.predicted_boundary_ids))
    after_ids = set(map(str, st.session_state.edited_boundary_ids))

    added_ids = sorted(list(after_ids - before_ids))
    removed_ids = sorted(list(before_ids - after_ids))

    before_df = get_boundary_df(full_df, list(before_ids))
    after_df = get_boundary_df(full_df, list(after_ids))

    before_stats = compute_region_stats(before_df)
    after_stats = compute_region_stats(after_df)

    before_boundary_class = infer_boundary_class_from_roi(before_df)
    after_boundary_class = infer_boundary_class_from_roi(after_df)

    return {
        "predicted_n_spots": len(before_ids),
        "edited_n_spots": len(after_ids),
        "added_spot_count": len(added_ids),
        "removed_spot_count": len(removed_ids),
        "before_tumor_fraction": before_stats["tumor_fraction_estimate"],
        "after_tumor_fraction": after_stats["tumor_fraction_estimate"],
        "tumor_fraction_delta": after_stats["tumor_fraction_estimate"] - before_stats["tumor_fraction_estimate"],
        "before_mean_uncertainty": before_stats["mean_uncertainty"],
        "after_mean_uncertainty": after_stats["mean_uncertainty"],
        "uncertainty_delta": after_stats["mean_uncertainty"] - before_stats["mean_uncertainty"],
        "before_boundary_class": before_boundary_class,
        "after_boundary_class": after_boundary_class,
        "boundary_class_changed": int(before_boundary_class != after_boundary_class),
        "edit_elapsed_seconds": st.session_state.boundary_edit_elapsed_seconds,
        "boundary_edit_count": st.session_state.boundary_edit_count,
        "boundary_undo_count": st.session_state.boundary_undo_count,
    }


# ------------------------------------------------------------
# Title and sidebar study controls
# ------------------------------------------------------------
st.title("Tumor Atlas HCI Study Prototype")
st.caption(
    "Supports RQ1, RQ2, and RQ4 with baseline vs Tumor Atlas views, ROI comparison, "
    "uncertainty-aware estimation tasks, and boundary editing."
)

with st.sidebar:
    st.header("Study Setup")

    participant_id = st.text_input("Participant ID", value=st.session_state.participant_id)
    st.session_state.participant_id = participant_id

    interface_mode = st.radio(
        "Condition",
        ["Baseline", "Tumor Atlas"],
        help="Baseline = simpler viewer. Tumor Atlas = richer interactive interface."
    )

    task_name = st.selectbox(
        "Task Prompt",
        [
            "Find the highest tumor hotspot",
            "Find one spatially heterogeneous region",
            "Find hotspot + heterogeneous region",
            "Estimate tumor fraction and boundary class",
            "Edit boundary and review before/after changes"
        ]
    )

    if task_name == "Estimate tumor fraction and boundary class":
        st.session_state.rq2_condition = st.radio(
            "RQ2 interface condition",
            ["ROI card only", "ROI card + uncertainty overlay"],
            help="Used to compare the effect of uncertainty communication."
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
# Compute gradients and region assignments
# ------------------------------------------------------------
coords = spots[["x", "y"]].values

nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(coords)
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
# Mode-dependent interface
# ------------------------------------------------------------
if interface_mode == "Baseline":
    color_choices = ["tumor_probability"]
    default_color = "tumor_probability"
    allow_candidate_panels = False
    allow_roi_cards = False
    allow_detail_stats = False
    help_text = "Baseline viewer: simple viewer with minimal decision support."
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
    help_text = "Tumor Atlas: richer exploratory condition with ROI summaries and linked region details."

if task_name == "Estimate tumor fraction and boundary class" and st.session_state.rq2_condition == "ROI card only":
    color_choices = [c for c in color_choices if c != "uncertainty"]

st.info(help_text)


# ------------------------------------------------------------
# Main layout
# ------------------------------------------------------------
left_col, right_col = st.columns([2.2, 1.0], gap="large")


# ------------------------------------------------------------
# Main viewer
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
        width=950,
        height=820,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    if interface_mode == "Tumor Atlas":
        fig.add_scatter(
            x=hotspot_candidates["x"],
            y=hotspot_candidates["y"],
            mode="markers",
            name="Hotspot candidates",
            marker=dict(symbol="x", size=11, line=dict(width=1)),
        )
        fig.add_scatter(
            x=hetero_candidates["x"],
            y=hetero_candidates["y"],
            mode="markers",
            name="Heterogeneity candidates",
            marker=dict(symbol="diamond-open", size=12, line=dict(width=1)),
        )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key="main_plot",
        on_select="rerun",
        selection_mode=["points", "box", "lasso"]
    )

    selected_ids = []

    if event is not None and hasattr(event, "selection") and event.selection:
        selected_points = event.selection.get("points", [])
        if selected_points:
            st.session_state.selection_count += 1
            increment_interaction(1)

            for p in selected_points:
                custom = p.get("customdata", [])
                if custom and len(custom) >= 1:
                    selected_ids.append(str(custom[0]))

    seen = set()
    selected_ids = [x for x in selected_ids if not (x in seen or seen.add(x))]
    st.session_state.last_selection_ids = selected_ids

    st.caption(
        "Use point, box, or lasso selection to define an ROI. "
        "In Tumor Atlas mode, the right panel summarizes the selection."
    )


# ------------------------------------------------------------
# Right panel
# ------------------------------------------------------------
with right_col:
    st.subheader("Study Support Panel")

    st.markdown("**Task prompt**")
    st.write(task_name)

    st.markdown("**Current condition**")
    st.write(interface_mode)

    if task_name == "Estimate tumor fraction and boundary class":
        st.markdown("**RQ2 condition**")
        st.write(st.session_state.rq2_condition)

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
            show_uncertainty_overlay = (
                task_name != "Estimate tumor fraction and boundary class"
                or st.session_state.rq2_condition == "ROI card + uncertainty overlay"
            )

            c1, c2 = st.columns(2)
            c1.metric("ROI spots", roi_stats["n_spots"])
            c2.metric("Dominant region", roi_stats["dominant_region"])

            c3, c4 = st.columns(2)
            c3.metric("Mean tumor probability", f"{roi_stats['mean_tumor_probability']:.3f}")
            c4.metric("Mean tumor gradient", f"{roi_stats['mean_tumor_gradient']:.3f}")

            if show_uncertainty_overlay:
                c5, c6 = st.columns(2)
                c5.metric("Mean uncertainty", f"{roi_stats['mean_uncertainty']:.3f}")
                c6.metric("Tumor fraction est.", f"{roi_stats['tumor_fraction_estimate']:.2%}")

                if roi_stats["mean_uncertainty"] >= 0.45:
                    st.error("High uncertainty ROI: interpret with caution.")
                elif roi_stats["mean_uncertainty"] >= 0.25:
                    st.warning("Moderate uncertainty ROI.")
                else:
                    st.success("Low uncertainty ROI.")
            else:
                st.metric("Tumor fraction est.", f"{roi_stats['tumor_fraction_estimate']:.2%}")

            st.metric("Heterogeneity score", f"{roi_stats['heterogeneity_score']:.3f}")

            inferred_boundary = infer_boundary_class_from_roi(selected_df)
            st.metric("Boundary class estimate", inferred_boundary)

            if roi_stats["dominant_region"] == "tumor_core":
                st.markdown("**Interpretation:** ROI is primarily tumor-dominant.")
            elif roi_stats["dominant_region"] == "invasion_front":
                st.markdown("**Interpretation:** ROI suggests a likely boundary / invasive-front region.")
            elif roi_stats["dominant_region"] == "adjacent_normal":
                st.markdown("**Interpretation:** ROI appears mainly adjacent normal tissue.")
            else:
                st.markdown("**Interpretation:** ROI is mixed or weakly classified.")

    if allow_detail_stats and not selected_df.empty:
        st.markdown("---")
        st.subheader("Linked ROI Details")

        st.dataframe(
            selected_df[
                ["spot_id", "x", "y", "region", "tumor_probability", "tumor_gradient", "uncertainty"]
            ].sort_values("tumor_probability", ascending=False),
            use_container_width=True,
            hide_index=True
        )

        stat_options = ["tumor_probability", "tumor_gradient"]
        if task_name != "Estimate tumor fraction and boundary class" or st.session_state.rq2_condition == "ROI card + uncertainty overlay":
            stat_options.append("uncertainty")

        stat_view = st.selectbox(
            "Distribution metric",
            stat_options,
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
            st.image(patch, caption="Zoomed tissue patch", use_container_width=True)

    # -------------------------
    # RQ2 module
    # -------------------------
    user_tumor_fraction = None
    user_boundary_class = None
    uncertainty_reasoning_note = ""
    if task_name == "Estimate tumor fraction and boundary class":
        st.markdown("---")
        st.subheader("RQ2: User Judgment")

        user_tumor_fraction = st.slider(
            "Your estimated tumor fraction",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01
        )

        user_boundary_class = st.radio(
            "Your estimated boundary class",
            ["adjacent_normal", "invasion_front"]
        )

        uncertainty_reasoning_note = st.text_area(
            "What evidence informed your estimate?",
            placeholder="Example: I used the ROI card values and the uncertainty warning..."
        )

        if not selected_df.empty:
            rq2_preview = score_rq2_response(
                selected_df,
                user_tumor_fraction=user_tumor_fraction,
                user_boundary_class=user_boundary_class,
                confidence_1_7=4
            )

            st.markdown("**RQ2 scoring preview**")
            p1, p2 = st.columns(2)
            p1.metric(
                "GT tumor fraction",
                f"{rq2_preview['gt_tumor_fraction']:.2%}" if not np.isnan(rq2_preview["gt_tumor_fraction"]) else "NA"
            )
            p2.metric("GT boundary class", rq2_preview["gt_boundary_class"])

            p3, p4 = st.columns(2)
            p3.metric(
                "Tumor fraction error",
                f"{rq2_preview['tumor_fraction_error']:.3f}" if not np.isnan(rq2_preview["tumor_fraction_error"]) else "NA"
            )
            p4.metric("Boundary correct", "Yes" if rq2_preview["boundary_correct"] == 1 else "No")

    # -------------------------
    # RQ4 module
    # -------------------------
    trust_after_edit = None
    boundary_edit_note = ""
    if task_name == "Edit boundary and review before/after changes":
        st.markdown("---")
        st.subheader("RQ4: Boundary Editing")

        if not selected_df.empty and not st.session_state.boundary_edit_mode:
            if st.button("Initialize predicted boundary from selected ROI", type="primary"):
                ok = initialize_predicted_boundary(selected_df)
                if ok:
                    st.success("Predicted boundary initialized. You can now edit it.")
                else:
                    st.warning("Select an ROI first.")

        if st.session_state.boundary_edit_mode:
            st.write({
                "edit_elapsed_seconds": round(current_boundary_edit_elapsed(), 1),
                "boundary_edit_count": st.session_state.boundary_edit_count,
                "undo_count": st.session_state.boundary_undo_count,
            })

            predicted_df = get_boundary_df(spots, st.session_state.predicted_boundary_ids)
            edited_df = get_boundary_df(spots, st.session_state.edited_boundary_ids)

            before_stats = compute_region_stats(predicted_df)
            after_stats = compute_region_stats(edited_df)

            st.markdown("**Before / After summary**")
            b1, b2 = st.columns(2)
            with b1:
                st.metric("Before tumor fraction", f"{before_stats['tumor_fraction_estimate']:.2%}")
                st.metric("Before dominant region", before_stats["dominant_region"])
                st.metric("Before uncertainty", f"{before_stats['mean_uncertainty']:.3f}")
                st.metric("Before boundary class", infer_boundary_class_from_roi(predicted_df))
            with b2:
                st.metric("After tumor fraction", f"{after_stats['tumor_fraction_estimate']:.2%}")
                st.metric("After dominant region", after_stats["dominant_region"])
                st.metric("After uncertainty", f"{after_stats['mean_uncertainty']:.3f}")
                st.metric("After boundary class", infer_boundary_class_from_roi(edited_df))

            st.markdown("**Boundary edit controls**")
            edit_mode = st.radio(
                "Edit operation",
                ["Add selected spots to boundary", "Remove selected spots from boundary"]
            )

            if st.button("Apply boundary edit"):
                selected_ids_now = list(map(str, st.session_state.last_selection_ids))
                if not selected_ids_now:
                    st.warning("Select spots in the viewer first.")
                else:
                    if edit_mode == "Add selected spots to boundary":
                        edit_summary = apply_boundary_edit(spots, added_ids=selected_ids_now, removed_ids=[])
                    else:
                        edit_summary = apply_boundary_edit(spots, added_ids=[], removed_ids=selected_ids_now)

                    st.success(
                        f"Edit applied. Added: {edit_summary['added_count']}, "
                        f"Removed: {edit_summary['removed_count']}."
                    )

            edit_col1, edit_col2 = st.columns(2)
            with edit_col1:
                if st.button("Undo to predicted boundary"):
                    undo_boundary_to_predicted(spots)
                    st.info("Boundary reset to predicted state.")
            with edit_col2:
                trust_after_edit = st.slider(
                    "How much do you trust the corrected result?",
                    1, 7, 4,
                    key="trust_after_edit"
                )

            boundary_edit_note = st.text_area(
                "What changed after your edit?",
                placeholder="Example: after removing uncertain spots, the tumor fraction dropped and the boundary looked more adjacent_normal."
            )

            final_boundary_summary = finalize_boundary_edit(spots)

            st.markdown("**Computed downstream effect**")
            d1, d2, d3 = st.columns(3)
            d1.metric("Added spots", final_boundary_summary["added_spot_count"])
            d2.metric("Removed spots", final_boundary_summary["removed_spot_count"])
            d3.metric("Boundary changed", "Yes" if final_boundary_summary["boundary_class_changed"] == 1 else "No")

            d4, d5 = st.columns(2)
            d4.metric("Tumor fraction delta", f"{final_boundary_summary['tumor_fraction_delta']:+.3f}")
            d5.metric("Uncertainty delta", f"{final_boundary_summary['uncertainty_delta']:+.3f}")

    # -------------------------
    # ROI collection
    # -------------------------
    st.markdown("---")
    st.subheader("ROI Collection")

    roi_action_col1, roi_action_col2 = st.columns(2)

    with roi_action_col1:
        if st.button("Save selected ROI"):
            ok = save_current_roi(spots, img)
            if ok:
                st.success("Selected ROI saved for comparison.")
            else:
                st.warning("No valid ROI selected to save.")

    with roi_action_col2:
        if st.button("Clear saved ROIs"):
            clear_saved_rois()
            st.info("Saved ROI collection cleared.")

    if st.session_state.saved_rois:
        roi_names = [roi["roi_name"] for roi in st.session_state.saved_rois]

        selected_manage_roi = st.selectbox("Manage saved ROI", roi_names, key="manage_roi_name")
        manage_col1, manage_col2 = st.columns(2)

        with manage_col1:
            new_name = st.text_input("Rename ROI", value=selected_manage_roi, key="rename_roi_input")
            if st.button("Apply rename"):
                rename_roi(selected_manage_roi, new_name)
                st.success("ROI renamed.")

        with manage_col2:
            roi_label = st.selectbox(
                "ROI label",
                ["unlabeled", "hotspot", "heterogeneous", "boundary", "other"],
                key="roi_label_select"
            )
            if st.button("Apply label"):
                relabel_roi(selected_manage_roi, roi_label)
                st.success("ROI label updated.")

        if st.button("Delete selected ROI"):
            delete_roi_by_name(selected_manage_roi)
            st.warning("ROI deleted.")

    # -------------------------
    # Task response
    # -------------------------
    st.markdown("---")
    st.subheader("Task Response")

    confidence = st.slider(
        "How confident are you in your selected answer?",
        1, 7, 4
    )

    selected_answer_type = st.multiselect(
        "What did you select?",
        [
            "Hotspot region",
            "Heterogeneous region",
            "Tumor fraction estimate",
            "Boundary class estimate",
            "Boundary correction"
        ]
    )

    usability_note = st.text_area(
        "Qualitative note on usability / what helped or made the task harder",
        placeholder="Example: side-by-side ROI comparison made it easier to compare suspicious regions."
    )

    submit_disabled = len(st.session_state.last_selection_ids) == 0

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

        if task_name == "Estimate tumor fraction and boundary class":
            rq2_scores = score_rq2_response(
                selected_df,
                user_tumor_fraction=user_tumor_fraction,
                user_boundary_class=user_boundary_class,
                confidence_1_7=confidence
            )

            record.update({
                "rq2_condition": st.session_state.rq2_condition,
                "rq2_user_tumor_fraction": rq2_scores["user_tumor_fraction"],
                "rq2_user_boundary_class": rq2_scores["user_boundary_class"],
                "rq2_gt_tumor_fraction": rq2_scores["gt_tumor_fraction"],
                "rq2_gt_boundary_class": rq2_scores["gt_boundary_class"],
                "rq2_tumor_fraction_error": rq2_scores["tumor_fraction_error"],
                "rq2_boundary_correct": rq2_scores["boundary_correct"],
                "rq2_calibration_gap": rq2_scores["calibration_gap"],
                "rq2_reasoning_note": uncertainty_reasoning_note,
            })

        if task_name == "Edit boundary and review before/after changes" and st.session_state.boundary_edit_mode:
            rq4_summary = finalize_boundary_edit(spots)

            record.update({
                "rq4_predicted_n_spots": rq4_summary["predicted_n_spots"],
                "rq4_edited_n_spots": rq4_summary["edited_n_spots"],
                "rq4_added_spot_count": rq4_summary["added_spot_count"],
                "rq4_removed_spot_count": rq4_summary["removed_spot_count"],
                "rq4_before_tumor_fraction": rq4_summary["before_tumor_fraction"],
                "rq4_after_tumor_fraction": rq4_summary["after_tumor_fraction"],
                "rq4_tumor_fraction_delta": rq4_summary["tumor_fraction_delta"],
                "rq4_before_mean_uncertainty": rq4_summary["before_mean_uncertainty"],
                "rq4_after_mean_uncertainty": rq4_summary["after_mean_uncertainty"],
                "rq4_uncertainty_delta": rq4_summary["uncertainty_delta"],
                "rq4_before_boundary_class": rq4_summary["before_boundary_class"],
                "rq4_after_boundary_class": rq4_summary["after_boundary_class"],
                "rq4_boundary_class_changed": rq4_summary["boundary_class_changed"],
                "rq4_edit_elapsed_seconds": rq4_summary["edit_elapsed_seconds"],
                "rq4_boundary_edit_count": rq4_summary["boundary_edit_count"],
                "rq4_boundary_undo_count": rq4_summary["boundary_undo_count"],
                "rq4_trust_after_edit": trust_after_edit,
                "rq4_edit_note": boundary_edit_note,
            })

            st.session_state.boundary_edit_mode = False

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


# ------------------------------------------------------------
# Side-by-side ROI comparison
# ------------------------------------------------------------
st.markdown("---")
st.header("Side-by-Side ROI Comparison")

if not st.session_state.saved_rois:
    st.info("Save two or more ROIs to compare them side by side.")
else:
    comparison_df = build_roi_comparison_table(st.session_state.saved_rois)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    roi_summary_export = comparison_df.copy()
    st.download_button(
        label="Download ROI comparison table as CSV",
        data=df_to_csv(roi_summary_export),
        file_name=f"{st.session_state.participant_id}_roi_comparison.csv",
        mime="text/csv"
    )

    saved_rois = st.session_state.saved_rois
    n_rois = len(saved_rois)

    display_limit = min(n_rois, 4)
    compare_cols = st.columns(display_limit)

    for i, roi in enumerate(saved_rois[:display_limit]):
        with compare_cols[i]:
            st.subheader(roi["roi_name"])
            st.caption(f"Label: {roi.get('roi_label', 'unlabeled')}")
            st.metric("Spots", roi["n_spots"])
            st.metric("Dominant region", roi["dominant_region"])
            st.metric("Mean tumor prob", f"{roi['mean_tumor_probability']:.3f}")
            st.metric("Mean gradient", f"{roi['mean_tumor_gradient']:.3f}")
            st.metric("Mean uncertainty", f"{roi['mean_uncertainty']:.3f}")
            st.metric("Heterogeneity", f"{roi['heterogeneity_score']:.3f}")
            st.metric("Tumor fraction", f"{roi['tumor_fraction_estimate']:.2%}")

            if roi["patch"] is not None:
                st.image(roi["patch"], caption=roi["roi_name"], use_container_width=True)

    if n_rois > 4:
        st.markdown("**Additional saved ROIs**")
        extra_rois = saved_rois[4:]
        extra_cols = st.columns(min(len(extra_rois), 4))
        for i, roi in enumerate(extra_rois[:4]):
            with extra_cols[i]:
                st.subheader(roi["roi_name"])
                st.caption(f"Label: {roi.get('roi_label', 'unlabeled')}")
                st.metric("Spots", roi["n_spots"])
                st.metric("Dominant region", roi["dominant_region"])
                st.metric("Mean tumor prob", f"{roi['mean_tumor_probability']:.3f}")
                st.metric("Mean gradient", f"{roi['mean_tumor_gradient']:.3f}")
                st.metric("Mean uncertainty", f"{roi['mean_uncertainty']:.3f}")
                st.metric("Heterogeneity", f"{roi['heterogeneity_score']:.3f}")
                st.metric("Tumor fraction", f"{roi['tumor_fraction_estimate']:.2%}")

                if roi["patch"] is not None:
                    st.image(roi["patch"], caption=roi["roi_name"], use_container_width=True)

    st.markdown("---")
    st.subheader("Cross-ROI Metric Comparison")

    metric_options = ["tumor_probability", "tumor_gradient", "uncertainty"]
    metric_to_compare = st.selectbox(
        "Compare ROI distributions by metric",
        metric_options,
        key="cross_roi_metric"
    )

    plot_rows = []
    for roi in saved_rois:
        roi_df = roi["roi_df"].copy()
        roi_df["ROI"] = roi["roi_name"]
        plot_rows.append(roi_df[["ROI", metric_to_compare]])

    if plot_rows:
        compare_plot_df = pd.concat(plot_rows, ignore_index=True)
        box_fig = px.box(
            compare_plot_df,
            x="ROI",
            y=metric_to_compare,
            points="all",
            title=f"{metric_to_compare} across saved ROIs"
        )
        st.plotly_chart(box_fig, use_container_width=True)

    if len(saved_rois) >= 2:
        st.markdown("---")
        st.subheader("Pairwise ROI Difference Summary")

        roi_names = [r["roi_name"] for r in saved_rois]
        diff_col1, diff_col2 = st.columns(2)

        with diff_col1:
            roi_a_name = st.selectbox("ROI A", roi_names, key="roi_a_compare")

        with diff_col2:
            roi_b_name = st.selectbox(
                "ROI B",
                roi_names,
                index=min(1, len(roi_names) - 1),
                key="roi_b_compare"
            )

        if roi_a_name != roi_b_name:
            roi_a = next(r for r in saved_rois if r["roi_name"] == roi_a_name)
            roi_b = next(r for r in saved_rois if r["roi_name"] == roi_b_name)

            diff_df = pd.DataFrame([
                {
                    "Metric": "Mean tumor probability",
                    "ROI A": roi_a["mean_tumor_probability"],
                    "ROI B": roi_b["mean_tumor_probability"],
                    "Difference (A-B)": roi_a["mean_tumor_probability"] - roi_b["mean_tumor_probability"],
                },
                {
                    "Metric": "Mean gradient",
                    "ROI A": roi_a["mean_tumor_gradient"],
                    "ROI B": roi_b["mean_tumor_gradient"],
                    "Difference (A-B)": roi_a["mean_tumor_gradient"] - roi_b["mean_tumor_gradient"],
                },
                {
                    "Metric": "Mean uncertainty",
                    "ROI A": roi_a["mean_uncertainty"],
                    "ROI B": roi_b["mean_uncertainty"],
                    "Difference (A-B)": roi_a["mean_uncertainty"] - roi_b["mean_uncertainty"],
                },
                {
                    "Metric": "Heterogeneity score",
                    "ROI A": roi_a["heterogeneity_score"],
                    "ROI B": roi_b["heterogeneity_score"],
                    "Difference (A-B)": roi_a["heterogeneity_score"] - roi_b["heterogeneity_score"],
                },
                {
                    "Metric": "Tumor fraction estimate",
                    "ROI A": roi_a["tumor_fraction_estimate"],
                    "ROI B": roi_b["tumor_fraction_estimate"],
                    "Difference (A-B)": roi_a["tumor_fraction_estimate"] - roi_b["tumor_fraction_estimate"],
                },
            ])

            st.dataframe(diff_df, use_container_width=True, hide_index=True)
        else:
            st.info("Choose two different ROIs for pairwise comparison.")