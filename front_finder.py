import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from PIL import Image
from pathlib import Path
from roi_comparison import run_all_markers


@st.cache_data
def load_data():

    preprocess_dir = Path("Preprocess_data")

    spots = pd.read_parquet(preprocess_dir / "spots.parquet")

    img = Image.open(preprocess_dir / "tissue_hires_image.png")

    return spots, img


spots, img = load_data()

W, H = img.size

st.title("Spatial Tumor Region Explorer")


# slider controls
st.sidebar.header("Region Parameters")

tumor_threshold = st.sidebar.slider(
    "Tumor core probability threshold",
    0.0, 1.0, 0.7
)

normal_threshold = st.sidebar.slider(
    "Normal probability threshold",
    0.0, 1.0, 0.3
)

gradient_threshold = st.sidebar.slider(
    "Invasion front gradient threshold",
    0.0, 1.0, 0.25
)

k_neighbors = st.sidebar.slider(
    "Spatial neighbors",
    3, 20, 6
)

# find neighbors

coords = spots[['x','y']].values

nbrs = NearestNeighbors(
    n_neighbors=k_neighbors
).fit(coords)

distances, indices = nbrs.kneighbors(coords)

# find tumor gradients

gradients = []

prob = spots["tumor_probability"].values

for i in range(len(spots)):

    neigh = indices[i]

    diff = np.abs(prob[i] - prob[neigh])

    gradients.append(diff.mean())

spots["tumor_gradient"] = gradients

# region classification 

spots["region"] = "other"

# Tumor core
spots.loc[
    spots.tumor_probability > tumor_threshold,
    "region"
] = "tumor_core"

# Adjacent normal
adjacent_normal = (
    (spots.tumor_probability < normal_threshold) &
    (spots.tumor_gradient > gradient_threshold/2)
)

spots.loc[adjacent_normal, "region"] = "adjacent_normal"

# Invasion front
invasion = (
    (spots.tumor_gradient > gradient_threshold) &
    (spots.tumor_probability.between(normal_threshold, tumor_threshold))
)

spots.loc[invasion, "region"] = "invasion_front"

# visualization 

color_option = st.selectbox(
    "Color spots by",
    [
        "region",
        "tumor_probability",
        "tumor_gradient",
        "immune_score",
        "proliferation_index",
        "uncertainty",
        "cell_type"
    ]
)

# plot spots on Tissues

fig = px.scatter(
    spots,
    x="x",
    y="y",
    color=color_option,
    hover_data=[
        "spot_id",
        "tumor_probability",
        "immune_score",
        "proliferation_index",
        "uncertainty",
        "cell_type"
    ],
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
    dragmode="lasso",
    width=900,
    height=800
)

st.plotly_chart(fig, use_container_width=True)

# region summary
st.subheader("Region Counts")

st.write(spots["region"].value_counts())

# roi comparisons
st.subheader("ROI")
markers = ["tumor_probability", "immune_score", "proliferation_index", "uncertainty"]


filtered_spots = spots[spots["region"] != "other"]

# comparisons
results = run_all_markers(filtered_spots, group_col="region", markers=markers)

results["p_value"] = results["p_value"].apply(lambda x: f"{x:.2e}")
sig_results = results[(results["effect_size_d"].abs() > 0.5)]

st.write("### Comparisons")
st.dataframe(sig_results.sort_values("p_value"))

st.write("### Side-by-side Boxplots")
for marker in markers:
    with st.expander(f"Boxplot: {marker}"):
        fig_box = px.box(
            filtered_spots,
            x="region",
            y=marker,
            color="region",
            points="all",
            title=f"{marker} by Region"
        )
        st.plotly_chart(fig_box, use_container_width=True)