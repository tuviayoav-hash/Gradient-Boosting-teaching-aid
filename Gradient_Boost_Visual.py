import io
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


LEARNING_RATES = [0.05, 0.1, 0.3, 0.7, 0.9]
MAX_DEPTHS = [1, 3, 6]
ITERATION_POINTS = [1, 5, 10, 50, 100]

# Load the dataset
def load_user_dataset(uploaded_file):
    if uploaded_file is None:
        X, y = load_diabetes(return_X_y=True)
        feature_names = [
            "age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"
        ]
        target_name = "disease_progression"

        return X, y, feature_names, target_name, None

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, None, None, None, f"Could not read CSV: {e}"

    if df.shape[1] < 2:
        return None, None, None, None, "Need at least 2 columns"

    target_name = df.columns[0]
    feature_names = list(df.columns[1:])

    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    # checks (same as before)
    if not pd.api.types.is_numeric_dtype(y):
        return None, None, None, None, "Target must be numeric"

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        return None, None, None, None, "All features must be numeric"

    if df.isnull().any().any():
        return None, None, None, None, "Missing values not allowed"

    return X.to_numpy(), y.to_numpy(), feature_names, target_name, None


# Gradient boost algo
@st.cache_data
def build_results_table(split_seed: int, X, y) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=split_seed
    )

    rows = []
    wanted = set(ITERATION_POINTS)
    max_estimators = max(ITERATION_POINTS)

    for lr in LEARNING_RATES:
        for depth in MAX_DEPTHS:
            model = GradientBoostingRegressor(
                learning_rate=lr,
                n_estimators=max_estimators,
                max_depth=depth,
                random_state=42
            )

            model.fit(X_train, y_train)

            for i, y_pred_test in enumerate(model.staged_predict(X_test), start=1):
                if i in wanted:
                    df_chunk = pd.DataFrame({
                        "y_true": y_test,
                        "y_pred": y_pred_test,
                        "learning_rate": lr,
                        "max_depth": depth,
                        "n_estimators": i,
                        "split_seed": split_seed
                    })
                    rows.append(df_chunk)

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(
        ["learning_rate", "max_depth", "n_estimators"]
    ).reset_index(drop=True)

    return df


## App itself
st.set_page_config(
    page_title="Gradient Boosting Fitting Demo",
    layout="wide"
)

st.title("Gradient Boosting Fitting Demo")

if "split_seed" not in st.session_state:
    st.session_state.split_seed = 42

if "layout_toggle" not in st.session_state:
    st.session_state.layout_toggle = "Mobile"


# Upload CSV (if the user wants, if not default is the diabetes data)
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

X, y, feature_names, target_name, error = load_user_dataset(uploaded_file)

if error:
    st.error(error)
    st.stop()

df = build_results_table(st.session_state.split_seed, X, y)

if st.button("Randomize train/test split"):
    st.session_state.split_seed = random.randint(1, 10_000_000)

# Fixed axis limits across all parameter choices
global_min = min(df["y_true"].min(), df["y_pred"].min())
global_max = max(df["y_true"].max(), df["y_pred"].max())

# Add a small margin so points are not on the edge
margin = 0.05 * (global_max - global_min)
axis_min = global_min - margin
axis_max = global_max + margin

st.subheader("Controls")
ctrl_depth, ctrl_lr, ctrl_iter = st.columns(3)
with ctrl_depth:
    selected_depth = st.radio(
        "Max depth",
        options=MAX_DEPTHS,
        index=MAX_DEPTHS.index(3),
        key="gb_depth",
        horizontal=True
    )
with ctrl_lr:
    selected_lr = st.radio(
        "Learning rate",
        options=LEARNING_RATES,
        index=LEARNING_RATES.index(0.1),
        key="gb_lr",
        horizontal=True
    )
with ctrl_iter:
    selected_iter = st.radio(
        "Number of iterations",
        options=ITERATION_POINTS,
        index=ITERATION_POINTS.index(10),
        key="gb_iter",
        horizontal=True
    )
        
"""
layout_choice = st.radio(
    "Layout",
    options=["Mobile", "Desktop"],
    horizontal=True,
    key="layout_toggle",
)

mobile_layout = layout_choice == "Mobile"
right_col = None

if mobile_layout:
    st.subheader("Controls")
    ctrl_depth, ctrl_lr, ctrl_iter = st.columns(3)
    with ctrl_depth:
        selected_depth = st.radio(
            "Max depth",
            options=MAX_DEPTHS,
            index=MAX_DEPTHS.index(3),
            key="gb_depth",
        )
    with ctrl_lr:
        selected_lr = st.radio(
            "Learning rate",
            options=LEARNING_RATES,
            index=LEARNING_RATES.index(0.1),
            key="gb_lr",
        )
    with ctrl_iter:
        selected_iter = st.radio(
            "Number of iterations",
            options=ITERATION_POINTS,
            index=ITERATION_POINTS.index(10),
            key="gb_iter",
        )
else:
    left_col, right_col = st.columns([1.3, 4.7])
    with left_col:
        st.subheader("Controls")
        selected_depth = st.radio(
            "Max depth",
            options=MAX_DEPTHS,
            index=MAX_DEPTHS.index(3),
            key="gb_depth",
        )
        selected_lr = st.radio(
            "Learning rate",
            options=LEARNING_RATES,
            index=LEARNING_RATES.index(0.1),
            key="gb_lr",
        )
        selected_iter = st.radio(
            "Number of iterations",
            options=ITERATION_POINTS,
            index=ITERATION_POINTS.index(10),
            key="gb_iter",
        )
"""

plot_df = df[
    (df["learning_rate"] == selected_lr) &
    (df["max_depth"] == selected_depth) &
    (df["n_estimators"] == selected_iter)
].copy()

rmse = mean_squared_error(
    plot_df["y_true"],
    plot_df["y_pred"]
) ** 0.5

fig, ax = plt.subplots(figsize=(5.5, 4))

ax.scatter(
    plot_df["y_true"],
    plot_df["y_pred"],
    alpha=0.7
)

ax.plot(
    [axis_min, axis_max],
    [axis_min, axis_max],
    linestyle="--"
)

ax.set_xlim(axis_min, axis_max)
ax.set_ylim(axis_min, axis_max)
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("Actual outcome")
ax.set_ylabel("Predicted outcome")
ax.set_title(
    f"Test subset | {selected_lr}, max_depth={selected_depth}, iterations={selected_iter}"
)

ax.text(
    0.03, 0.97,
    f"RMSE = {rmse:.2f}",
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

fig.tight_layout()

## Metadata
st.write(f"Current train/test split seed: **{st.session_state.split_seed}**")

st.divider()
st.subheader("Dataset summary")

n_obs = len(y)
n_features = len(feature_names)

col1, col2 = st.columns(2)

col1.metric("Number of observations", n_obs)
col2.metric("Number of features", n_features)

st.markdown(f"**Target variable:** `{target_name}`")

st.markdown("**Feature variables:**")
st.write(", ".join(feature_names))

"""
if mobile_layout:
    st.pyplot(fig, use_container_width=True)
else:
    with right_col:
        st.pyplot(fig, use_container_width=True)

if mobile_layout:
    m1, m2 = st.columns(2)
    m1.metric("Learning rate", selected_lr)
    m2.metric("Max depth", selected_depth)
    m3, m4 = st.columns(2)
    m3.metric("Iterations", selected_iter)
    m4.metric("RMSE", f"{rmse:.2f}")
else:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Learning rate", selected_lr)
    metric_col2.metric("Max depth", selected_depth)
    metric_col3.metric("Iterations", selected_iter)
    metric_col4.metric("RMSE", f"{rmse:.2f}")
"""
