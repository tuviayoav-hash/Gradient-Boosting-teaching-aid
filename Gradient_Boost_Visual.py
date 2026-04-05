import random
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


st.set_page_config(
    page_title="Gradient Boosting Overfitting Demo",
    layout="wide"
)

LEARNING_RATES = [0.05, 0.1, 0.3, 0.7, 0.9]
MAX_DEPTHS = [1, 3, 6]
ITERATION_POINTS = [1, 5, 10, 50, 100]


@st.cache_data
def build_results_table(split_seed: int) -> pd.DataFrame:
    X, y = load_diabetes(return_X_y=True)

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


if "split_seed" not in st.session_state:
    st.session_state.split_seed = 42

if "layout_toggle" not in st.session_state:
    st.session_state.layout_toggle = "Mobile"


st.title("Gradient Boosting Overfitting Demo")
st.write(f"Current train/test split seed: **{st.session_state.split_seed}**")

top_left, top_right = st.columns([2, 1])

with top_left:
    if st.button("Re-compute table with a new train/test split"):
        st.session_state.split_seed = random.randint(1, 10_000_000)

df = build_results_table(st.session_state.split_seed)

with top_right:
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download current table as CSV",
        data=csv_data,
        file_name=f"gb_test_predictions_seed_{st.session_state.split_seed}.csv",
        mime="text/csv"
    )

# Fixed axis limits across all parameter choices
global_min = min(df["y_true"].min(), df["y_pred"].min())
global_max = max(df["y_true"].max(), df["y_pred"].max())

# Add a small margin so points are not on the edge
margin = 0.05 * (global_max - global_min)
axis_min = global_min - margin
axis_max = global_max + margin

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
    f"Test subset | learning_rate={selected_lr}, max_depth={selected_depth}, iterations={selected_iter}"
)

ax.text(
    0.03, 0.97,
    f"RMSE = {rmse:.2f}",
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

fig.tight_layout()

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

with st.expander("Show filtered data"):
    st.dataframe(plot_df, use_container_width=True)
