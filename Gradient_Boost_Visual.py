import random
import numpy as np
import pandas as pd
import plotly.graph_objects as plotly
import streamlit as st

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

## Parameters
LEARNING_RATES = [0.1, 0.3, 0.5, 0.7]
MAX_DEPTHS = [1, 3, 5, 7, 9]
ITERATION_POINTS = [1, 10, 50, 100, 200]

########################
# FUNCTIONS
########################

# Load the dataset
def load_user_dataset(uploaded_file):
    if uploaded_file is None:
        data = fetch_california_housing()
        X = data.data
        y = data.target
        feature_names = data.feature_names
        target_name = "Median House Value"

        return X, y, feature_names, target_name, None

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, None, None, None, f"Could not read CSV: {e}"

    if df.shape[1] < 2:
        return None, None, None, None, "Need at least 2 columns"

    target_name = df.columns[-1]
    feature_names = list(df.columns[:-1])

    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    # checks
    if not pd.api.types.is_numeric_dtype(y):
        return None, None, None, None, "Target must be numeric"

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        return None, None, None, None, "All features must be numeric"

    if df.isnull().any().any():
        return None, None, None, None, "Missing values not allowed"

    return X.to_numpy(), y.to_numpy(), feature_names, target_name, None

# Sample dataset (if requested)
def maybe_sample_dataset(X, y, feature_names, max_rows=10000, seed=123):
    if len(X) <= max_rows:
        return X, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_rows, replace=False)

    return X[idx], y[idx]

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

            for i, (y_pred_train, y_pred_test) in enumerate(
                zip(model.staged_predict(X_train), model.staged_predict(X_test)), start=1
            ):
                if i in wanted:
                    for subset, y_true, y_pred in [
                        ("train", y_train, y_pred_train),
                        ("test",  y_test,  y_pred_test),
                    ]:
                        rows.append(pd.DataFrame({
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "learning_rate": lr,
                            "max_depth": depth,
                            "n_estimators": i,
                            "split_seed": split_seed,
                            "subset": subset,
                        }))

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(
        ["learning_rate", "max_depth", "n_estimators", "subset"]
    ).reset_index(drop=True)

    return df

# Compute RMSEs for all settings
@st.cache_data
def build_rmse_table(df_results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = df_results.groupby(["learning_rate", "max_depth", "n_estimators", "subset"])

    for (lr, depth, n_est, subset), g in grouped:
        rmse = mean_squared_error(g["y_true"], g["y_pred"]) ** 0.5
        rows.append({
            "learning_rate": lr,
            "max_depth": depth,
            "n_estimators": n_est,
            "subset": subset,
            "rmse": rmse
        })

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

#######################
## App button callbacks
#######################

def set_best_rmse():
    test_rmse = rmse_table[rmse_table["subset"] == "test"]
    best_row = test_rmse.loc[test_rmse["rmse"].idxmin()]
    st.session_state["selected_lr"] = float(best_row["learning_rate"])
    st.session_state["selected_depth"] = int(best_row["max_depth"])
    st.session_state["selected_iter"] = int(best_row["n_estimators"])


def randomize_seed():
    st.session_state["split_seed"] = random.randint(1, 1000)

# Session state defaults:
if "split_seed" not in st.session_state:
    st.session_state["split_seed"] = 42

if "selected_depth" not in st.session_state:
    st.session_state["selected_depth"] = 3

if "selected_lr" not in st.session_state:
    st.session_state["selected_lr"] = 0.1

if "selected_iter" not in st.session_state:
    st.session_state["selected_iter"] = 10

if "use_sampling" not in st.session_state:
    st.session_state["use_sampling"] = "Sample data (1K points)"

######################
## App itself
######################

st.set_page_config(
    page_title="Gradient Boosting Fitting Demo",
    page_icon="📈",
    layout="wide"
)

st.header("Gradient Boosting Fitting Demo")

with st.expander("About this app", expanded=False):
    st.markdown(
        """
        Gradient Boosting is a powerful predictive algorithm for structured tabular data. However, it is often treated as a "black box".

        This app was built as a teaching aid to help develop intuition for how the model behaves, specifically how three key parameters affect generalization and overfitting.

        Rather than showing only the RMSE, the app plots the actual values against the predicted ones. The visual intuition is simple: the closer the points lie to the fixed 45° line, the better the model is performing.

        **Questions you might explore:**
        - How do underfitting and overfitting emerge under different parameter settings?
        - How do learning rate and tree depth interact?
        - Can increasing the number of iterations eventually hurt performance?
        """
    )

# Read data source from session state (widgets are rendered below the plot)
uploaded_file = st.session_state.get("csv_upload")

# Validate file size if a file is present
if uploaded_file is not None and uploaded_file.size / (1024 * 1024) > 50:
    uploaded_file = None  # fall back to default; error is shown below near the widget

X, y, feature_names, target_name, error = load_user_dataset(uploaded_file)

if error:
    st.error(error)
    st.stop()

# Read sampling mode from session state (widget is rendered below the plot)
use_sampling = st.session_state.get("use_sampling", "Sample data (1K points)")

if use_sampling == "Sample data (1K points)":
    X_model, y_model = maybe_sample_dataset(X, y, feature_names, max_rows=1000, seed=123)
elif use_sampling == "Sample data (10K points)":
    X_model, y_model = maybe_sample_dataset(X, y, feature_names, max_rows=10000, seed=123)
else:
    X_model, y_model = X, y

# build results
df = build_results_table(st.session_state["split_seed"], X_model, y_model)
rmse_table = build_rmse_table(df)

# Fixed axis limits across all parameter choices
global_min = min(df["y_true"].min(), df["y_pred"].min())
global_max = max(df["y_true"].max(), df["y_pred"].max())

# Add a small margin so points are not on the edge
margin = 0.01 * (global_max - global_min)
axis_min = global_min - margin
axis_max = global_max + margin

st.markdown(
    """
    <style>
    div[data-testid="stRadio"] label p { font-size: 0.85em !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Controls
with st.expander("Controls", expanded=True):
    ctrl_depth, ctrl_lr, ctrl_iter = st.columns(3)
    with ctrl_depth:
        selected_depth = st.radio(
            "Max depth",
            options=MAX_DEPTHS,
            key="selected_depth",
            horizontal=True
        )
    with ctrl_lr:
        selected_lr = st.radio(
            "Learning rate",
            options=LEARNING_RATES,
            key="selected_lr",
            horizontal=True,
            format_func=lambda x: f"{int(x * 100)}%"
        )
    with ctrl_iter:
        selected_iter = st.radio(
            "Number of iterations",
            options=ITERATION_POINTS,
            key="selected_iter",
            horizontal=True
        )

selected_mask = (
    (df["learning_rate"] == st.session_state["selected_lr"]) &
    (df["max_depth"] == st.session_state["selected_depth"]) &
    (df["n_estimators"] == st.session_state["selected_iter"])
)
plot_df_test  = df[selected_mask & (df["subset"] == "test")].copy()
plot_df_train = df[selected_mask & (df["subset"] == "train")].copy()


def make_scatter(plot_df, color, axis_min, axis_max):
    rmse = mean_squared_error(plot_df["y_true"], plot_df["y_pred"]) ** 0.5
    n = len(plot_df)

    fig = plotly.Figure()

    fig.add_trace(plotly.Scatter(
        x=plot_df["y_true"],
        y=plot_df["y_pred"],
        mode="markers",
        marker=dict(color=color, opacity=0.5, size=5),
        name="Predictions",
        hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(plotly.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        line=dict(dash="dash", color="gray", width=1.5),
        name="Perfect fit",
        hoverinfo="skip"
    ))

    fig.update_layout(
        template="simple_white",
        xaxis_title="Actual outcome",
        yaxis_title="Predicted outcome",
        xaxis=dict(range=[axis_min, axis_max]),
        yaxis=dict(range=[axis_min, axis_max], scaleanchor="x", scaleratio=1),
        height=450,
        dragmode=False,
        legend=dict(x=0.98, y=0.05, xanchor="right", yanchor="bottom", bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=40, b=50, l=60, r=20),
    )

    return fig, rmse, n

col_test, col_train = st.columns(2)
with col_test:
    with st.expander("Test subset", expanded=True):
        fig, rmse, n = make_scatter(plot_df_test,  "#2E86AB", axis_min, axis_max)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
        st.markdown(f"<p style='font-size:1.6rem;white-space:nowrap'>RMSE: <b>{rmse:.2f}</b> &nbsp;·&nbsp; N: {n}</p>", unsafe_allow_html=True)
with col_train:
    with st.expander("Train subset", expanded=True):
        fig, rmse, n = make_scatter(plot_df_train, "#E07B39", axis_min, axis_max)
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
        st.markdown(f"<p style='font-size:1.6rem;white-space:nowrap'>RMSE: <b>{rmse:.2f}</b> &nbsp;·&nbsp; N: {n}</p>", unsafe_allow_html=True)

## Other buttons
col_a, col_b = st.columns(2)
with col_a:
    st.button("Show lowest RMSE setting", on_click=set_best_rmse, use_container_width=True)
with col_b:
    st.button("Randomize train/test seed", on_click=randomize_seed, use_container_width=True)
    st.caption(f"Current seed: {st.session_state['split_seed']}")

with st.expander("Data size option"):
    st.radio(
        "Choose computation mode",
        options=["Sample data (1K points)", "Sample data (10K points)", "Use full data"],
        key="use_sampling",
        horizontal=True
    )
    st.caption(f"The full dataset includes {len(y)} observations. Using full data may take much longer to compute, especially for large datasets.")
    if use_sampling == "Sample data (10K points)":
        if len(X) > 1000:
            st.warning("10K sampling is selected. This may take a while to compute.")
        else:
            st.warning("Dataset has less than 1,000 points")
    elif use_sampling == "Use full data":
        if len(X) > 10000:
            st.warning("Full-data mode is selected. This may take a long while to compute.")
        else:
            st.warning("Dataset has less than 10,000 points")

with st.expander("Dataset summary"):
    if uploaded_file is None:
        name_data = "California Housing Dataset (from sklearn)"
    else:
        name_data = uploaded_file.name

    st.metric("Dataset", name_data)
    st.metric("Target variable", target_name)
    st.metric("Number of features", len(feature_names))

    with st.expander("Feature list"):
        st.write("  \n".join(f"- {f}" for f in feature_names))

with st.expander("Upload another table"):
    st.markdown(
        """
        Done with the default dataset? Upload your own!

        Must follow format:
        - CSV file
        - Target variable is the last column
        - All variables are numeric (for now)
        - No missing data
        """
    )
    _uploaded = st.file_uploader("Upload CSV (max 50 MB recommended)", type=["csv"], key="csv_upload")
    if _uploaded is not None and _uploaded.size / (1024 * 1024) > 50:
        st.error("File too large (max 50 MB recommended).")
