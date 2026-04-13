import io
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

## Parameters
LEARNING_RATES = [0.1, 0.3, 0.7, 0.9]
MAX_DEPTHS = [1, 3, 6, 9]
ITERATION_POINTS = [1, 5, 10, 50, 100]

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

# Compute RMSEs for all settings
@st.cache_data
def build_rmse_table(df_results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = df_results.groupby(["learning_rate", "max_depth", "n_estimators"])

    for (lr, depth, n_est), g in grouped:
        rmse = mean_squared_error(g["y_true"], g["y_pred"]) ** 0.5
        rows.append({
            "learning_rate": lr,
            "max_depth": depth,
            "n_estimators": n_est,
            "rmse": rmse
        })

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

#######################
## App button callbacks
#######################

def set_best_rmse():
    best_row = rmse_table.loc[rmse_table["rmse"].idxmin()]
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
    layout="wide"
)

st.title("Gradient Boosting Fitting Demo")

st.markdown(
    """
    Gradient Boosting is a powerful predictive algorithm for structured tabular data. However, it is often treated as a "black box".

    This interactive app was built as a teaching aid to help users develop intuition for how gradient boosting behaves, especially how three key parameters - the learning rate, maximum tree depth, and number of iterations - affect generalization and overfitting.

    Rather than showing only the RMSE on the test subset, the app plots actual values against predicted values in a scatter plot.

    The visual intuition is simple: the closer the points lie to the fixed 45° reference line, the better the model is performing on the test split.

    Questions you might explore:
    - How do underfitting and overfitting emerge under different parameter settings?
    - How do learning rate and tree depth interact?
    - Can increasing the number of iterations eventually hurt performance?

    The default dataset includes sklearn's California House Prices table. Below is an option to upload your own data!
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
margin = 0.05 * (global_max - global_min)
axis_min = global_min - margin
axis_max = global_max + margin

# Controls
st.subheader("Controls")
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
        horizontal=True
    )
with ctrl_iter:
    selected_iter = st.radio(
        "Number of iterations",
        options=ITERATION_POINTS,
        key="selected_iter",
        horizontal=True
    )

plot_df = df[
    (df["learning_rate"] == st.session_state["selected_lr"]) &
    (df["max_depth"] == st.session_state["selected_depth"]) &
    (df["n_estimators"] == st.session_state["selected_iter"])
].copy()

rmse = mean_squared_error(
    plot_df["y_true"],
    plot_df["y_pred"]
) ** 0.5

fig, ax = plt.subplots(figsize=(3.6, 2.7), dpi=200)

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
ax.set_title("Test subset")

ax.text(
    0.03, 0.97,
    f"RMSE = {rmse:.2f}",
    transform=ax.transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
st.pyplot(fig, use_container_width=False)

## Other buttons
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    st.button(
        "Show lowest RMSE setting",
        on_click=set_best_rmse
    )

with btn_col2:
    st.button(
        "Randomize train/test seed",
        on_click=randomize_seed
    )

## Metadata
st.divider()
st.subheader("Dataset summary")

if uploaded_file is None:
    name_data = "California Housing Dataset (from sklearn)"
else:
    name_data = uploaded_file.name

st.metric("Name of dataset", name_data)

col1, col2 = st.columns(2)
col1.metric("Target variable", target_name)
with col2:
    st.write("**Feature variables:**")
    with st.expander(f"{len(feature_names)} features"):
        for f in feature_names:
            st.write(f"- {f}")
            
## Data source controls
st.divider()

# Upload CSV (if the user wants, if not default is the california prices dataset)
st.subheader("Upload another table")
st.markdown(
    """
    Done with the default dataset? Upload you own!

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

# Constrain datasize?
st.divider()

st.subheader("Data size option")

st.radio(
    "Choose computation mode",
    options=["Sample data (1K points)", "Sample data (10K points)", "Use full data"],
    key="use_sampling",
    horizontal=True
)

st.caption(f"The current dataset includes {len(y_model)} observations.")
st.caption("Using full data may take much longer to compute, especially for large datasets.")

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
