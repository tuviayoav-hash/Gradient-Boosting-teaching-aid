import random
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


st.set_page_config(
    page_title="Gradient Boosting Overfitting Demo",
    layout="wide"
)

# Hyper-parameters
LEARNING_RATES = [0.05, 0.1, 0.3, 0.7, 0.9]
MAX_DEPTHS = [1, 3, 6]
ITERATION_POINTS = [1, 5, 10, 50, 100]


# Save results in table
@st.cache_data
def build_results_table(split_seed: int) -> pd.DataFrame:
    X, y = load_diabetes(return_X_y=True)
    
  # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=split_seed
    )

    rows = []
    wanted = set(ITERATION_POINTS)
    max_estimators = max(ITERATION_POINTS)

  # Main loop
    for lr in LEARNING_RATES:
        for depth in MAX_DEPTHS:
            model = GradientBoostingRegressor(
                learning_rate=lr,
                n_estimators=max_estimators,
                max_depth=depth,
                random_state=42
            )

            model.fit(X_train, y_train)

          # Extract prediction from specific stages of algo
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



# StreamLit interface
if "split_seed" not in st.session_state:
    st.session_state.split_seed = 42


st.title("Gradient Boosting Overfitting Demo")
st.write(f"Current train/test split seed: **{st.session_state.split_seed}**")

button_col, download_col = st.columns([2, 1])

with button_col:
    if st.button("Re-compute table with a new train/test split"):
        st.session_state.split_seed = random.randint(1, 10_000_000)

df = build_results_table(st.session_state.split_seed)

with download_col:
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download current table as CSV",
        data=csv_data,
        file_name=f"gb_test_predictions_seed_{st.session_state.split_seed}.csv",
        mime="text/csv"
    )

st.subheader("Controls and plot")

left_col, center_col, right_col = st.columns([1.2, 5, 1.2])

with left_col:
    selected_depth = st.radio(
        "Max depth",
        options=MAX_DEPTHS,
        index=MAX_DEPTHS.index(3)
    )

with right_col:
    selected_lr = st.radio(
        "Learning rate",
        options=LEARNING_RATES,
        index=LEARNING_RATES.index(0.1)
    )

with center_col:
    selected_iter = st.select_slider(
        "Number of iterations",
        options=ITERATION_POINTS,
        value=10
    )

    plot_df = df[
        (df["learning_rate"] == selected_lr) &
        (df["max_depth"] == selected_depth) &
        (df["n_estimators"] == selected_iter)
    ].copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        plot_df["y_true"],
        plot_df["y_pred"],
        alpha=0.7
    )

    line_min = min(plot_df["y_true"].min(), plot_df["y_pred"].min())
    line_max = max(plot_df["y_true"].max(), plot_df["y_pred"].max())
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--")

    ax.set_xlabel("Actual outcome")
    ax.set_ylabel("Predicted outcome")
    ax.set_title(
        f"Test subset | learning_rate={selected_lr}, max_depth={selected_depth}, iterations={selected_iter}"
    )

    st.pyplot(fig)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Learning rate", selected_lr)
metric_col2.metric("Max depth", selected_depth)
metric_col3.metric("Iterations", selected_iter)
metric_col4.metric("Test observations", len(plot_df))

with st.expander("Show filtered data"):
    st.dataframe(plot_df, use_container_width=True)
