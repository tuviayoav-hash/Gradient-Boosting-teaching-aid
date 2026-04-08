# Gradient Boosting teaching aid
[Link to the app on StreamLit](https://gradient-boosting-teaching-aid.streamlit.app/)

Gradient Boosting is a powerful prediction algorithm for structured tabular data.

This interactive Streamlit app was developed as a teaching aid, for building intuition about how gradient boosting behaves.

Specifically, how the setting of three parameters - the learning rate, the maximum depth, and the iteration count - affects generalization and overfitting.

## What this project does
Instead of just providing the RMSE on the test sub-set, the app plots the actual values vs. the predictions on a scatter plot.

The intuition presented is as the dots are horizontally closer to the fixed 45° reference line, so does the algorithm perform better on the test split.

Axes are fixed across all configurations, so you can visually compare models without misleading rescaling.

## Why is this useful
Gradient boosting is often treated as a “black box.”

This tool makes its behavior explicit, and allows to intuitivelly understand:
* How does underfitting or overfitting emerge from different settings?
* How do the learning rate and and the depth interact with one another?
* Can increasing iterations eventually hurt performance?

## Features
The App offers interactive controls for the parameters:
* Learning rate
* Tree depth
* Number of iterations

Additionally, it allows:
* To instantly jump to the lowest-RMSE configuration
* To randomize train/test split
* To use the full data or downsample for efficiency's sake
* To upload your own dataset!

## Uploading your own data
Instead of using the default California Housing Prices dataset, you can upload a CSV file directly in the app.

Expected format:
* Last column is target variable
* All columns are numeric (for now)
* No missing values
* Dataset not exceeding 50MB

## Notes on performance
Large datasets can slow computation.

You can switch between:
* Small sample of 1,000 observations from the data (fast)
* Larger sample of 10,000 observations (a bit slower, but more accurate)
* Full data (Most accurate but slowest)


## Design philosophy
This app focuses on intuition over abstraction:
* Visual feedback instead of only metrics
* Fixed scales to prevent misleading comparisons
* Minimal but meaningful controls

## Contributing
Contributions are welcome!

If you have ideas for improvements, feel free to open an issue or submit a pull request.
