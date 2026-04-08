# Gradient Boosting teaching aid
[Link to the app on StreamLit](https://gradient-boosting-teaching-aid.streamlit.app/)

Gradient Boosting is a powerful predictive algorithm for structured tabular data.

This interactive Streamlit app was built as a teaching aid to help users develop intuition for how gradient boosting behaves, especially how three key parameters - the learning rate, maximum tree depth, and number of iterations - affect generalization and overfitting.


## What this app does
Rather than showing only the RMSE on the test subset, the app plots actual values against predicted values in a scatter plot.

The visual intuition is simple: the closer the points lie to the fixed 45° reference line, the better the model is performing on the test split.

To make comparisons meaningful, the axes remain fixed across all parameter configurations, so the visual impression is not distorted by automatic rescaling.


## Why is this useful
Gradient boosting is often treated as a “black box.” This app makes its behavior more transparent and helps users build intuition about questions such as:
- How do underfitting and overfitting emerge under different parameter settings?
- How do learning rate and tree depth interact?
- Can increasing the number of iterations eventually hurt performance?

## Features
The App offers interactive controls for the parameters:
* Learning rate;
* Tree depth;
* Number of iterations.

It also allows users to:
* Jump instantly to the lowest-RMSE configuration;
* Randomize the train/test split to a new seed;
* Choose between full data (better accuracy) or sampled data (faster computation);
* Upload their own dataset!

## Uploading your own data
Instead of using the default California Housing dataset from scikit-learn, users can upload a CSV file directly in the app.

Expected format:
* Last column is target variable;
* All columns are numeric (for now);
* No missing values;
* File size should not exceed 50MB.

## Notes on performance
Large datasets can take longer to compute.

Users can switch between:
* Small sample of 1,000 observations (fast)
* Larger sample of 10,000 observations (balanced)
* Full data (Most accurate but slowest)


## Design philosophy
This app focuses on intuition over abstraction:
* Visual feedback instead rather only metrics;
* Fixed scales to prevent misleading comparisons;
* Minimal but meaningful controls.

## Contributing
Contributions are welcome!

If you have ideas for improvements, feel free to open an issue or submit a pull request.
