# Predict Customer Churn

Predicting Credit Card Customer Churn with Clean Code — a project of the
[**ML DevOps Engineer** Nanodegree](https://www.udacity.com/) (Udacity).

---

## Project Description

This project predicts which credit card customers are likely to **churn**
(i.e. leave the bank) so that the business can identify and target them
proactively. The goal is to refactor a monolithic exploratory notebook into a
modular, production-ready Python pipeline that follows the **PEP 8** style
guide, is fully documented, and is unit-tested and logged.

The machine learning pipeline:

- loads the raw bank data (`data/bank_data.csv`);
- runs **exploratory data analysis (EDA)** and saves the resulting figures;
- **encodes** the categorical features using the mean of the target per category;
- **splits** the data into training and test sets;
- trains a **Random Forest** (tuned with randomised hyper-parameter search) and a
  **Logistic Regression** (wrapped in a scaling pipeline);
- **evaluates** both models and saves ROC curves, classification reports and a
  feature-importance chart;
- persists the trained models so they can be loaded and used in production.

Both scripts are executable from the command line and every run is logged to a
`.log` file inside the `logs/` directory.

## Files and Data Description

### Main files

- `churn_library.py`
  The core pipeline. Contains `import_data()`, `perform_eda()`,
  `encoder_helper()`, `perform_feature_engineering()`, `train_models()`,
  `classification_report_image()` and `feature_importance_plot()`. Running it
  end-to-end produces all figures, models and logs.

- `churn_script_logging_and_tests.py`
  Implements a unit test for every function in `churn_library.py` using simple
  assertions that returned values are non-empty and that the expected artifacts
  are created. It logs an INFO message on success and an ERROR message on
  failure to `logs/churn_library.log`.

- `churn_notebook.ipynb`
  The original exploratory notebook that this project refactors. It is kept for
  reference and is **not** required to run the pipeline.

- `guide.ipynb`
  Udacity-provided setup and troubleshooting notebook.

- `requirements.txt`
  Python dependencies needed to run the project.

### Data

- `data/bank_data.csv`
  A dataset of credit card customers. It contains 10 127 rows and 23 columns,
  including features such as `Customer_Age`, `Dependent_count`,
  `Months_on_book`, `Credit_Limit`, `Total_Trans_Amt`,
  `Total_Trans_Ct`, `Education_Level`, `Marital_Status` and
  `Income_Category`. The target is derived from the `Attrition_Flag`: an
  `Existing Customer` is encoded as `0` and an `Attrited Customer` as `1`.

## Installation / Environment Setup

The project uses a Python virtual environment. Python 3.11 or newer is
recommended.

```bash
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install the dependencies
pip install -r requirements.txt
```

If you use [`uv`](https://docs.astral.sh/uv/) you can create the environment and
install the dependencies in one step:

```bash
uv venv .venv --python python3.11
uv pip install --python .venv/bin/python -r requirements.txt
```

## Running the pipeline

Run the full machine-learning pipeline from the repository root:

```bash
python churn_library.py
```

This performs EDA, feature engineering, model training and evaluation, and
writes all outputs to the directories listed below.

## Running the tests and logging

Run the unit tests and generate the execution log from the repository root:

```bash
python churn_script_logging_and_tests.py
```

Each test logs an INFO message on success and an ERROR message on failure. The
messages are written to `logs/churn_library.log`, which is created during
execution.

## Output locations

- **Figures (EDA):** `images/eda/`
  - `churn_distribution.png`, `customer_age_distribution.png`
  - `marital_status_distribution.png`, `total_transaction_distribution.png`
  - `churn_heatmap.png`
- **Figures (results):** `images/results/`
  - `roc_curve_result.png`, `feature_importances.png`
  - `logistic_results.png`, `random_forest_results.png`
- **Models:** `models/`
  - `rfc_model.pkl` (Random Forest), `logistic_model.pkl` (Logistic Regression)
- **Logs:** `logs/`
  - `churn_library.log`

## Repository platform

This project is hosted on **GitHub**: <https://github.com/WietRob/cd0580-project-starter>.

---

## Authors and Acknowledgement

Based on the [Udacity cd0580-project-starter](https://github.com/udacity/cd0580-project-starter)
repository. Credit card churn dataset provided by Udacity.