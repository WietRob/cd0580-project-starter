# -*- coding: utf-8 -*-
"""Predict customer churn for a credit card company.

This module refactors the exploratory ``churn_notebook.ipynb`` into a reusable,
production-ready pipeline. It performs exploratory data analysis (EDA), encodes
categorical features, trains a random forest and a logistic regression model,
and evaluates both classifiers, persisting every artifact (figures, models and
output directories) to disk so the results can be inspected and reused.

Functions
---------
import_data:
    Load the raw bank data from a CSV file into a pandas DataFrame.
perform_eda:
    Run exploratory data analysis and save the resulting figures.
encoder_helper:
    Mean-encode a list of categorical columns against a response column.
perform_feature_engineering:
    Build the feature matrix and split it into train and test sets.
train_models:
    Train both classifiers, evaluate them and save models and figures.
classification_report_image:
    Render classification reports as images.
feature_importance_plot:
    Render the feature importance of a tree-based model as a bar chart.

Author: Roberto Schmidt
Date created: 07.09.2026
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sns.set()

EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"
DATA_PATH = "./data/bank_data.csv"

# Quantitative features retained in the model (see churn_notebook.ipynb).
QUANT_COLUMNS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

# Categorical features that are mean-encoded during feature engineering.
CATEGORY_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]


def create_output_directories():
    """Create the directories used to persist pipeline outputs.

    input:
            None
    output:
            None
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def import_data(pth):
    """Return a dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    """Perform EDA on df and save figures to the images/eda directory.

    input:
            df: pandas dataframe
    output:
            None
    """
    create_output_directories()

    # Binary target derived from the attrition flag.
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # Univariates quantitive distribution.
    plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.title("Churn Distribution")
    plt.savefig(os.path.join(EDA_DIR, "churn_distribution.png"))
    plt.close()

    # Univariates quantitative distribution for customer age.
    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.title("Customer Age Distribution")
    plt.savefig(os.path.join(EDA_DIR, "customer_age_distribution.png"))
    plt.close()

    # Univariates categorical distribution for marital status.
    plt.figure(figsize=(20, 10))
    df["Marital_Status"].value_counts(normalize=True).plot(kind="bar")
    plt.title("Marital Status Distribution")
    plt.savefig(os.path.join(EDA_DIR, "marital_status_distribution.png"))
    plt.close()

    # Distribution of the total transaction count with a kernel density
    # estimate.
    plt.figure(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.title("Total Transaction Count Distribution")
    plt.savefig(os.path.join(EDA_DIR, "total_transaction_distribution.png"))
    plt.close()

    # Bivariates correlation heatmap for every numeric column.
    plt.figure(figsize=(20, 10))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=False, cmap="Dark2_r", linewidths=2)
    plt.title("Feature Correlation Heatmap")
    plt.savefig(os.path.join(EDA_DIR, "churn_heatmap.png"))
    plt.close()


def encoder_helper(df, category_lst, response):
    """Mean-encode categorical columns against the response column.

    Each categorical column in ``category_lst`` is replaced by a new column
    holding the mean of ``response`` for every category, following the pattern
    ``<column>_<response>`` (e.g. ``Gender_Churn``).

    input:
            df: pandas dataframe
            category_lst: list of categorical columns
            response: response column name
    output:
            df: updated dataframe
    """
    for col in category_lst:
        encoded_col = col + "_" + response
        df[encoded_col] = df[col].map(df.groupby(col)[response].mean())
    return df


def perform_feature_engineering(df, response):
    """Encode categories, build the feature matrix and split into train/test.

    Following the sequence diagram, ``encoder_helper`` is invoked here so that
    categorical mean-encoding, feature-matrix construction and the train/test
    split are kept together.

    input:
              df: pandas dataframe (already contains the response column)
              response: response column name
    output:
              x_train: feature matrix for training
              x_test: feature matrix for testing
              y_train: target vector for training
              y_test: target vector for testing
    """
    encoded_df = encoder_helper(df, CATEGORY_COLUMNS, response)

    encoded_cols = [col + "_" + response for col in CATEGORY_COLUMNS]
    keep_cols = QUANT_COLUMNS + encoded_cols

    x_data = encoded_df[keep_cols]
    y_data = encoded_df[response]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42
    )
    return x_train, x_test, y_train, y_test


# The six-argument signature is required by the starter test contract
# (churn_script_logging_and_tests.py), so the arg-count refactor warnings
# are intentionally disabled here.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """Save classification reports as images in the images/results directory.

    input:
            predictions and labels for the training and test sets of both
            the logistic regression and the random forest
    output:
            None
    """
    create_output_directories()

    # Logistic regression.
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str("Logistic Regression Train"), {"fontsize": 10},
             fontproperties="monospace")
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {"fontsize": 10}, fontproperties="monospace")
    plt.text(0.01, 0.6, str("Logistic Regression Test"), {"fontsize": 10},
             fontproperties="monospace")
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {"fontsize": 10}, fontproperties="monospace")
    plt.axis("off")
    plt.savefig(os.path.join(RESULTS_DIR, "logistic_results.png"))
    plt.close()

    # Random forest.
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str("Random Forest Train"), {"fontsize": 10},
             fontproperties="monospace")
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_rf)),
             {"fontsize": 10}, fontproperties="monospace")
    plt.text(0.01, 0.6, str("Random Forest Test"), {"fontsize": 10},
             fontproperties="monospace")
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)),
             {"fontsize": 10}, fontproperties="monospace")
    plt.axis("off")
    plt.savefig(os.path.join(RESULTS_DIR, "random_forest_results.png"))
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    """Create and save the feature importance plot for a tree-based model.

    input:
            model: model object with a feature_importances_ attribute
            x_data: pandas dataframe holding the feature values
            output_pth: path used to persist the figure
    output:
            None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()


def _plot_roc_curves(lr, rfc, x_test, y_test, output_pth):
    """Render combined ROC curves for the logistic and random forest models.

    input:
            lr: fitted logistic regression pipeline
            rfc: best fitted random forest classifier
            x_test: test feature matrix
            y_test: test target vector
            output_pth: path used to persist the figure
    output:
            None
    """
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    RocCurveDisplay.from_estimator(rfc, x_test, y_test, ax=ax)
    RocCurveDisplay.from_estimator(lr, x_test, y_test, ax=ax)

    plt.title("ROC Curves — Logistic Regression vs Random Forest")
    plt.savefig(output_pth)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    """Train models, save the results and persist all artifacts.

    The random forest is tuned with random search cross-validation, while the
    logistic regression is wrapped in a scaling pipeline. Models, ROC curves,
    classification reports and the feature importance plot are all saved.

    input:
              x_train: training feature matrix
              x_test: testing feature matrix
              y_train: training target vector
              y_test: testing target vector
    output:
              None
    """
    create_output_directories()

    # Random forest with randomised hyper-parameter search.
    rfc = RandomForestClassifier(random_state=42)
    param_dist = {
        "n_estimators": [200, 300],
        "max_features": ["sqrt"],
        "max_depth": [5, 8, 10],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "criterion": ["gini"],
    }
    cv_rfc = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=param_dist,
        n_iter=12,
        cv=3,
        random_state=42,
        n_jobs=-1,
        error_score="raise",
    )

    # Logistic regression with feature scaling.
    lrc = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000)),
        ]
    )

    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Persist the trained models.
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            MODELS_DIR,
            "rfc_model.pkl"))
    joblib.dump(lrc, os.path.join(MODELS_DIR, "logistic_model.pkl"))

    # Persist the evaluation artifacts.
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )
    _plot_roc_curves(
        lrc,
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        os.path.join(RESULTS_DIR, "roc_curve_result.png"),
    )
    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_test,
        os.path.join(RESULTS_DIR, "feature_importances.png"),
    )


if __name__ == "__main__":
    create_output_directories()

    bank_df = import_data(DATA_PATH)

    perform_eda(bank_df)

    x_train_data, x_test_data, y_train_data, y_test_data = \
        perform_feature_engineering(bank_df, "Churn")
    train_models(x_train_data, x_test_data, y_train_data, y_test_data)
