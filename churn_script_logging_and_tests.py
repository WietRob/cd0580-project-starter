# -*- coding: utf-8 -*-
"""Test the functions of the churn_library pipeline and log all results.

This module exercises every public function exposed by ``churn_library.py`` on
the real ``data/bank_data.csv`` dataset. Each test asserts that the returned
values are non-empty and that the expected artifacts (figures, models and the
execution log) are created. Successes are reported as INFO messages, while
failures are reported as ERROR messages, and every message is written to
``logs/churn_library.log``.

The tests can be run from the command line with::

    python churn_script_logging_and_tests.py

Author: Roberto Schmidt
Date created: 07.09.2026
"""

import logging
import os

import churn_library as cls

LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")
DATA_PATH = "./data/bank_data.csv"

logger = logging.getLogger(__name__)


def _configure_logging():
    """Configure the logger to write INFO and ERROR messages to the log file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _prepare_dataframe():
    """Load the bank data and add the binary churn target column.

    input:
            None
    output:
            df: pandas dataframe with the churn response column
    """
    df = cls.import_data(DATA_PATH)
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return df


def test_import(import_data):
    """Test that the data import returns a non-empty dataframe.

    input:
            import_data: the churn_library.import_data function
    output:
            None
    """
    try:
        df = import_data(DATA_PATH)
        logger.info("Testing import_data: SUCCESS - dataframe loaded")

    except FileNotFoundError as err:
        logger.error(
            "Testing import_data: ERROR - file %s not found",
            DATA_PATH)
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logger.info(
            "Testing import_data assertions: SUCCESS - %d rows x %d columns",
            df.shape[0],
            df.shape[1],
        )

    except AssertionError as err:
        logger.error(
            "Testing import_data assertions: ERROR - row or column count is zero")
        raise err


def test_eda(perform_eda):
    """Test that the EDA figures are written to the images/eda directory.

    input:
            perform_eda: the churn_library.perform_eda function
    output:
            None
    """
    try:
        df = cls.import_data(DATA_PATH)
        perform_eda(df)

        expected_files = [
            os.path.join(cls.EDA_DIR, "churn_distribution.png"),
            os.path.join(cls.EDA_DIR, "customer_age_distribution.png"),
            os.path.join(cls.EDA_DIR, "marital_status_distribution.png"),
            os.path.join(cls.EDA_DIR, "total_transaction_distribution.png"),
            os.path.join(cls.EDA_DIR, "churn_heatmap.png"),
        ]
        for image_path in expected_files:
            assert os.path.isfile(
                image_path), f"Missing EDA figure: {image_path}"

        logger.info(
            "Testing perform_eda: SUCCESS - %d figures created",
            len(expected_files))

    except Exception as err:
        logger.error("Testing perform_eda: ERROR - %s", err)
        raise err


def test_encoder_helper(encoder_helper):
    """Test that every categorical column is mean-encoded against the target.

    input:
            encoder_helper: the churn_library.encoder_helper function
    output:
            None
    """
    try:
        df = _prepare_dataframe()
        encoded_df = encoder_helper(df, cls.CATEGORY_COLUMNS, "Churn")

        for column in cls.CATEGORY_COLUMNS:
            encoded_column = column + "_Churn"
            assert encoded_column in encoded_df.columns, f"Missing column {encoded_column}"
            assert not encoded_df[encoded_column].isnull().all()

        logger.info(
            "Testing encoder_helper: SUCCESS - %d columns encoded",
            len(cls.CATEGORY_COLUMNS),
        )

    except Exception as err:
        logger.error("Testing encoder_helper: ERROR - %s", err)
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """Test that the feature engineering returns valid train and test splits.

    input:
            perform_feature_engineering: the churn_library.
                perform_feature_engineering function
    output:
            None
    """
    try:
        df = _prepare_dataframe()
        df = cls.encoder_helper(df, cls.CATEGORY_COLUMNS, "Churn")
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, "Churn")

        assert x_train.shape[0] > 0 and x_test.shape[0] > 0
        assert y_train.shape[0] > 0 and y_test.shape[0] > 0
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]

        logger.info(
            "Testing perform_feature_engineering: SUCCESS - "
            "train=%d / test=%d samples",
            x_train.shape[0],
            x_test.shape[0],
        )

    except Exception as err:
        logger.error("Testing perform_feature_engineering: ERROR - %s", err)
        raise err


def test_train_models(train_models):
    """Test that the model files, reports and figures are all persisted.

    input:
            train_models: the churn_library.train_models function
    output:
            None
    """
    try:
        df = _prepare_dataframe()
        df = cls.encoder_helper(df, cls.CATEGORY_COLUMNS, "Churn")
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            df, "Churn"
        )
        train_models(x_train, x_test, y_train, y_test)

        expected_files = [
            os.path.join(cls.MODELS_DIR, "rfc_model.pkl"),
            os.path.join(cls.MODELS_DIR, "logistic_model.pkl"),
            os.path.join(cls.RESULTS_DIR, "roc_curve_result.png"),
            os.path.join(cls.RESULTS_DIR, "feature_importances.png"),
            os.path.join(cls.RESULTS_DIR, "logistic_results.png"),
            os.path.join(cls.RESULTS_DIR, "random_forest_results.png"),
        ]
        for artifact_path in expected_files:
            assert os.path.isfile(
                artifact_path), f"Missing artifact: {artifact_path}"

        logger.info(
            "Testing train_models: SUCCESS - models and %d result figures created",
            len(expected_files) - 2,
        )

    except Exception as err:
        logger.error("Testing train_models: ERROR - %s", err)
        raise err


if __name__ == "__main__":
    _configure_logging()

    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    logger.info("All tests completed successfully.")
    print("Tests completed. Check logs/churn_library.log for details.")
