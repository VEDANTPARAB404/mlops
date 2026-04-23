import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from pathlib import Path
from collections import Counter
import re

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import resample
from logger_config import logger
from exceptions import TrainingError, ValidationError


LABEL_ENCODER_PATH = Path("label_encoder.pkl")
ARTIFACTS_DIR = Path("/data")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_BOUNDS_PATH = ARTIFACTS_DIR / "feature_bounds.pkl"
PROCESSED_DATA_PATH = ARTIFACTS_DIR / "processed_uploaded.csv"
CONFUSION_MATRIX_PATH = "static/confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = "static/feature_importance.png"
TOP_FEATURE_LIMIT = 12

IDENTIFIER_CANONICAL_NAMES = {
    "id",
    "uuid",
    "guid",
    "pk",
    "primarykey",
    "rowid",
    "recordid",
    "userid",
    "customerid",
    "orderid",
    "transactionid",
    "sessionid",
    "deviceid",
    "accountid",
    "profileid",
    "memberid",
    "kepid",
}
IDENTIFIER_PATTERN = re.compile(
    r"(^id$|^uuid$|^guid$|^pk$|(^|[_\-\s])(id|uuid|guid|pk|identifier|key)([_\-\s]|$))",
    re.IGNORECASE,
)


def _encode_target(y):
    if y.dtype != "object":
        if LABEL_ENCODER_PATH.exists():
            LABEL_ENCODER_PATH.unlink()
            logger.info("Removed old label encoder")
        logger.info(f"Target is numeric, no encoding needed. Unique values: {sorted(y.unique().tolist())}")
        return y, False

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    logger.info(f"Label encoder fitted and saved. Classes: {label_encoder.classes_.tolist()}")
    return encoded, True


def _canonical_column_name(column_name):
    return re.sub(r"[^a-z0-9]", "", str(column_name).lower())


def _drop_identifier_columns(df, target_column):
    drop_cols = []
    for col in df.columns:
        if col == target_column:
            continue

        raw_name = str(col).strip().lower()
        canonical_name = _canonical_column_name(col)

        if canonical_name in IDENTIFIER_CANONICAL_NAMES or IDENTIFIER_PATTERN.search(raw_name):
            drop_cols.append(col)

    if not drop_cols:
        return df

    logger.info(f"Dropping identifier-like column(s) from training: {drop_cols}")
    return df.drop(columns=drop_cols)


def _clean_and_filter_dataset(df, target_column, sample_limit=20000):
    initial_rows = len(df)
    df = df.dropna(subset=[target_column]).drop_duplicates().copy()

    # Trim string fields and fill missing values with stable defaults.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = [c for c in df.columns if c not in numeric_cols]
    for col in cat_cols:
        mode = df[col].mode(dropna=True)
        fallback = mode.iloc[0] if not mode.empty else "unknown"
        df[col] = df[col].fillna(fallback)

    feature_numeric_cols = [c for c in numeric_cols if c != target_column]
    if feature_numeric_cols:
        keep_mask = pd.Series(True, index=df.index)
        for col in feature_numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            keep_mask &= df[col].between(lower, upper)
        filtered = df[keep_mask]
        if not filtered.empty:
            df = filtered

    if len(df) > sample_limit:
        logger.info(f"Sampling dataset from {len(df)} rows to {sample_limit} rows")
        df = df.sample(n=sample_limit, random_state=42)

    logger.info(f"Data cleaned/filtered: {initial_rows} -> {len(df)} rows")
    return df.reset_index(drop=True)


def _prepare_features(df, target_column):
    X_raw = df.drop(target_column, axis=1)
    y = df[target_column]

    X = pd.get_dummies(X_raw)
    feature_bounds = {
        col: {"min": float(X[col].min()), "max": float(X[col].max())}
        for col in X.columns
    }
    return X, y, feature_bounds


def _select_top_features(X_train, y_train, max_features=TOP_FEATURE_LIMIT):
    selector = RandomForestClassifier(
        n_estimators=120,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    selector.fit(X_train, y_train)

    importance_df = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": selector.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    positive = importance_df[importance_df["Importance"] > 0]["Feature"].tolist()
    selected = positive[:max_features] if positive else []
    if not selected:
        selected = importance_df["Feature"].head(min(max_features, len(importance_df))).tolist()

    return selected


def _balance_training_data(X_train, y_train):
    y_series = pd.Series(y_train)
    class_counts = y_series.value_counts()
    if class_counts.empty or len(class_counts) < 2:
        return X_train, y_train, False

    max_count = int(class_counts.max())
    min_count = int(class_counts.min())
    if max_count == min_count:
        return X_train, y_train, False

    train_df = X_train.copy()
    train_df["_target"] = y_series.values
    balanced_parts = []
    for cls in class_counts.index:
        cls_df = train_df[train_df["_target"] == cls]
        balanced_parts.append(
            resample(cls_df, replace=True, n_samples=max_count, random_state=42)
        )

    balanced_df = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    y_balanced = balanced_df.pop("_target")
    logger.info(f"Balanced training data: min class {min_count} -> {max_count}")
    return balanced_df, y_balanced, True


def _save_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    logger.info("Confusion matrix plot saved")


def _save_feature_importance(model, columns):
    feat_df = pd.DataFrame(
        {
            "Feature": columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=feat_df.head(10), x="Importance", y="Feature")
    plt.title("Top 10 Feature Importances")
    plt.savefig(FEATURE_IMPORTANCE_PATH)
    plt.close()
    logger.info("Feature importance plot saved")


def _build_test_prediction_payload(y_test, y_pred, target_encoded):
    actual_values = list(y_test)
    predicted_values = list(y_pred)

    if target_encoded and LABEL_ENCODER_PATH.exists():
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        actual_values = label_encoder.inverse_transform(actual_values).tolist()
        predicted_values = label_encoder.inverse_transform(predicted_values).tolist()

    actual_values = [str(value) for value in actual_values]
    predicted_values = [str(value) for value in predicted_values]

    rows = []
    correct_count = 0

    for index, (actual, predicted) in enumerate(zip(actual_values, predicted_values), start=1):
        is_correct = actual == predicted
        if is_correct:
            correct_count += 1
        rows.append(
            {
                "index": index,
                "actual": actual,
                "predicted": predicted,
                "is_correct": is_correct,
            }
        )

    actual_counter = Counter(actual_values)
    predicted_counter = Counter(predicted_values)
    labels = sorted(set(actual_counter.keys()) | set(predicted_counter.keys()))

    return {
        "test_predictions": rows,
        "test_summary": {
            "total": len(rows),
            "correct": correct_count,
            "incorrect": len(rows) - correct_count,
        },
        "class_comparison": {
            "labels": labels,
            "actual_counts": [actual_counter[label] for label in labels],
            "predicted_counts": [predicted_counter[label] for label in labels],
        },
    }


def train_model(file_path, target_column):
    """Train a RandomForest model on the provided dataset."""
    try:
        logger.info(f"Starting model training with file: {file_path}, target: {target_column}")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        
        df = pd.read_csv(file_path, sep=None, engine='python')
        logger.info(f"Loaded dataset with shape: {df.shape}")

        if target_column not in df.columns:
            error_msg = f"Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()}"
            logger.error(error_msg)
            raise ValidationError(error_msg)

        logger.info(f"Target column '{target_column}' found")

        df = _drop_identifier_columns(df, target_column)

        df = _clean_and_filter_dataset(df, target_column)
        if df.empty:
            raise ValidationError("No usable rows remain after cleaning/filtering")

        if len(df.columns) <= 1:
            raise ValidationError("No usable feature columns remain after removing identifier-like columns")

        X, y, feature_bounds = _prepare_features(df, target_column)
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

        min_class_count = int(y.value_counts().min())
        y, target_encoded = _encode_target(y)
        y = np.asarray(y)

        logger.info("Splitting data into train/test sets (80/20)")

        stratify_y = y if np.unique(y).size > 1 else None
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_y
        )

        top_features = _select_top_features(X_train_full, y_train_full)
        if not top_features:
            raise ValidationError("Unable to determine top features for training")

        logger.info(f"Using top {len(top_features)} feature(s): {top_features}")

        X_top = X[top_features].copy()
        feature_bounds = {feature: feature_bounds[feature] for feature in top_features}

        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_top), columns=top_features)

        processed_export = X_scaled.copy()
        processed_export[target_column] = y
        processed_export.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info(f"Saved processed dataset to: {PROCESSED_DATA_PATH}")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=stratify_y
        )

        X_train, y_train, was_balanced = _balance_training_data(X_train, y_train)
        if was_balanced:
            logger.info("Applied random oversampling to balance training classes")

        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        logger.info("Training RandomForestClassifier (n_estimators=120, n_jobs=-1)")
        model.fit(X_train, y_train)
        logger.info("Model training completed")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test set accuracy: {accuracy:.4f}")

        cv_splits = min(3, len(X_scaled), min_class_count)
        if cv_splits < 2:
            raise ValidationError("Need at least 2 samples per class to run cross-validation")

        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_splits)
        cv_mean = cv_scores.mean()
        logger.info(f"Cross-validation scores: {cv_scores}, Mean: {cv_mean:.4f}")

        _save_confusion_matrix(y_test, y_pred)
        _save_feature_importance(model, X_scaled.columns)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(X_scaled.columns.tolist(), FEATURES_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_bounds, FEATURE_BOUNDS_PATH)
        joblib.dump({"accuracy": float(accuracy), "cv_mean": float(cv_mean), "target_encoded": target_encoded}, METRICS_PATH)
        logger.info("Model and features saved to pickle files")

        report = classification_report(y_test, y_pred)
        logger.info(f"Classification report:\n{report}")
        test_payload = _build_test_prediction_payload(y_test, y_pred, target_encoded)
        
        logger.info("Training completed successfully")
        return {
            "accuracy": accuracy,
            "cv_mean": cv_mean,
            "report": report,
            "test_predictions": test_payload["test_predictions"],
            "test_summary": test_payload["test_summary"],
            "class_comparison": test_payload["class_comparison"],
            "target_encoded": target_encoded,
            "processed_csv_path": str(PROCESSED_DATA_PATH),
            "feature_bounds": feature_bounds,
        }

    except ValidationError as exc:
        logger.error(f"Validation error: {str(exc)}")
        raise
    except Exception as exc:
        logger.error(f"Unexpected error during training: {str(exc)}", exc_info=True)
        raise TrainingError(f"Training failed: {str(exc)}")
