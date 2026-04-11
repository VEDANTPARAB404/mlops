import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from logger_config import logger
from exceptions import TrainingError, ValidationError


LABEL_ENCODER_PATH = Path("label_encoder.pkl")
ARTIFACTS_DIR = Path("/data")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.pkl"
CONFUSION_MATRIX_PATH = "static/confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = "static/feature_importance.png"


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

        X = df.drop(target_column, axis=1)
        y = df[target_column]
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

        X = pd.get_dummies(X)
        logger.info(f"After one-hot encoding: {X.shape}")

        min_class_count = int(y.value_counts().min())
        y, target_encoded = _encode_target(y)

        logger.info("Splitting data into train/test sets (80/20)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

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

        cv_splits = min(3, len(X), min_class_count)
        if cv_splits < 2:
            raise ValidationError("Need at least 2 samples per class to run cross-validation")

        cv_scores = cross_val_score(model, X, y, cv=cv_splits)
        cv_mean = cv_scores.mean()
        logger.info(f"Cross-validation scores: {cv_scores}, Mean: {cv_mean:.4f}")

        _save_confusion_matrix(y_test, y_pred)
        _save_feature_importance(model, X.columns)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(X.columns.tolist(), FEATURES_PATH)
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
        }

    except ValidationError as exc:
        logger.error(f"Validation error: {str(exc)}")
        raise
    except Exception as exc:
        logger.error(f"Unexpected error during training: {str(exc)}", exc_info=True)
        raise TrainingError(f"Training failed: {str(exc)}")
