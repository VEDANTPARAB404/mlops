from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, Request, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib
import time
import uuid
import os
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from logger_config import logger
from auth import verify_api_key
from exceptions import (
    MLPredictorException,
    ModelNotFoundError,
    InvalidInputError,
    PredictionError,
    TrainingError,
    ValidationError,
)
from train_model import train_model

app = FastAPI(title="ML Predictor", description="FastAPI ML Model Prediction Service")

cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = Path("/data")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_BOUNDS_PATH = ARTIFACTS_DIR / "feature_bounds.pkl"
PROCESSED_CSV_PATH = ARTIFACTS_DIR / "processed_uploaded.csv"

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Prometheus Metrics ──────────────────────────────────────────────
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"])
TRAINING_RUNS = Counter("training_runs_total", "Total model training runs", ["status"])
TRAINING_LATENCY = Histogram("training_duration_seconds", "Model training latency")
MODEL_ACCURACY = Gauge("model_accuracy", "Latest trained model accuracy")
PREDICTION_RUNS = Counter("prediction_runs_total", "Total prediction requests", ["status"])
PREDICTION_LATENCY = Histogram("prediction_duration_seconds", "Prediction latency")


class PredictionRequest(BaseModel):
    data: dict


def load_model_and_features():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists() or not SCALER_PATH.exists() or not FEATURE_BOUNDS_PATH.exists():
        raise ModelNotFoundError("Model not found. Please train a model first.")
    return (
        joblib.load(MODEL_PATH),
        joblib.load(FEATURES_PATH),
        joblib.load(SCALER_PATH),
        joblib.load(FEATURE_BOUNDS_PATH),
    )


def _validate_input_bounds(input_map: dict, feature_bounds: dict):
    violations = []
    for feature, value in input_map.items():
        bounds = feature_bounds.get(feature)
        if not bounds:
            continue
        min_v = float(bounds.get("min", value))
        max_v = float(bounds.get("max", value))
        if value < min_v or value > max_v:
            violations.append(f"{feature}: {value} not in [{min_v}, {max_v}]")

    if violations:
        details = "; ".join(violations[:5])
        if len(violations) > 5:
            details += f"; and {len(violations) - 5} more"
        raise InvalidInputError(f"Out-of-bound feature value(s): {details}")


def load_persisted_metrics():
    if METRICS_PATH.exists():
        metrics = joblib.load(METRICS_PATH)
        if isinstance(metrics, dict) and "accuracy" in metrics:
            try:
                MODEL_ACCURACY.set(float(metrics["accuracy"]))
            except (TypeError, ValueError):
                logger.warning("Persisted accuracy value is invalid")
            return metrics
    return {}


def should_decode_prediction() -> bool:
    if METRICS_PATH.exists():
        metrics = joblib.load(METRICS_PATH)
        if isinstance(metrics, dict) and "target_encoded" in metrics:
            return bool(metrics["target_encoded"])
    return LABEL_ENCODER_PATH.exists()


load_persisted_metrics()


# ── Middleware ──────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    start, status = time.time(), "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception:
        raise
    finally:
        duration = time.time() - start
        REQUEST_COUNT.labels(request.method, request.url.path, status).inc()
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)
        logger.info(f"[{request_id}] {status} in {duration:.3f}s")


# ── Exception Handlers ─────────────────────────────────────────────
_ERROR_MAP = {
    ValidationError:    (400, "Validation failed"),
    InvalidInputError:  (400, "Invalid input"),
    ModelNotFoundError: (404, "Model not found"),
    PredictionError:    (500, "Prediction failed"),
    TrainingError:      (500, "Training failed"),
    MLPredictorException: (500, "Server error"),
}

for _exc_class, (_code, _label) in _ERROR_MAP.items():
    def _make_handler(code=_code, label=_label):
        async def handler(request: Request, exc):
            logger.warning(f"{label}: {exc}")
            return JSONResponse(status_code=code, content={"error": label, "details": str(exc)})
        return handler
    app.add_exception_handler(_exc_class, _make_handler())


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error", "details": "An unexpected error occurred"})


# ── Routes ──────────────────────────────────────────────────────────
@app.get("/")
async def home(request: Request):
    # Use keyword args for compatibility with newer Starlette TemplateResponse signature.
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/processed-csv")
async def download_processed_csv(api_key: str = Depends(verify_api_key)):
    if not PROCESSED_CSV_PATH.exists():
        raise ModelNotFoundError("Processed CSV not found. Train a model first.")

    return FileResponse(
        path=PROCESSED_CSV_PATH,
        filename="processed_uploaded.csv",
        media_type="text/csv",
    )


@app.post("/train")
async def train_endpoint(
    file: UploadFile | None = File(None),
    target: str = Form(...),
    use_preprocessed_data: bool = Form(False),
    api_key: str = Depends(verify_api_key),
):
    logger.info(
        f"Training request - File: {getattr(file, 'filename', None)}, Target: {target}, "
        f"use_preprocessed_data={use_preprocessed_data}"
    )

    if use_preprocessed_data:
        if not PROCESSED_CSV_PATH.exists():
            raise ValidationError("Preprocessed CSV not found. Train once with the raw CSV first.")
        training_path = str(PROCESSED_CSV_PATH)
        logger.info(f"Using preprocessed CSV for training: {training_path}")
    else:
        if file is None or not file.filename:
            raise ValidationError("Please upload a CSV file")
        if not file.filename.endswith(".csv"):
            raise ValidationError("File must be in CSV format")

        contents = await file.read()
        Path("uploaded.csv").write_bytes(contents)
        training_path = "uploaded.csv"
        logger.info(f"Using uploaded raw CSV for training: {training_path}")

    start = time.time()
    try:
        results = train_model(training_path, target)
        TRAINING_RUNS.labels(status="success").inc()
    except ValidationError:
        TRAINING_RUNS.labels(status="error").inc()
        raise
    except Exception as e:
        TRAINING_RUNS.labels(status="error").inc()
        raise TrainingError(f"Model training failed: {e}")
    finally:
        TRAINING_LATENCY.observe(time.time() - start)

    MODEL_ACCURACY.set(float(results["accuracy"]))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "accuracy": float(results["accuracy"]),
            "cv_mean": float(results["cv_mean"]),
            "target_encoded": bool(results.get("target_encoded", False)),
        },
        METRICS_PATH,
    )
    features = joblib.load(FEATURES_PATH)
    logger.info(f"Training done - {len(features)} features, accuracy: {results['accuracy']:.4f}")

    return {
        "status": "success",
        "features": features,
        "accuracy": float(results["accuracy"]),
        "cv_mean": float(results["cv_mean"]),
        "report": results["report"],
        "test_predictions": results["test_predictions"],
        "test_summary": results["test_summary"],
        "class_comparison": results["class_comparison"],
        "processed_csv_path": results.get("processed_csv_path"),
        "processed_csv_download_url": "/processed-csv",
        "feature_bounds": results.get("feature_bounds", {}),
        "training_source": "preprocessed" if use_preprocessed_data else "raw",
    }


@app.post("/predict")
async def predict(request: PredictionRequest, api_key: str = Depends(verify_api_key)):
    start = time.time()
    try:
        model, features, scaler, feature_bounds = load_model_and_features()

        try:
            input_map = {f: float(request.data[f]) for f in features}
        except KeyError as e:
            raise InvalidInputError(f"Missing feature: {e}")
        except ValueError as e:
            raise InvalidInputError(f"Invalid data type: {e}")

        _validate_input_bounds(input_map, feature_bounds)

        input_frame = pd.DataFrame([input_map], columns=features)
        input_scaled = scaler.transform(input_frame)

        prediction = model.predict(input_scaled)

        if should_decode_prediction() and LABEL_ENCODER_PATH.exists():
            try:
                prediction = joblib.load(LABEL_ENCODER_PATH).inverse_transform(prediction)
            except ValueError as exc:
                logger.warning(f"Skipping label decoding for prediction result: {exc}")

        PREDICTION_RUNS.labels(status="success").inc()
        return {"status": "success", "prediction": str(prediction[0])}

    except (InvalidInputError, ModelNotFoundError):
        PREDICTION_RUNS.labels(status="error").inc()
        raise
    except Exception as e:
        PREDICTION_RUNS.labels(status="error").inc()
        raise PredictionError(f"Prediction failed: {e}")
    finally:
        PREDICTION_LATENCY.observe(time.time() - start)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
