from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, Request, Depends, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import time
import uuid
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from logger_config import logger
from auth import verify_api_key
from exceptions import (
    MLPredictorException,
    ModelNotFoundError,
    InvalidInputError,
    PredictionError,
    ValidationError,
)
from train_model import train_model

app = FastAPI(title="ML Predictor", description="FastAPI ML Model Prediction Service")

ARTIFACTS_DIR = Path("/data")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "features.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.pkl"

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
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise ModelNotFoundError("Model not found. Please train a model first.")
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)


def load_persisted_metrics():
    if METRICS_PATH.exists():
        metrics = joblib.load(METRICS_PATH)
        if isinstance(metrics, dict) and "accuracy" in metrics:
            try:
                MODEL_ACCURACY.set(float(metrics["accuracy"]))
            except (TypeError, ValueError):
                logger.warning("Persisted accuracy value is invalid")


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


@app.post("/train")
async def train_endpoint(
    file: UploadFile = File(...),
    target: str = Form(...),
    api_key: str = Depends(verify_api_key),
):
    logger.info(f"Training request - File: {file.filename}, Target: {target}")
    if not file.filename or not file.filename.endswith(".csv"):
        raise ValidationError("File must be in CSV format")

    contents = await file.read()
    Path("uploaded.csv").write_bytes(contents)

    start = time.time()
    try:
        results = train_model("uploaded.csv", target)
        TRAINING_RUNS.labels(status="success").inc()
    except ValidationError:
        TRAINING_RUNS.labels(status="error").inc()
        raise
    except Exception as e:
        TRAINING_RUNS.labels(status="error").inc()
        raise PredictionError(f"Model training failed: {e}")
    finally:
        TRAINING_LATENCY.observe(time.time() - start)

    MODEL_ACCURACY.set(float(results["accuracy"]))
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"accuracy": float(results["accuracy"]), "cv_mean": float(results["cv_mean"])}, METRICS_PATH)
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
    }


@app.post("/predict")
async def predict(request: PredictionRequest, api_key: str = Depends(verify_api_key)):
    start = time.time()
    try:
        model, features = load_model_and_features()

        try:
            input_data = [float(request.data[f]) for f in features]
        except KeyError as e:
            raise InvalidInputError(f"Missing feature: {e}")
        except ValueError as e:
            raise InvalidInputError(f"Invalid data type: {e}")

        prediction = model.predict([input_data])

        if LABEL_ENCODER_PATH.exists():
            prediction = joblib.load(LABEL_ENCODER_PATH).inverse_transform(prediction)

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
