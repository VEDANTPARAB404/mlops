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
    ValidationError
)
from train_model import train_model

app = FastAPI(title="ML Predictor", description="FastAPI ML Model Prediction Service")

MODEL_PATH = Path("model.pkl")
FEATURES_PATH = Path("features.pkl")
LABEL_ENCODER_PATH = Path("label_encoder.pkl")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")


# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)
TRAINING_RUNS = Counter(
    "training_runs_total",
    "Total model training runs",
    ["status"],
)
TRAINING_LATENCY = Histogram(
    "training_duration_seconds",
    "Model training latency in seconds",
)
MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Latest trained model accuracy",
)
PREDICTION_RUNS = Counter(
    "prediction_runs_total",
    "Total prediction requests",
    ["status"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Prediction latency in seconds",
)


class PredictionRequest(BaseModel):
    """Request schema for model predictions"""
    data: dict


class PredictionResponse(BaseModel):
    """Response schema for model predictions"""
    prediction: str


def error_response(status_code: int, error: str, details: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "details": details})


def load_required_artifacts():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise ModelNotFoundError("Model not found. Please train a model first.")
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)


# Middleware for request/response logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and responses."""
    request_id = str(uuid.uuid4())[:8]
    
    # Log request
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} - "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(process_time)
        
        # Log response
        logger.info(
            f"[{request_id}] Response: {response.status_code} - "
            f"Duration: {process_time:.3f}s"
        )
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        REQUEST_COUNT.labels(request.method, request.url.path, "500").inc()
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(process_time)
        logger.error(
            f"[{request_id}] Request failed with exception - Duration: {process_time:.3f}s",
            exc_info=True
        )
        raise


# Exception handlers
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {str(exc)}")
    return error_response(400, "Validation failed", str(exc))


@app.exception_handler(InvalidInputError)
async def invalid_input_error_handler(request: Request, exc: InvalidInputError):
    """Handle invalid input feature values."""
    logger.warning(f"Invalid input: {str(exc)}")
    return error_response(400, "Invalid input", str(exc))


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError):
    """Handle missing model errors."""
    logger.warning(f"Model not found: {str(exc)}")
    return error_response(404, "Model not found", str(exc))


@app.exception_handler(PredictionError)
async def prediction_error_handler(request: Request, exc: PredictionError):
    """Handle prediction errors."""
    logger.error(f"Prediction error: {str(exc)}")
    return error_response(500, "Prediction failed", str(exc))


@app.exception_handler(MLPredictorException)
async def ml_predictor_error_handler(request: Request, exc: MLPredictorException):
    """Handle general ML predictor errors."""
    logger.error(f"ML Predictor error: {str(exc)}")
    return error_response(500, "Server error", str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return error_response(500, "Internal server error", "An unexpected error occurred")


@app.get("/")
async def home(request: Request):
    logger.info("Home page accessed")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/train")
async def train_endpoint(
    file: UploadFile = File(...), 
    target: str = Form(...),
    api_key: str = Depends(verify_api_key)
):
    """Train a model and save it. Requires API key authentication."""
    logger.info(f"Training request received (authenticated) - File: {file.filename}, Target: {target}")
    training_start = time.time()

    if not file.filename.endswith(".csv"):
        raise ValidationError("File must be in CSV format")

    file_path = "uploaded.csv"
    contents = await file.read()
    with open(file_path, "wb") as uploaded_file:
        uploaded_file.write(contents)
    logger.info(f"File saved: {file_path}, Size: {len(contents)} bytes")

    try:
        results = train_model(file_path, target)
        TRAINING_RUNS.labels(status="success").inc()
    except ValidationError:
        TRAINING_RUNS.labels(status="error").inc()
        TRAINING_LATENCY.observe(time.time() - training_start)
        raise
    except Exception as exc:
        TRAINING_RUNS.labels(status="error").inc()
        TRAINING_LATENCY.observe(time.time() - training_start)
        logger.error(f"Training failed: {str(exc)}")
        raise PredictionError(f"Model training failed: {str(exc)}")

    TRAINING_LATENCY.observe(time.time() - training_start)
    MODEL_ACCURACY.set(float(results["accuracy"]))

    features = joblib.load(FEATURES_PATH)
    logger.info(f"Training completed - Features: {len(features)}, Accuracy: {results['accuracy']:.4f}")

    return JSONResponse(
        {
            "status": "success",
            "features": features,
            "accuracy": float(results["accuracy"]),
            "cv_mean": float(results["cv_mean"]),
            "report": results["report"],
            "test_predictions": results["test_predictions"],
            "test_summary": results["test_summary"],
            "class_comparison": results["class_comparison"],
        }
    )


@app.post("/predict")
async def predict(request: PredictionRequest, api_key: str = Depends(verify_api_key)):
    """Make predictions using trained model. Requires API key authentication."""
    prediction_start = time.time()
    try:
        logger.info(f"Prediction request received (authenticated) - Data keys: {list(request.data.keys())}")
        model, features = load_required_artifacts()
        
        logger.info(f"Model loaded - Expected features: {len(features)}")

        # Validate input
        try:
            input_data = [float(request.data[feature]) for feature in features]
        except KeyError as e:
            raise InvalidInputError(f"Missing required feature: {str(e)}")
        except ValueError as e:
            raise InvalidInputError(f"Invalid data type for features: {str(e)}")

        prediction = model.predict([input_data])
        logger.info(f"Raw prediction: {prediction[0]}")

        # Only decode if label encoder exists
        if LABEL_ENCODER_PATH.exists():
            le = joblib.load(LABEL_ENCODER_PATH)
            prediction = le.inverse_transform(prediction)
            logger.info(f"Decoded prediction: {prediction[0]}")

        logger.info(f"Prediction success: {prediction[0]}")
        PREDICTION_RUNS.labels(status="success").inc()
        PREDICTION_LATENCY.observe(time.time() - prediction_start)
        return JSONResponse({
            "status": "success",
            "prediction": str(prediction[0])
        })
    except (InvalidInputError, ModelNotFoundError) as e:
        PREDICTION_RUNS.labels(status="error").inc()
        PREDICTION_LATENCY.observe(time.time() - prediction_start)
        raise
    except Exception as e:
        PREDICTION_RUNS.labels(status="error").inc()
        PREDICTION_LATENCY.observe(time.time() - prediction_start)
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise PredictionError(f"Prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check called")
    return JSONResponse(
        status_code=200,
        content={"status": "healthy"}
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ML Predictor FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
