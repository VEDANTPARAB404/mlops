# ML Predictor - API Documentation

## Overview
FastAPI-based ML prediction service with structured logging and API key authentication.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and update with your API keys:
```bash
cp .env.example .env
```

Edit `.env`:
```
API_KEYS=your-secret-key-1,your-secret-key-2
ENV=development
HOST=0.0.0.0
PORT=8000
```

### 3. Run the Server
```bash
python app.py
```

Server runs at `http://localhost:8000`

---

## Endpoints

### Public Endpoints

#### GET `/`
Home page with UI for model training and predictions.
- **Authentication**: Not required
- **Response**: HTML page

#### GET `/health`
Health check endpoint.
- **Authentication**: Not required
- **Response**:
```json
{
  "status": "healthy"
}
```

---

### Protected Endpoints (Require API Key)

#### POST `/train`
Train a new model from CSV data.

**Authentication**: Header `X-API-Key: <your-api-key>`

**Request**:
- `file` (multipart/form-data): CSV file
- `target` (form-data): Target column name

**Example**:
```bash
curl -X POST http://localhost:8000/train \
  -H "X-API-Key: your-secret-key-1" \
  -F "target=label" \
  -F "file=@dataset.csv"
```

**Response** (200 OK):
```json
{
  "status": "success",
  "features": ["feature1", "feature2", "feature3"],
  "accuracy": 0.9234,
  "cv_mean": 0.9156,
  "report": "... classification report ..."
}
```

**Error Responses**:
- 400 Bad Request: Invalid file format or missing target
- 401 Unauthorized: Missing or invalid API key
- 500 Internal Server Error: Training failed

---

#### POST `/predict`
Make predictions using the trained model.

**Authentication**: Header `X-API-Key: <your-api-key>`

**Request Body** (JSON):
```json
{
  "data": {
    "feature1": 1.5,
    "feature2": 2.3,
    "feature3": 0.8
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-secret-key-1" \
  -H "Content-Type: application/json" \
  -d '{"data": {"feature1": 1.5, "feature2": 2.3, "feature3": 0.8}}'
```

**Response** (200 OK):
```json
{
  "status": "success",
  "prediction": "class_A"
}
```

**Error Responses**:
- 400 Bad Request: Missing required features or invalid data types
- 401 Unauthorized: Missing or invalid API key
- 404 Not Found: Model not trained yet
- 500 Internal Server Error: Prediction failed

---

## Authentication

### API Key Setup
1. Generate secure API keys
2. Add them to `.env` file (comma-separated):
   ```
   API_KEYS=key1,key2,key3
   ```
3. Include in requests via `X-API-Key` header

### Using API Key
```bash
# Include in all protected endpoints
-H "X-API-Key: your-api-key"
```

---

## Logging

### Log Files
- **Location**: `logs/app.log`
- **Format**: Rotating file handler (max 10MB, keeps 5 backups)
- **Levels**: DEBUG (file) and INFO (console)

### Example Log Entries
```
2026-02-22 10:15:30 - ml_predictor - INFO - [app.py:78] - POST /train - Client: 127.0.0.1
2026-02-22 10:15:32 - ml_predictor - INFO - [train_model.py:25] - Training RandomForestClassifier (n_estimators=300)
2026-02-22 10:15:35 - ml_predictor - INFO - [app.py:89] - Response: 200 - Duration: 5.234s
```

---

## Error Handling

### Exception Types
- `ValidationError`: Invalid input data or missing fields
- `ModelNotFoundError`: Model file not found
- `PredictionError`: Prediction execution failed
- `TrainingError`: Model training failed
- `InvalidInputError`: Feature data validation failed

All errors return structured JSON responses:
```json
{
  "error": "Error type",
  "details": "Error details"
}
```

---

## Example Workflow

### 1. Train a Model
```bash
# Assuming you have a dataset.csv with a 'target' column
curl -X POST http://localhost:8000/train \
  -H "X-API-Key: sk-test-12345678901234567890" \
  -F "target=target" \
  -F "file=@dataset.csv"
```

### 2. Get Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: sk-test-12345678901234567890" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "feature1": 1.2,
      "feature2": 3.4,
      "feature3": 5.6
    }
  }'
```

---

## Monitoring with Prometheus and Grafana

The project includes built-in monitoring using Prometheus and Grafana to track API and model performance metrics.

### Metrics Endpoint

#### GET `/metrics`
Prometheus scrape endpoint (no authentication required).
- **Format**: Prometheus text exposition format

### Available Metrics

- `http_requests_total{method,endpoint,status_code}`: Total API requests
- `http_request_duration_seconds{method,endpoint}`: HTTP request latency histogram
- `training_runs_total{status}`: Count of model training runs (`success` or `error`)
- `training_duration_seconds`: Model training latency histogram
- `model_accuracy`: Latest training accuracy gauge
- `prediction_runs_total{status}`: Count of prediction requests (`success` or `error`)
- `prediction_duration_seconds`: Prediction latency histogram

### Running the Monitoring Stack

Use Docker Compose to run the application with Prometheus and Grafana:

```bash
docker compose -f docker-compose.monitoring.yml up --build
```

This starts:
- **App**: `http://localhost:8000`
- **Prometheus UI**: `http://localhost:9090` (query metrics and check targets)
- **Grafana UI**: `http://localhost:3000` (dashboards)

### Accessing Grafana

1. Open `http://localhost:3000`
2. Login with default credentials:
   - Username: `admin`
   - Password: `admin`
3. Navigate to **Dashboards** → **ML Predictor** → **ML Predictor - Monitoring Overview**

The dashboard shows:
- Request rate (requests/sec)
- Model accuracy (latest value)
- P95 request latency
- Prediction latency
- Request rate by endpoint

### Verifying Metrics Collection

1. Make a health check request:
   ```bash
   curl http://localhost:8000/health
   ```

2. View collected metrics:
   ```bash
   curl http://localhost:8000/metrics
   ```

3. Query metrics in Prometheus UI:
   - Go to `http://localhost:9090`
   - Check **Status** → **Targets** to verify the app is being scraped
   - Run queries like `http_requests_total` in the query editor

---

## Kubernetes Deployment

The repository includes Kubernetes manifests in `k8s/` for app deployment, service exposure, persistent model storage, and Prometheus Operator monitoring resources.

### Prerequisites

- A running Kubernetes cluster
- `kubectl` configured for that cluster
- Prometheus Operator / kube-prometheus-stack installed (for `ServiceMonitor` and `PrometheusRule` CRDs)
- Docker image pushed to a registry reachable by your cluster (or preloaded in local cluster runtime)

### 1. Build and Push Image

```bash
docker build -t <your-registry>/ml-predictor:latest .
docker push <your-registry>/ml-predictor:latest
```

Update `k8s/deployment.yaml` image field to your published image.

### 2. Configure Secret

Edit `k8s/secret.yaml` and set a strong value for `API_KEYS`.

### 3. Apply Manifests

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/servicemonitor.yaml
kubectl apply -f k8s/prometheusrule.yaml
```

### 4. Verify Application

```bash
kubectl get pods -n ml-predictor
kubectl get svc -n ml-predictor
kubectl get endpoints -n ml-predictor ml-predictor
kubectl logs -n ml-predictor deploy/ml-predictor
```

### 5. Verify Monitoring Objects

```bash
kubectl get servicemonitor -n ml-predictor
kubectl get prometheusrule -n ml-predictor
```

Check in Prometheus that target `ml-predictor` is `UP` and run queries:

- `http_requests_total`
- `rate(http_requests_total[5m])`
- `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))`

### 6. Access the Service Locally

```bash
kubectl port-forward -n ml-predictor svc/ml-predictor 8000:80
```

Then open `http://localhost:8000`.

---

## Project Structure

```
ml-predictor/
├── app.py                 # Main FastAPI application
├── docker-compose.monitoring.yml   # App + Prometheus + Grafana stack
├── train_model.py         # Model training logic
├── auth.py                # API key authentication
├── logger_config.py       # Logging configuration
├── exceptions.py          # Custom exception classes
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (gitignored)
├── .env.example           # Environment template
├── .gitignore             # Git ignore rules
├── logs/                  # Log files (generated)
├── monitoring/
│   ├── prometheus.yml     # Prometheus scrape configuration
│   └── grafana/
│       ├── dashboards/
│       │   └── ml_predictor_overview.json
│       └── provisioning/
│           ├── datasources/
│           │   └── datasource.yml
│           └── dashboards/
│               └── dashboard.yml
├── static/                # Static files (plots, etc.)
├── templates/
│   └── index.html         # Frontend UI
└── README.md              # This file
```

---

## Known Limitations & Future Work

- Single-threaded training (no concurrent model training)
- Model stored in-memory; consider Redis for production
- No database for training history
- Kubernetes deployment (Expt 12)

---

## Support

For issues or questions, check the logs directory for detailed error messages.
