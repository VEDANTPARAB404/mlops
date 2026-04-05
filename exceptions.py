"""Custom exception classes for ML Predictor."""


class MLPredictorException(Exception):
    """Base exception class for ML Predictor."""
    ...


class ModelNotFoundError(MLPredictorException):
    """Raised when model file is not found."""
    ...


class InvalidInputError(MLPredictorException):
    """Raised when input data is invalid."""
    ...


class TrainingError(MLPredictorException):
    """Raised when model training fails."""
    ...


class PredictionError(MLPredictorException):
    """Raised when prediction fails."""
    ...


class ValidationError(MLPredictorException):
    """Raised when request validation fails."""
    ...
