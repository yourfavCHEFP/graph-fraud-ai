import logging

from src.inference.predictor import FraudPredictor

logger = logging.getLogger("graph-fraud-api")

_predictor = None


def get_predictor():
    """Return the singleton predictor.

    FIX: this docstring previously said the predictor is "created only
    when /predict is called, not by Render readiness probes" -- that was
    the actual bug (see project's debugging summary): loading the model +
    graph + running the one-time full-graph forward pass all happened
    inside the first live HTTP request, where a slow/OOM'd load manifests
    as a silent 502 with nothing in the logs. It's now created explicitly
    at app startup (see deployment/fastapi/app.py's startup event) -- this
    function just returns the already-created singleton, or raises clearly
    if startup never actually completed.
    """
    if _predictor is None:
        raise RuntimeError(
            "Predictor was never initialized at startup. Check the startup "
            "logs for the actual load failure -- the app should not have "
            "reached a state where /predict is reachable but the predictor "
            "is still None."
        )
    return _predictor


def initialize_predictor():
    """Called once from app.py's startup event. Raises loudly on failure."""
    global _predictor
    logger.info("Initializing FraudPredictor at startup...")
    _predictor = FraudPredictor()
    logger.info("FraudPredictor ready: %s", _predictor.model_name)
    return _predictor
