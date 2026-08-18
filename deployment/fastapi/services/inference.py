from src.inference.predictor import FraudPredictor

_predictor = None


def get_predictor():
    """Return a singleton predictor.

    The object is created only when /predict is called, not by Render readiness
    probes. This prevents health checks from triggering heavy graph inference.
    """
    global _predictor

    if _predictor is None:
        _predictor = FraudPredictor()

    return _predictor
