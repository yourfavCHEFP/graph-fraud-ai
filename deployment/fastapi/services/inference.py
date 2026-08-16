from src.inference.predictor import FraudPredictor

_predictor = None


def get_predictor():

    global _predictor

    if _predictor is None:
        _predictor = FraudPredictor()

    return _predictor
