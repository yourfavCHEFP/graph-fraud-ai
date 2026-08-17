"""Phase 15 model explanation helpers."""

def explain_prediction(probability, threshold):
    risk = []

    if probability >= threshold:
        risk.append("model confidence exceeded fraud threshold")
    else:
        risk.append("model confidence below fraud threshold")

    return {
        "risk_level": "high" if probability >= threshold else "low",
        "risk_factors": risk,
    }
