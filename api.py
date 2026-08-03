"""An optional FastAPI version of the sentiment prediction service."""

from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field


# This API is optional. The Streamlit app already works without it.
PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / "best_sentiment_system.joblib"

LOW_CONFIDENCE_THRESHOLD = 0.60
POSSIBLE_NEGATIVE_THRESHOLD = 0.35


artifact = joblib.load(MODEL_PATH)
model = artifact["system"]

app = FastAPI(
    title="Amazon Review Sentiment API",
    description="A small demo API for musical instrument reviews.",
    version="1.0.0",
)


class ReviewRequest(BaseModel):
    """The text that a user sends to the prediction API."""

    review_text: str = Field(
        min_length=1,
        examples=[
            "The sound quality is great, but the case feels cheap."
        ],
    )


@app.get("/health")
def health_check():
    """A simple check that the API and model loaded correctly."""
    return {
        "status": "ok",
        "model": "TF-IDF + Logistic Regression",
    }


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    """Return the label, confidence, probabilities, and review status."""
    review_text = request.review_text.strip()
    probabilities = model.predict_proba([review_text])[0]

    probability_by_class = {
        label: float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    predicted_label = max(
        probability_by_class,
        key=probability_by_class.get,
    )
    confidence = probability_by_class[predicted_label]
    negative_probability = probability_by_class["Negative"]

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        review_status = "Needs Review: Low Confidence"
    elif negative_probability >= POSSIBLE_NEGATIVE_THRESHOLD:
        review_status = "Needs Review: Possible Negative"
    else:
        review_status = "Auto Accepted"

    return {
        "review_text": review_text,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probability_by_class,
        "review_status": review_status,
    }
