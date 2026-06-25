from pydantic import BaseModel


class PredictionInput(BaseModel):
    """Input data required for Fashion MNIST model prediction."""

    pixels: list[float]


class PredictionOutput(BaseModel):
    """Prediction response returned by the Fashion MNIST model."""

    predicted_class: int
    class_name: str
    confidence: float