from typing import Annotated

from pydantic import BaseModel, Field

PixelValue = Annotated[float, Field(ge=0, le=255)]


class PredictionInput(BaseModel):
    """Input data required for Fashion MNIST model prediction."""

    pixels: list[PixelValue] = Field(min_length=784, max_length=784)


class PredictionOutput(BaseModel):
    """Prediction response returned by the Fashion MNIST model."""

    predicted_class: int
    class_name: str
    confidence: float
