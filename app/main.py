import json
from pathlib import Path

from fastapi import FastAPI

from app.ml.class_names import FASHION_MNIST_CLASS_NAMES
from app.model import load_model
from app.predict import predict
from app.schemas import PredictionInput, PredictionOutput

app = FastAPI(
    title="Fashion MNIST Model API",
    description=(
        "API for serving a trained Custom CNN model "
        "for Fashion MNIST classification."
    ),
    version="0.1.0",
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_custom_cnn.pth"
PREPROCESSING_PARAMS_PATH = BASE_DIR / "models" / "preprocessing_params.json"

DEVICE = "cpu"


with open(PREPROCESSING_PARAMS_PATH, "r") as file:
    preprocessing_params = json.load(file)

MEAN = preprocessing_params["fashion_mnist_mean"]
STD = preprocessing_params["fashion_mnist_std"]


model = load_model(
    model_path=MODEL_PATH,
    device=DEVICE,
)


@app.get("/")
def root() -> dict:
    return {"message": "Fashion MNIST Model API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict_image(request: PredictionInput) -> dict:
    result = predict(
        model=model,
        pixels=request.pixels,
        class_names=FASHION_MNIST_CLASS_NAMES,
        mean=MEAN,
        std=STD,
        device=DEVICE,
    )

    return result