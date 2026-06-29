from typing import Sequence, Union

import torch

from app.ml.class_names import get_class_name


def prepare_pixels(
    pixels: Sequence[Union[int, float]],
    mean: float = 0.0,
    std: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convert flattened Fashion MNIST pixels into a tensor ready for CNN prediction.

    Parameters
    ----------
    pixels : Sequence[int | float]
        Flattened list of 784 pixel values.
    mean : float
        Mean used for normalization.
    std : float
        Standard deviation used for normalization.
    device : str
        Device used for prediction.

    Returns
    -------
    torch.Tensor
        Tensor with shape (1, 1, 28, 28).
    """
    tensor = torch.tensor(pixels, dtype=torch.float32)
    tensor = tensor.view(1, 1, 28, 28)
    tensor = tensor / 255.0
    tensor = (tensor - mean) / std

    return tensor.to(device)


def predict(
    model: torch.nn.Module,
    pixels: Sequence[Union[int, float]],
    class_names: Sequence[str],
    mean: float = 0.0,
    std: float = 1.0,
    device: str = "cpu",
) -> dict:
    """
    Run prediction for a single Fashion MNIST image.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model used for prediction.
    pixels : Sequence[int | float]
        Flattened list of 784 pixel values.
    class_names : Sequence[str]
        Names of Fashion MNIST classes.
    mean : float
        Mean used for normalization.
    std : float
        Standard deviation used for normalization.
    device : str
        Device used for prediction.

    Returns
    -------
    dict
        Prediction result with class index, class name and confidence.
    """
    model.to(device)
    model.eval()

    input_tensor = prepare_pixels(
        pixels=pixels,
        mean=mean,
        std=std,
        device=device,
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    predicted_class_id = int(predicted_class.item())

    return {
        "predicted_class": predicted_class_id,
        "class_name": get_class_name(predicted_class_id),
        "confidence": float(confidence.item()),
    }