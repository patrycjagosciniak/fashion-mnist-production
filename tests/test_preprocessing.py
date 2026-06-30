import pandas as pd
import torch
from PIL import Image

from app.ml.preprocessing import calculate_mean_std, get_cnn_transform


def test_calculate_mean_std_returns_scaled_values():
    dataframe = pd.DataFrame({
        "label": [0, 1],
        "pixel1": [0, 255],
        "pixel2": [255, 0],
    })

    mean, std = calculate_mean_std(dataframe)

    assert mean == 0.5
    assert std == 0.5


def test_get_cnn_transform_returns_tensor_with_correct_shape():
    image = Image.new("L", (28, 28), color=128)

    transform = get_cnn_transform(mean=0.5, std=0.5)
    transformed_image = transform(image)

    assert isinstance(transformed_image, torch.Tensor)
    assert transformed_image.shape == (1, 28, 28)