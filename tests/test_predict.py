import torch

from app.predict import predict, prepare_pixels


class DummyModel(torch.nn.Module):
    """Simple model used only for testing prediction logic."""

    def forward(self, x):
        batch_size = x.shape[0]
        logits = torch.zeros((batch_size, 10))
        logits[:, 3] = 10.0
        return logits


def test_prepare_pixels_returns_tensor_with_expected_shape():
    pixels = [0] * 784

    tensor = prepare_pixels(pixels, mean=0.0, std=1.0)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 28, 28)


def test_predict_returns_class_name_and_confidence():
    model = DummyModel()
    pixels = [0] * 784
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    result = predict(
        model=model,
        pixels=pixels,
        class_names=class_names,
        mean=0.0,
        std=1.0,
    )

    assert result["predicted_class"] == 3
    assert result["class_name"] == "Dress"
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0