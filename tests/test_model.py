import torch

from app.model import CustomCNN_1, load_model


def test_custom_cnn_forward_output_shape():
    model = CustomCNN_1()

    x = torch.randn(2, 1, 28, 28)
    output = model(x)

    assert output.shape == (2, 10)


def test_load_model_returns_eval_model(tmp_path):
    model = CustomCNN_1()
    model_path = tmp_path / "test_model.pth"

    torch.save(model.state_dict(), model_path)

    loaded_model = load_model(model_path)

    assert isinstance(loaded_model, CustomCNN_1)
    assert loaded_model.training is False