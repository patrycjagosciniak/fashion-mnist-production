import pandas as pd
from PIL import Image
from torchvision import transforms

from app.dataset import FashionMNISTCSV


def test_fashion_mnist_csv_returns_correct_length():
    dataframe = pd.DataFrame({
        "label": [0, 1],
        **{f"pixel{i}": [0, 255] for i in range(784)},
    })

    dataset = FashionMNISTCSV(dataframe)

    assert len(dataset) == 2


def test_fashion_mnist_csv_returns_image_and_label():
    dataframe = pd.DataFrame({
        "label": [3],
        **{f"pixel{i}": [128] for i in range(784)},
    })

    dataset = FashionMNISTCSV(dataframe)

    image, label = dataset[0]

    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert image.size == (28, 28)
    assert label == 3


def test_fashion_mnist_csv_applies_transform():
    dataframe = pd.DataFrame({
        "label": [2],
        **{f"pixel{i}": [128] for i in range(784)},
    })

    transform = transforms.ToTensor()
    dataset = FashionMNISTCSV(dataframe, transform=transform)

    image, label = dataset[0]

    assert image.shape == (1, 28, 28)
    assert label == 2