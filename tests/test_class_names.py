import pytest

from app.ml.class_names import FASHION_MNIST_CLASS_NAMES, get_class_name


def test_fashion_mnist_has_10_classes():
    assert len(FASHION_MNIST_CLASS_NAMES) == 10


def test_get_class_name_returns_correct_name():
    assert get_class_name(0) == "T-shirt/top"
    assert get_class_name(9) == "Ankle boot"


def test_get_class_name_raises_error_for_invalid_class_id():
    with pytest.raises(KeyError):
        get_class_name(10)