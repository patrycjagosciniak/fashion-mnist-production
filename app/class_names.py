FASHION_MNIST_CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def get_class_name(class_id: int) -> str:
    """
    Return Fashion MNIST class name for a given class id.
    """
    return FASHION_MNIST_CLASS_NAMES[class_id]