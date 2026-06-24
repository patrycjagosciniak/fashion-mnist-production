import torch
import torch.nn as nn


class CustomCNN_1(nn.Module):
    """Custom CNN architecture selected as the final model for Fashion MNIST.

    The model takes one-channel 28x28 grayscale images as input and returns
    logits for 10 Fashion MNIST classes.

    Parameters
    ----------
    num_classes : int, default=10
        Number of output classes.
    dropout_rate : float, default=0.3
        Dropout probability used in the classifier part.
    """

    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(CustomCNN_1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """Run a forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images with shape `(batch_size, 1, 28, 28)`.

        Returns
        -------
        torch.Tensor
            Raw class scores for each input image.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(model_path, device="cpu", num_classes=10, dropout_rate=0.3):
    """Load the trained CustomCNN_1 model from saved weights.

    Parameters
    ----------
    model_path : str
        Path to the saved `.pth` file with model weights.
    device : str, default="cpu"
        Device used to load the model.
    num_classes : int, default=10
        Number of output classes.
    dropout_rate : float, default=0.3
        Dropout probability used in the model architecture.

    Returns
    -------
    CustomCNN_1
        Loaded model in evaluation mode.
    """
    model = CustomCNN_1(
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.to(device)
    model.eval()

    return model