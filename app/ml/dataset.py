import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FashionMNISTCSV(Dataset):
    """PyTorch Dataset for Fashion MNIST data stored in CSV format.

    The dataset expects a pandas DataFrame where the `label` column contains
    class labels and the remaining columns contain pixel values for 28x28
    grayscale images.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame with labels and image pixels.
    transform : callable, optional
        Transformations applied to each image before returning it.
    """

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

        self.labels = self.dataframe["label"].values
        self.images = self.dataframe.drop("label", axis=1).values

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Return one image and its label by index.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple
            A tuple containing the transformed image and its integer label.
        """
        image = self.images[idx].reshape(28, 28).astype(np.uint8)
        label = int(self.labels[idx])

        image = Image.fromarray(image, mode="L")

        if self.transform:
            image = self.transform(image)

        return image, label