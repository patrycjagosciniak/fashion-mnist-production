from torchvision import transforms


def calculate_mean_std(dataframe):
    """Calculate mean and standard deviation for Fashion MNIST pixel values.

    The calculation should be done only on the training data to avoid data
    leakage from validation or test sets.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Training DataFrame with a `label` column and pixel columns.

    Returns
    -------
    tuple
        Mean and standard deviation scaled to the 0-1 range.
    """
    pixel_columns = dataframe.drop("label", axis=1)

    mean = pixel_columns.values.mean() / 255.0
    std = pixel_columns.values.std() / 255.0

    return mean, std


def get_cnn_transform(mean, std):
    """Create image transformations for the custom CNN model.

    Parameters
    ----------
    mean : float
        Mean pixel value calculated on the training data.
    std : float
        Standard deviation of pixel values calculated on the training data.

    Returns
    -------
    torchvision.transforms.Compose
        Transform pipeline converting images to tensors and normalizing them.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[mean],
            std=[std]
        )
    ])