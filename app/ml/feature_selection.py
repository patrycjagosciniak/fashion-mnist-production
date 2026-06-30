def get_pixel_columns(dataframe):
    """Return feature columns containing Fashion MNIST pixel values.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing a `label` column and pixel feature columns.

    Returns
    -------
    list
        Names of columns used as model input features.
    """
    return [column for column in dataframe.columns if column != "label"]


def select_pixel_features(dataframe):
    """Select only pixel feature columns from a Fashion MNIST DataFrame.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing a `label` column and pixel feature columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with pixel feature columns only.
    """
    return dataframe[get_pixel_columns(dataframe)]

