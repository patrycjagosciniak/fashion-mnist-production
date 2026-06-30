import pandas as pd

from app.ml.feature_selection import get_pixel_columns, select_pixel_features


def test_get_pixel_columns_returns_only_pixel_columns():
    dataframe = pd.DataFrame({
        "label": [0],
        "pixel1": [10],
        "pixel2": [20],
    })

    result = get_pixel_columns(dataframe)

    assert result == ["pixel1", "pixel2"]


def test_select_pixel_features_returns_only_pixel_values():
    dataframe = pd.DataFrame({
        "label": [0, 1],
        "pixel1": [10, 30],
        "pixel2": [20, 40],
    })

    result = select_pixel_features(dataframe)

    assert list(result.columns) == ["pixel1", "pixel2"]
    assert result.shape == (2, 2)