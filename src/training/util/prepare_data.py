import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import BoxCox, Scaler
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

DT_FORMAT = "%Y-%m-%d"


def interpolate_outliers(
    data: pd.DataFrame, column: str, window: int = 30, n_sigma: float = 3
):
    """
    Linear interpolates series passed as column in the data.
    """
    rolling_mean = data[column].rolling(window).mean()
    rolling_std = data[column].rolling(window).std()
    upper, lower = (
        rolling_mean + n_sigma * rolling_std,
        rolling_mean - n_sigma * rolling_std,
    )
    mask_outlier = data.loc[
        ((data.unemp > upper) | (data.unemp < lower)) & (data.index.year < 2009)
    ]
    # Interpolate those outliers.
    data.loc[mask_outlier.index, column] = np.nan
    data.loc[mask_outlier.index, column] = data.unemp.interpolate()
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes nans and imputes outliers by linear interpolation.
    """
    data.Period = data.Period.apply(lambda x: dt.datetime.strptime(x, DT_FORMAT))
    data.set_index("Period", inplace=True)
    # Linear interpolation.
    data.loc[:, "unemp"] = data.unemp.interpolate()
    # Get indexes of the outliers.
    data = interpolate_outliers(data, "unemp")
    return data


def transform_to_ts(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(series)


def partition(series: TimeSeries, train_ratio: float) -> Tuple[TimeSeries, TimeSeries]:
    return series.split_before(train_ratio)


def preprocess(
    series: TimeSeries,
    transformer: Pipeline,
    train: TimeSeries,
    test: TimeSeries,
) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
    # Avoid fitting the transformer on the validation set.
    train_transformed = transformer.fit_transform(train)
    test_transformed, series_transformed = (
        transformer.transform(ser) for ser in (test, series)
    )
    return series_transformed, train_transformed, test_transformed
