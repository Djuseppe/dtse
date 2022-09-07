import datetime as dt
from typing import Any, Tuple

import darts
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.pipeline import Pipeline
from darts.metrics import mape, mse
from matplotlib import pyplot as plt

plt.style.use("ggplot")


def get_prediction(
    model: Any,
    pred_horizon: int,
    train_series: TimeSeries,
    predict_from: bool,
    num_samples: int,
) -> TimeSeries:
    if predict_from:
        return model.predict(
            n=pred_horizon, series=train_series, num_samples=num_samples
        )
    return model.predict(n=pred_horizon)


# def evaluate_model(
#     model: Any,
#     train_series: TimeSeries,
#     test_series: TimeSeries,
#     pred_horizon: int,
#     predict_from_train: bool = False,
#     num_samples: int = 100,
#     transformer: Pipeline = None,
#     metrics: dict[str : darts.metrics] = None,
#     plot: bool = True,
#     n_digits: int = 2,
#     plot_scale: float = 0.8,
# ) -> Tuple[dict, TimeSeries, TimeSeries, plt.figure]:
#     # Get prediction.
#     pred_series = get_prediction(
#         model, pred_horizon, train_series, predict_from_train, num_samples
#     )
#     # Inverse-transform TS if transformer was passed.
#     if transformer is not None:
#         train_series, test_series, pred_series = (
#             transformer.inverse_transform(ser)
#             for ser in (train_series, test_series, pred_series)
#         )
#     # Calculate metrics for back test and for test/validation.
#     if metrics is None:
#         metrics = dict(MSE=mse, MAPE=mape)
#     metrics_test = {}
#     for name, metric in metrics.items():
#         metrics_test[name] = round(metric(pred_series, test_series), n_digits)
#     # Plot results.
#     plt.figure(figsize=np.array((8, 5)) * plot_scale)
#     train_series.plot(label="train data")
#     test_series.plot(label="test data")
#     pred_series.plot(label="pred data")
#     plt.legend()
#     title = ""
#     for name, metric in metrics_test.items():
#         title += f" {name} = {metric} |"
#     plt.title(title)
#     plt.ylabel(train_series.columns[0])
#     plt.xlabel(f"Time [{train_series.freq_str}]")
#     plt.legend()
#     return metrics_test, pred_series, test_series, fig


def evaluate_model(
        model: Any,
        series: TimeSeries,
        train_series: TimeSeries,
        test_series: TimeSeries,
        transformer: Pipeline = None,
        start: (float, dt.datetime) = 0.5,
        back_test_horizon: int = 4,
        metrics: dict[str: darts.metrics] = None,
        n_digits: int = 2,
        plot_scale: float = 0.8,
        **kwargs
) -> Tuple[dict, dict, pd.DataFrame, plt.figure]:
    # Get back test.
    backtest_series = model.historical_forecasts(
        series,
        start=start,
        forecast_horizon=back_test_horizon,
        verbose=False,
        last_points_only=True,
        **kwargs
    )
    # Inverse-transform TS if transformer was passed.
    if transformer is not None:
        series, train_series, test_series, backtest_series = (
            transformer.inverse_transform(ser) for ser in
            (series, train_series, test_series, backtest_series)
        )
    # Create DF with back test values.
    df = pd.concat([
        series.pd_series(),
        backtest_series.pd_series(),
        pd.Series(
            np.zeros(len(series)),
            index=series.time_index).astype(bool)],
        names=["data",  "back_test", "is_train"],
        axis=1)
    df.loc[df.index <= train_series.end_time(), "is_train"] = True
    # Calculate metrics for back test and for test/validation.
    if metrics is None:
        metrics = dict(MSE=mse, MAPE=mape)
    metrics_backtest, metrics_test = {}, {}
    for name, metric in metrics.items():
        metrics_backtest[name] = round(metric(series, backtest_series), n_digits)
        metrics_test[name] = round(metric(series, test_series), n_digits)
    # Plot results.
    fig = plt.figure(figsize=np.array((8, 5)) * plot_scale)
    train_series.plot(label="train data")
    test_series.plot(label="test data")
    backtest_series.plot(label="back test")
    plt.legend()
    title = f"horizon {back_test_horizon} time steps"
    for name, metric in metrics_backtest.items():
        title += f" | {name} = {metric}"
    plt.title(title)
    plt.ylabel(series.columns[0])
    plt.xlabel(f"Time [{series.freq_str}]")
    plt.legend()
    return metrics_test, metrics_backtest, df, fig
