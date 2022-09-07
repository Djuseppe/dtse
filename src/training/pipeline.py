import logging
import os
import time
import warnings

import click
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel
from darts.utils.likelihood_models import GaussianLikelihood
from experiment_tracker import MlFlowLogger
from sklearn.preprocessing import StandardScaler
from util import (
    FileUtil,
    clean_data,
    evaluate_model,
    partition,
    preprocess,
    transform_to_ts,
)
from util.config import load_config

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config", "config.json")
warnings.filterwarnings("ignore")

logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__file__)


@click.command()
@click.argument(
    "input_path",
    type=click.Path(exists=True),
)
@click.argument("output_path", type=click.Path())
@click.option("--track", type=bool)
def run_pipeline(input_path: str, output_path: str, track: bool = True) -> None:
    logger.info("Pipeline started.")
    # Load configuration.
    config = load_config(CONFIG_FILE)
    # Setup folder structure
    file_util = FileUtil(input_path, output_path)
    # file_util.create_training_folders()

    # 1. Prepare data
    # Read raw data.
    raw_data = file_util.get_raw_data(config["raw_data"])
    # Clean and transform data to darts.TimeSeries.
    data = clean_data(raw_data)
    # Save cleaned data.
    file_util.save_data(data, os.path.join(input_path, "interim", "data_cleaned.csv"))
    series = transform_to_ts(data.unemp)
    # Train-test split.
    train, test = partition(series, config["train_fraction"])

    # 2. Transform data.
    transformer = Pipeline(
        [
            Scaler(StandardScaler(), "StandardScaler"),
        ]
    )
    series_transformed, train_transformed, test_transformed = preprocess(
        series, transformer, train, test
    )
    # Save transformer.
    file_util.save_transformer(
        transformer, os.path.join(output_path, f"{config['name']}_transformer.pickle")
    )

    # 3. Init and fit model.
    start_time = time.time()
    rnn = RNNModel(
        **config["training_parameters"],
        log_tensorboard=True,
        likelihood=GaussianLikelihood(),
    )
    rnn.fit(train_transformed, verbose=False)

    # 4. Evaluate model.
    scores_test, _, _, fig = evaluate_model(
        rnn, series_transformed, train_transformed, test_transformed, retrain=False
    )

    # 5. Save assets.
    # Save model.
    model_file = f"{config['name']}.pt"
    file_util.save_model(rnn, model_file)
    # Evaluate time elapsed.
    stop_time = time.time()
    elapsed = (stop_time - start_time) / 60
    logger.info(f"Elapsed {elapsed:.2f} min")

    # 6. Track experiment by tracker.
    if track:
        tracker = MlFlowLogger()
        training_params = config.pop("training_parameters")
        config.update(
            **training_params,
            transformer=transformer.__repr__(),
            elapsed=f"{elapsed:.2f} min",
        )
        tracker.log(
            metrics=scores_test,
            artifacts=(
                os.path.join(input_path, "raw", config["raw_data"]),
                figure_file,
                os.path.join(output_path, model_file),
            ),
            params=config,
        )
    logger.info("Pipeline has stopped.")


if __name__ == "__main__":
    run_pipeline()
