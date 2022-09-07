import os
import pickle
from typing import Any

import pandas as pd


class FileUtil:
    """
    Class for file operations: saving, loading, etc.
    """

    def __init__(self, input_path: str = None, output_path: str = None) -> None:
        self.input_path = input_path
        self.output_path = output_path

    def get_raw_data(self, file: str) -> pd.DataFrame:
        """
        Function to read raw csv data from file and return a data frame.
        """
        return pd.read_csv(os.path.join(self.input_path, "raw", file))

    def get_clean_data(self, file: str = "data_cleaned.csv") -> pd.DataFrame:
        """
        Function to read clean csv data from file and return a data frame.
        """
        return pd.read_csv(os.path.join(self.input_path, "interim", file))

    @staticmethod
    def save_data(data: pd.DataFrame, file: str):
        """
        Function to save csv data to file.
        """
        data.to_csv(file)

    def save_model(self, model: Any, model_name: str):
        """
        Function to save darts model.
        """
        model.save(os.path.join(self.output_path, model_name))

    @staticmethod
    def save_figure(figure, file: str) -> None:
        """
        Function to save figure.
        """
        figure.savefig(file, format="png")

    @staticmethod
    def save_transformer(transformer: Any, file: str) -> None:
        """
        Function to save transformer to pickle file.
        """
        with open(file, "wb") as f:
            pickle.dump(transformer, f)

    @staticmethod
    def load_transformer(file: str) -> Any:
        """
        Function to read transformer from pickle file.
        """
        with open(file, "rb") as f:
            return pickle.load(f)
