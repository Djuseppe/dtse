import json
import os
from typing import Any, Dict


def load_config(file: str = os.path.join("config", "config.json")) -> Dict[str, Any]:
    # """
    # Function to load config from a destination on disk-
    # :param file: path to config file on disk
    # :return: config file as json
    # """
    # return json.loads(
    #     pkgutil.get_data(__name__, "../config/" + file).decode("utf-8"))
    with open(file, "r") as f:
        return json.load(f)
