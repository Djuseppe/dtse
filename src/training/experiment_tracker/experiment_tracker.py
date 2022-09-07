import os
from typing import Iterable, Optional

import boto3
import mlflow
import requests
from dotenv import load_dotenv
from requests.exceptions import ConnectionError

# Filter keys from params dict to exclude secrets from being stored by mlflow.
TEXT = "Service is not available at {url}."


def ping(url: str) -> bool:
    try:
        requests.get(url)
        return True
    except ConnectionError:
        return False


class S3Wrapper:
    """
    Wraps boto3 client for simplicity of methods.
    """

    def __init__(self, s3_uri: str) -> None:
        self.s3_uri = s3_uri
        if not ping(self.s3_uri):
            raise ConnectionError(TEXT.format(url=self.s3_uri))
        self.client = boto3.resource("s3", endpoint_url=self.s3_uri)

    def check_bucket_existence(self, bucket: str) -> bool:
        return self.client.Bucket(bucket) in self.client.buckets.all()

    def create_bucket(self, name: str) -> None:
        self.client.create_bucket(Bucket=name)


class MlFlowLogger:
    """
    MLFlow client wrapper, pings url of services (MLFlow and S3 storage at init).
    Logs experiment artifacts, metrics and parameters.
    """

    def __init__(self, dotenv_path: Optional[str] = None) -> None:
        # Load environmental variables.
        load_dotenv(dotenv_path, verbose=True)
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not ping(self.mlflow_tracking_uri):
            raise ConnectionError(TEXT.format(url=self.mlflow_tracking_uri))
        minio = S3Wrapper(os.getenv("MLFLOW_S3_ENDPOINT_URL"))
        bucket = os.getenv("AWS_S3_BUCKET")
        if not minio.check_bucket_existence(bucket):
            minio.create_bucket(bucket)

    @staticmethod
    def log(
        params: Optional[dict], metrics: Optional[dict], artifacts: Optional[Iterable]
    ) -> None:
        with mlflow.start_run() as _:
            mlflow.get_artifact_uri()
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            if artifacts:
                if isinstance(artifacts, (list, tuple)):
                    for artifact in artifacts:
                        mlflow.log_artifact(artifact)
                else:
                    mlflow.log_artifact(artifacts)
