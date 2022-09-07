# DTSE
DTSE interview task.

### Requirements

To be able to run the in this repo [make utility](https://www.gnu.org/software/make/)
and [docker](https://www.docker.com/) are required.

To set up environment execute the following:
```bash
make install
```

### The whole task consists of different parts:**
1. EDA & modeling is done in [Jupyter ntb](./notebooks/eda_modelling.ipynb).
There's also an HTML version this notebook, however it's not a part of the repo,
but can be easily obtained by calling (ntb appears in the `./output`):
```bash
make notebook
```
2. Neural network (N-BEATS) was trained separately
in the [notebook](./notebooks/nbeats.ipynb) on Colab.

3. There is an automated pipeline for training RNN model.
In order to run an experiment one needs to follow the next steps:

- make sure that docker daemon is running
- one needs to put required environmental variables
to the `.env` file (example is provided in `.env.example`)
But just for simplicity I left those variable in the `.env` file,
though it's not a good technical solution.
- start dockerized MLFLow service by execution of:
```bash
make start-mlflow
```
MlFlow is running with remote dockerized tracking server,
backend (PostgresSQL) and artifact stores (Minio).
Thus, it is ready to be deployed.
More details can be found [here](docker-compose.yaml).
- let the docker up all required containers (it takes some time)
- make sure that MLFlow is running at `http://127.0.0.1:5000`
(default host & port)
- then user has to provide RNN hyperparameters in the
`./src/training/config/config.json`
- then run an experiment by ([source code](./src/training/pipeline.py)):
```bash
make train
```
- the experiment inputs (data and parameters)
and outputs (model, metric graphs) show be tracked with MLFlow,
so one can see results in the MLFlow service at `http://127.0.0.1:5000`
- to stop MLFlow run the cmd:
```bash
make stop-mlflow
```
