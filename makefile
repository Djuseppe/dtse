.DEFAULT: train

MLFLOW_HOME					:= ./mlflow_service
MLFLOW_DOCKER_PATH			:= ./$(MLFLOW_HOME)/mlflow_image
MLFLOW_TAG					:= mlflow_server
MLFLOW_RESULTS_FOLDER		:= ./$(MLFLOW_HOME)/mlruns
NTB_NAME					:= eda_modelling


build-mlflow-server:
	docker build -f $(MLFLOW_DOCKER_PATH)/Dockerfile -t $(MLFLOW_TAG) .

start-mlflow: build-mlflow-server
	docker-compose up -d --build

stop-mlflow:
	docker-compose down

check:
	poetry check

install: check
	poetry install --no-root

lock:
	poetry lock

dvc:
	poetry run dvc pull

clean:
	rm -rf ./build/
	rm -rf ./dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf darts_logs
	rm -rf notebooks/darts_logs
	rm -rf src/training/darts_logs

lint:
	poetry run yapf -d -r --style google -vv -e .venv .

type-check:
	poetry run mypy src --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs

datatest: dvc
	poetry run python -m unittest discover -v -s ./tests  -p data*.py

train: clean dvc datatest
	poetry run python src/training/pipeline.py data models --track true; \
	rm -rf darts_logs

test:
	poetry run coverage run --source src -m unittest discover -v -s ./tests  -p systest*.py
	poetry run coverage report -m --fail-under 0
	poetry run coverage html -d build/test-coverage
	poetry run coverage json -o build/test-coverage.json --pretty-print
	poetry run coverage erase

notebook:
	mkdir -p output; \
	rm -rf ./output/$(NTB_NAME).html; \
	poetry run jupyter nbconvert ./notebooks/$(NTB_NAME).ipynb --TagRemovePreprocessor.enabled=True \
       --TagRemovePreprocessor.remove_cell_tags="['verbose', 'hidden_cell', 'hide_cell']" \
       --to html --TemplateExporter.exclude_input=True --no-prompt; \
    cp ./notebooks/$(NTB_NAME).html ./output/$(NTB_NAME).html

build: clean lint
	poetry build -f wheel -n
