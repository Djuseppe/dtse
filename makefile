.DEFAULT: build

check:
	poetry check

install: check
	poetry install --no-root

lock:
	poetry lock

dvc:
	poetry run dvc pull

clean:
	rm -f -r ./build/
	rm -f -r ./dist/
	rm -f -r *.egg-info
	rm -f .coverage

build: clean
	poetry build