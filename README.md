# Imajin

Tools to simulate optics and microscopy experiments.

## Development

### Setup the development environment

1. Install [pyenv](https://github.com/pyenv/pyenv): `curl https://pyenv.run | bash`
2. Install Python interpreters: `pyenv install 3.9.12 && pyenv install 3.10.6`
3. Install [poetry](https://python-poetry.org/docs/): `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`
4. Activate the virtual environment: `poetry shell`
5. Install the dependencies: `poetry install`

### Run the tests

```console
poetry run tox
```

### Lock dependencies

```console
poetry lock
```
