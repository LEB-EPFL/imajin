# Imajin

Tools to simulate optics and microscopy experiments.

## Development

### Setup the development environment

1. Install [pyenv](https://github.com/pyenv/pyenv): `curl https://pyenv.run | bash`
2. Install Python interpreters: `pyenv install 3.9.12 && pyenv install 3.10.6`
3. Install [poetry](https://python-poetry.org/docs/): `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`
4. Activate the virtual environment: `poetry shell`
5. Install the dependencies: `poetry install`

### Run the tests and linters

```console
poetry run tox
```

### Format the code

```console
poetry run tox -e format
```

### Lock dependencies

```console
poetry lock
```

### Design Guidelines

- Functions/methods that expect sequence-like types (e.g. lists, tuples, etc.) as inputs should accept NumPy [ArrayLike](https://numpy.org/devdocs/reference/typing.html#numpy.typing.ArrayLike) types because these can be either NumPy arrays or native Python sequences.
