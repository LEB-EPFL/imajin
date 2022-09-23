# Imajin

Tools to simulate optics and microscopy experiments.

## Getting started

See the [integration tests](tests/integration/) for examples.

## Installation

Choose one of the following methods. It is recommended to install into a virtual environment.

```console
# With SSH:
pip install git+ssh://git@github.com/LEB-EPFL/imajin.git

# With a GitHub access token:
pip install git+https://"$GITHUB_TOKEN"@github.com/LEB-EPFL/imajin.git
```

See https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token for how to generate an access token. The token requires `repo` privileges.

## Development

### Setup the development environment

1. Install [pyenv](https://github.com/pyenv/pyenv): `curl https://pyenv.run | bash`
2. Install Python interpreters: `pyenv install 3.9.13 && pyenv install 3.10.6`
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
