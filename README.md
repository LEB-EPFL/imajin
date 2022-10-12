# Imajin

![build](https://github.com/LEB-EPFL/imajin/actions/workflows/build.yml/badge.svg)

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
4. Set the virtual environment Python version to 3.10: `poetry env use 3.10`
4. Activate the virtual environment: `poetry shell`
5. Install the dependencies: `poetry install`

### Testing

#### Run all tests and linters (except for benchmarks)

```console
poetry run tox
```

#### Run specific tests and linters

```console
# Black, isort, mypy, pylint
poetry run tox -e black
# etc. ...

# Python 3.9/3.10 tests
poetry run tox -e py39
poetry run tox -e py310
```

### Run the benchmarks

```console
poetry run tox -e benchmark
```

### Format the code

```console
poetry run tox -e format
```

### Lock dependencies

```console
poetry lock
```

### Profiling

Profiling will tell you how long the code spends in each function call. A few benchmarks are already setup for profiling.

```
poetry run tox -e profile
```

To view the results of, for example, test_benchmark_simulator_0_result.json:

```console
poetry run vizviewer test_benchmark_simulator_0_result.json
```

### Troubleshooting

#### Poetry is using the wrong Python version

Check the version of Python used by Poetry to create the virtual environment:

```console
poetry env info
```

Set the Python version to 3.10 (for example):

```console
poetry env use 3.10
```
