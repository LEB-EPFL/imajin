# Imajin

Tools to simulate optics and microscopy experiments.

## Development

```console
conda env create -n leb.imajin -f environment/conda.yaml
conda-lock install --name leb.imajin environment/conda-lock.yaml
conda activate leb.imajin
poetry install
```

### Lock dependencies

```console
conda-lock -f environment/conda.yaml --lockfile environment/conda-lock.yaml
poetry lock
```
