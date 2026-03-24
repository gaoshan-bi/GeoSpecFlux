# GeoSpecFlux

Code for the paper **GeoSpecFlux: Geometry- and Spectral-Aligned Multimodal Learning for Robust Carbon Flux Prediction**.

This repository contains the training and evaluation code for multimodal carbon flux prediction from meteorological drivers and multispectral remote sensing observations.

## Scope

This repository is for code only.

- The dataset is not uploaded to GitHub.
- The paper PDF is not included in the GitHub code release.

## Dataset

Please download the dataset from Zenodo:

https://zenodo.org/records/11403428

After extracting the archive, place the data under the repository root in the following structure:

```text
data/
  <SITE_ID>/
    <DATE_RANGE>/
      predictors.csv
      targets.csv
      modis.pkl
      meta.json
```

The code expects the dataset root to be `./data`.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python GeoSpexFlux/run_local.py \
  --config GeoSpexFlux/config.yml \
  --data_dir ./data \
  --run_dir ./runs/geospecflux/default
```

## Evaluation

```bash
python GeoSpexFlux/test_ecoperceiver.py
```

Update the checkpoint paths inside the evaluation script if needed before running inference.

## Project Layout

- `GeoSpexFlux/model.py`: GeoSpecFlux model definition
- `GeoSpexFlux/dataset.py`: dataset loading and collation
- `GeoSpexFlux/run_local.py`: training entry point
- `GeoSpexFlux/test_ecoperceiver.py`: evaluation / inference script
- `GeoSpexFlux/config.yml`: experiment configuration

## License

The code is released under the license in [LICENSE](LICENSE).

## Acknowledgment

This repository was developed based on and adapted from the CarbonSense codebase:

https://github.com/mjfortier/CarbonSense
