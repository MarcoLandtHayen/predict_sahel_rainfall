predict_sahel_rainfall
==============================
[![Build Status](https://github.com/MarcoLandtHayen/predict_sahel_rainfall/workflows/Tests/badge.svg)](https://github.com/MarcoLandtHayen/predict_sahel_rainfall/actions)
[![codecov](https://codecov.io/gh/MarcoLandtHayen/predict_sahel_rainfall/branch/main/graph/badge.svg)](https://codecov.io/gh/MarcoLandtHayen/predict_sahel_rainfall)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)
[![Docker Image Version (latest by date)](https://img.shields.io/docker/v/mlandthayen/predict_sahel_rainfall?label=DockerHub)](https://hub.docker.com/r/mlandthayen/predict_sahel_rainfall/tags)

## Outline

In this project, we aim to predict rainfall in the African Sahel region with various machine learning and deep learning methods. As data, we use a CICMoD - a climate index collection based on model data from two state-of-the-art earth system models. For details, see: https://github.com/MarcoLandtHayen/climate_index_collection

## Development

For now, we're developing in a Docker container with JupyterLab environment, Tensorflow and several extensions, based on martinclaus/py-da-stack.

To start a JupyterLab within this container, run
```shell
$ docker pull mlandthayen/py-da-tf:shap
$ docker run -p 8888:8888 --rm -it -v $PWD:/work -w /work mlandthayen/py-da-tf:shap jupyter lab --ip=0.0.0.0
```
and open the URL starting on `http://127.0.0.1...`.

Then, open a Terminal within JupyterLab and run
```shell
$ python -m pip install -e .
```
to have a local editable installation of the package.

## Container Image

Additionally, there's a container image having **predict_sahel_rainfall** as pre-installed Python package: https://hub.docker.com/r/mlandthayen/predict_sahel_rainfall.

### Use with Docker

You can use it wherever Docker is installed by running:
```shell
$ docker pull mlandthayen/reconstruct_missing_data:<tag>
$ docker run -p 8888:8888 --rm -it -v $PWD:/work -w /work mlandthayen/reconstruct_missing_data:<tag> jupyter lab --ip=0.0.0.0
```

and open the URL starting on `http://127.0.0.1...`.
Here, `<tag>` can either be `latest` or a more specific tag.

### Use with Singularity

You can use it wherever Singularity is installed by essentially running:
```shell
$ singularity pull --disable-cache <target.sif> docker://mlandthayen/climate_index_collection:<tag>
$ singularity run --bind $WORK <target.sif> jupyter lab --no-browser --ip $(hostname) $WORK
```
Here, `<tag>` can either be `latest` or a more specific tag.
And `<target.sif>`specifies the target file to store the container image.

--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
