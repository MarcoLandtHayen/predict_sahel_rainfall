FROM mlandthayen/py-da-tf:shap AS buildstage

WORKDIR /source
USER root

RUN apt-get update
RUN apt-get install -y git

COPY . .
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
RUN python -m pip wheel .

FROM mlandthayen/py-da-tf:shap AS app

COPY --from=buildstage /source/predict*.whl .

RUN python -m pip install predict*.whl
