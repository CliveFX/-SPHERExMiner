ARG RAPIDS_IMAGE=rapidsai/base:26.06-cuda12-py3.12
FROM ${RAPIDS_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com
ENV LUXQUARRY_WORKDIR=/workspace/luxquarry_allsky_engine

WORKDIR /workspace

COPY pyproject.toml README.md /workspace/luxquarry_allsky_engine/
COPY src /workspace/luxquarry_allsky_engine/src

WORKDIR /workspace/luxquarry_allsky_engine

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -e .

RUN python -m compileall -q src/luxquarry_allsky_engine \
    && luxquarry-allsky --help >/tmp/luxquarry_cli_help.txt

ENTRYPOINT ["luxquarry-allsky"]
CMD ["env-probe"]
