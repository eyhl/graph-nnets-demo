FROM python:3.11-slim


RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential git curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements-docker.txt ./requirements-docker.txt
COPY ./setup.py ./setup.py
COPY ./src ./src
COPY ./entrypoint.sh ./entrypoint.sh

RUN pip install --no-cache-dir -U pip setuptools wheel \
  && pip install --no-cache-dir -r requirements-docker.txt \
  && pip install --no-cache-dir -e .

# Torch / PyG (CPU) pinned for reproducibility.
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
  && pip install --no-cache-dir scipy \
  && pip install --no-cache-dir \
    torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cpu.html \
    torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html \
    torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cpu.html \
    torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html \
    torch-geometric -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

ENV PYTHONPATH="/app"

ENTRYPOINT ["sh", "entrypoint.sh"]
