# The DevFest Helper service (server.py). Build context: the repo root.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    PIP_NO_CACHE_DIR=1

# CPU-only PyTorch first, so laya doesn't pull the multi-GB CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Bake the Laya checkpoint (~800 MB) into the image, so a cold start doesn't download it.
RUN python -c "from laya import Router; Router().predict('warm-up', {'q': {'type': 'noul', 'instructions': 'Is this a test?'}})"

WORKDIR /app
COPY . /app/

CMD exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
