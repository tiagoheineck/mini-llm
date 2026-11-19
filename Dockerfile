FROM python:3.11-slim

WORKDIR /app

# PyTorch versão CPU (bem menor que a com CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir pyyaml numpy

COPY model.py train.py generate.py evaluate.py server.py config.yaml ./
COPY static/ ./static/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "train.py"]
