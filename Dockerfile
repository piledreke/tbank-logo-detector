# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY validate.py ./validate.py

# Port
EXPOSE 8000

# Weights expected via mounted volume or URL at runtime

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


