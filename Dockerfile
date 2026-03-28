FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p output downloads logos data .cache

CMD ["sh", "-c", "uvicorn web_app:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 360"]
