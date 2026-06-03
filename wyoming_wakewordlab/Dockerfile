FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    wakewordlab \
    wyoming

COPY wyoming_wakewordlab/ ./wyoming_wakewordlab/

EXPOSE 10400

ENTRYPOINT ["python", "-m", "wyoming_wakewordlab"]
CMD ["--uri", "tcp://0.0.0.0:10400", "--models", "hey_jarvis"]
