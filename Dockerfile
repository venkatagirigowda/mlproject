FROM python:3.8-slim
WORKDIR /app
COPY . /app
EXPOSE 8000
RUN apt-get update -y && apt-get install -y \
    build-essential \
    python3-dev \
    awscli \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]