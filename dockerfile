FROM python:3.14-slim

COPY src /app/src
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /app

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
git \
curl \
&& rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /app/src

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "main.py"]