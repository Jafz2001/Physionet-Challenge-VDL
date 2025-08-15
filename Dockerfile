FROM python:3.8.6-slim

# Requisitos del sistema (soundfile suele necesitar libsndfile1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Estructura del challenge
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

# Actualiza pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Instala TODO desde requirements (incluido torch)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
