FROM python:3.10-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install -y \
    qtbase5-dev \
    qt5-qmake \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && apt-get clean

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
