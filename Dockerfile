FROM python:3.10-slim-bullseye

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
RUN pip3 install --no-cache-dir -r requirements.txt
