FROM python:3.8-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN mkdir /app
COPY . /app
RUN cd ./app
