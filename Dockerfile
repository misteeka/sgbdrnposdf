FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql postgresql-contrib\
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
    
COPY / /app

RUN pip3 install -r requirements.txt

EXPOSE 28003

HEALTHCHECK CMD curl --fail http://localhost:28003/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=28003", "--server.address=0.0.0.0"]