# Dockerfile for Fastapi service in main.py

FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port the app runs on
ENTRYPOINT sh -c "sleep 5 && uvicorn main:app --host=0.0.0.0 --port=8000 --reload"
