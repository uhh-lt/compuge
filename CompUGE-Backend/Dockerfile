# Dockerfile for Fastapi services in main.py

FROM python:3.9-bookworm

# Set the working directory in the container
WORKDIR /app
COPY . /app

# Install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip list

# Expose the port the app runs on
ENTRYPOINT sh -c "sleep 5 && uvicorn main:app --host=0.0.0.0 --port=8000 --reload"
