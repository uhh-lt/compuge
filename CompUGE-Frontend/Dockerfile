FROM python:3.9-bookworm


# Set the working directory in the container
WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Gradio Frontend port
EXPOSE 7860

# Run the Gradio Frontend
ENTRYPOINT sh -c "python app.py"

