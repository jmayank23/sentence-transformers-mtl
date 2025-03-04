# Use an official Python runtime as a parent image
FROM python:3.9.7-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container at /app
COPY . /app

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Run task 1 to pre-download the model weights.
RUN python -m src.task1.sentence_transformer

# Set environment variable to prevent needing to download the model again
ENV HF_HOME=/app/model_weights

# Command to run when the container starts (you can override this)
CMD ["bash"]