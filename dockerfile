# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app

# Ensure predict.py is executable
RUN chmod +x predict.py

# Run predict.py when the container launches
CMD ["python", "predict.py"]
