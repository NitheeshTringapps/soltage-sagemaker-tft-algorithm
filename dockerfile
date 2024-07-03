FROM python:3.8-slim

# Install necessary libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.0.1 \
    optuna==3.4 \
    pandas \
    numpy \
    pytorch-forecasting \
    boto3 \
    pytorch-lightning \
    tensorflow \
    tensorboard

# Set up the environment variable for SageMaker
ENV SAGEMAKER_PROGRAM train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_REGION ap-south-1

# Copy the training script to the container
COPY train.py /opt/ml/code/train.py

# Define the entry point
ENTRYPOINT ["python", "/opt/ml/code/train.py"]
