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
    tensorboard \
    flask \
    gevent \
    gunicorn

# Set up the environment variable for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_REGION ap-south-1

# Copy the training and serving scripts to the container
COPY train.py /opt/ml/code/train.py
COPY serve.py /opt/ml/code/serve.py
COPY entrypoint.sh /opt/ml/code/entrypoint.sh

# Ensure the entrypoint script is executable
RUN chmod +x /opt/ml/code/entrypoint.sh

# Set up the entry point
ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]

# Expose the port on which the Flask app will run (default is 8080)
EXPOSE 8080