#!/bin/bash

# Determine if the container is running in training or serving mode
if [ "$1" == "train" ]; then
    shift
    python /opt/ml/code/train.py "$@"
else
    python /opt/ml/code/serve.py "$@"
fi
