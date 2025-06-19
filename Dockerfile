# Dockerfile
FROM continuumio/miniconda3

# Set working directory inside container
WORKDIR /workspace
COPY . /workspace

# Create conda environment with Python 3.9
RUN conda create -y -n pcl_env python=3.9

# Use conda environment for all subsequent commands
SHELL ["conda", "run", "-n", "pcl_env", "/bin/bash", "-c"]

# Install pip and dependencies from req_new.txt
COPY req_new.txt .

RUN pip install --no-cache-dir --timeout=60 -i https://pypi.tuna.tsinghua.edu.cn/simple -r req_new.txt

# Set PYTHONPATH to allow importing modules from /workspace
ENV PYTHONPATH=/workspace

# Run unit tests
RUN pytest tests/

# Default command: run training script
CMD ["conda", "run", "--no-capture-output", "-n", "pcl_env", "python", "train1.py"]
