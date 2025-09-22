# Custom CUDA image with Python pre-installed
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

# Install Python and dependencies once
RUN apt-get update -qq && \
    apt-get install -y -qq python3-full python3-pip python3-venv git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace