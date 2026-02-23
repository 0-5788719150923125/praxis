# Base image can be overridden via build arg for different architectures
# Default: NVIDIA CUDA for x86_64 GPU
# ARM64 CPU: ubuntu:24.04
ARG BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# Install Python and dependencies once
RUN apt-get update -qq && \
    apt-get install -y -qq python3-full python3-pip python3-venv git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Configure git to trust the workspace directory
RUN git config --global --add safe.directory /workspace

# Set working directory
WORKDIR /workspace