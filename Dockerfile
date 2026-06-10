# Base image can be overridden via build arg for different architectures
# Default: NVIDIA CUDA for x86_64 GPU
# ARM64 CPU: ubuntu:24.04
ARG BASE_IMAGE=nvidia/cuda:13.2.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# Install Python and dependencies once
RUN apt-get update -qq && \
    apt-get install -y -qq python3-full python3-pip python3-venv git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# TeX toolchain for the living research paper: the PaperBuildCallback recompiles
# research/main.pdf on the --save-every cadence via latexmk. Without these the
# callback finds no latexmk and silently disables the rebuild. Targeted texlive
# set (not texlive-full, ~5GB) covering main.tex's usage: pdflatex+bibtex
# (base), amsthm/amsmath (recommended), pgfplots (extra), tikz/pgf (pictures),
# and recommended fonts.
RUN apt-get update -qq && \
    apt-get install -y -qq \
        latexmk \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-pictures \
        texlive-fonts-recommended && \
    rm -rf /var/lib/apt/lists/*

# Configure git to trust the workspace directory
RUN git config --global --add safe.directory /workspace/praxis

# Set working directory
WORKDIR /workspace/praxis
