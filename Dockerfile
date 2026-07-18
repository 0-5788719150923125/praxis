# Base image can be overridden via build arg for different architectures
# Default: NVIDIA CUDA for x86_64 GPU
# ARM64 CPU: ubuntu:24.04
ARG BASE_IMAGE=nvidia/cuda:13.2.0-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# Silence the NVIDIA entrypoint banner (no-op on non-CUDA base images)
RUN rm -f /opt/nvidia/entrypoint.d/*banner* /opt/nvidia/entrypoint.d/*.txt

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

# Node.js + wrangler CLI for the Cloudflare Pages integration (--publish-snapshot).
# The container runs as a non-root user, so wrangler cannot be installed at
# runtime (apt / npm -g need root) - bake it into the image here, as root, at
# build time. Pinned to wrangler@3: Ubuntu 24.04's apt Node.js is v18, and
# wrangler v4+ requires Node >=22, so latest wrangler fails at runtime with a
# Node-version error. v3 needs only Node >=16.17 and its `pages deploy` is all
# we use. (Bump to a newer Node via NodeSource if you ever need wrangler v4.)
RUN apt-get update -qq && \
    apt-get install -y -qq nodejs npm && \
    npm install -g wrangler@3 && \
    npm cache clean --force && \
    rm -rf /var/lib/apt/lists/*

# Configure git to trust the workspace directory
RUN git config --global --add safe.directory /workspace/praxis

# Set working directory
WORKDIR /workspace/praxis
