FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Set environment to non-interactive for clean installs
ENV DEBIAN_FRONTEND=noninteractive

# Install git and other system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone and install causal-conv1d
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git && \
    cd causal-conv1d && \
    git checkout v1.4.0 && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install . && \
    cd ..

# Clone and install mamba
RUN git clone https://github.com/state-spaces/mamba.git && \
    cd mamba && \
    git checkout v2.2.4 && \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE \
    CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE \
    pip install --no-build-isolation . && \
    cd ..

# Install dayhoff package from PyPI
RUN pip install dayhoff

# Add GitHub to known_hosts to avoid host verification error
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# Clone the private or public Dayhoff repo 
RUN --mount=type=ssh git clone git@github.com:microsoft/dayhoff.git /dayhoff

# Set working directory to inside the cloned repo
WORKDIR /dayhoff
