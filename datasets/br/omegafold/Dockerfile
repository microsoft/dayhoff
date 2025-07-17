ARG PYTORCH_TAG=2.3.1-cuda12.1-cudnn8-devel
FROM pytorch/pytorch:${PYTORCH_TAG}

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        git \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

RUN pip install biopython && pip install --no-deps git+https://github.com/alexj-lee/OmegaFold.git && \
    mkdir -p /root/.cache/omegafold_ckpt && wget https://helixon.s3.amazonaws.com/release1.pt -O /root/.cache/omegafold_ckpt/model.pt
