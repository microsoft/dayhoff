FROM nvcr.io/nvidia/pytorch:23.09-py3


RUN pip install --upgrade pip && \
    pip install "jax[cuda]" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install git+https://github.com/sokrypton/ColabDesign

