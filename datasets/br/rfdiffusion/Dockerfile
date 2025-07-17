FROM nvcr.io/nvidia/pytorch:23.09-py3

RUN apt-get update && \
        apt-get install -y \
        wget \
        git 
RUN git clone https://github.com/sokrypton/RFdiffusion.git
RUN pip install jedi \
        omegaconf \
        hydra-core \
        icecream \
        pyrsistent
RUN pip install --no-dependencies dgl==2.0.0 -f https://data.dgl.ai/wheels/cu121/repo.html 
RUN pip install --no-dependencies e3nn==0.3.3 
RUN pip install opt_einsum_fx
RUN cd RFdiffusion/env/SE3Transformer && \
        pip install .
RUN wget -qnc https://files.ipd.uw.edu/krypton/ananas && chmod +x ananas
RUN mv RFdiffusion/* /root
RUN pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1
RUN ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign
ENV DGLBACKEND="pytorch"
