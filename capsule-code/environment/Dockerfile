# hash:sha256:4bdfc2026ec8d8736ba89dced6966373520532d66432c7231325eb044c2d9fb8
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.4ubuntu1 \
        libgit2-dev=0.26.0+dfsg.1-1.1ubuntu0.2 \
        libssl-dev=1.1.1* \
        pkg-config=0.29.1-0ubuntu2 \
        python3-opencv=3.2.0+dfsg-4ubuntu0.1 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        r-base==3.6.1 \
        r-car==3.0_2 \
        r-hmisc==4.2_0 \
        r-psych==1.8.12 \
        r-readxl==1.3.1 \
        r-teachingdemos==2.10 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    autograd==1.1.7 \
    matplotlib==3.0.2 \
    opencv-python==4.0.0.21 \
    pillow==6.2.2 \
    scikit-image==0.15.0 \
    scikit-learn==0.20.3 \
    scipy==1.2.0 \
    seaborn==0.9.0 \
    setproctitle==1.1.10 \
    spectral==0.19 \
    tensorboardx==1.6 \
    torch==1.2.0 \
    torchvision==0.4.0 \
    tqdm==4.43.0 \
    xlrd==1.2.0

COPY postInstall /
RUN /postInstall
