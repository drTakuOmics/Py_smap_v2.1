# Use CUDA/Ubuntu 22.04 as the base image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Install system dependencies first (e.g., wget, unzip, bash, file, and other required libraries)
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    vim \
    libxt6 \
    bash \
    file \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download MATLAB MCR installer for runtime files
#RUN wget -q https://ssd.mathworks.com/supportfiles/downloads/R2022b/Release/10/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2022b_Update_10_glnxa64.zip
RUN wget -q https://ssd.mathworks.com/supportfiles/downloads/R2023b/Release/0/deployment\_files/installer/complete/glnxa64/MATLAB\_Runtime\_R2023b\_glnxa64.zip

# Unzip and install MCR silently
#RUN unzip MATLAB_Runtime_R2022b_Update_10_glnxa64.zip -d /tmp/mcr_install && \
RUN unzip MATLAB_Runtime_R2023b_glnxa64.zip -d /tmp/mcr_install && \
    cd /tmp/mcr_install && \
    ./install -mode silent -agreeToLicense yes -destinationFolder /opt/mcr && \
    rm -rf /tmp/mcr_install

# Set MCRROOT and LD_LIBRARY_PATH environment variables after installing system packages
#ENV MCRROOT /opt/mcr/R2022b
ENV MCRROOT /opt/mcr/R2023b
ENV LD_LIBRARY_PATH ${MCRROOT}/runtime/glnxa64:${MCRROOT}/bin/glnxa64:${MCRROOT}/sys/os/glnxa64:${MCRROOT}/sys/opengl/lib/glnxa64

# Copy program files including executables to the image
COPY . /opt/smap/

# Set the working directory
WORKDIR /opt/smap

# Start a shell so you can run the script manually
ENTRYPOINT ["/bin/bash"]

