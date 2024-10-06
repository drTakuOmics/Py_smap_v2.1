# Use Ubuntu 22.04 as the base image
#FROM ubuntu:22.04
FROM nvidia/cuda:12.6.1-base-ubuntu22.04

# Install dependencies (e.g., wget, unzip)
RUN apt-get update && apt-get install -y wget unzip vim libxt6

# Download MATLAB MCR installer
#RUN wget -q https://ssd.mathworks.com/supportfiles/downloads/R2022b/Release/9/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2022b_glnxa64.zip
#RUN wget --progress=dot:giga https://ssd.mathworks.com/supportfiles/downloads/R2022b/Release/9/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2022b_glnxa64.zip

RUN wget -q https://ssd.mathworks.com/supportfiles/downloads/R2022b/Release/10/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2022b_Update_10_glnxa64.zip


# Unzip and install MCR silently
#RUN unzip MATLAB_Runtime_R2022b_Update_10_glnxa64.zip && \
#    ./install -mode silent -agreeToLicense yes -destinationFolder /opt/mcr
RUN unzip MATLAB_Runtime_R2022b_Update_10_glnxa64.zip -d /tmp/mcr_install && \
    cd /tmp/mcr_install && \
    ./install -mode silent -agreeToLicense yes -destinationFolder /opt/mcr


#RUN unzip -o -y MATLAB_Runtime_R2022b_Update_10_glnxa64.zip -d /tmp/mcr_install && \
#    cd /tmp/mcr_install && \
#    ./install -mode silent -agreeToLicense yes -destinationFolder /opt/mcr


# Set MCRROOT environment variable
ENV MCRROOT /opt/mcr/R2022b

#ENV LD_LIBRARY_PATH /opt/mcr/v912/runtime/glnxa64:/opt/mcr/v912/bin/glnxa64:/opt/mcr/v912/sys/os/glnxa64
ENV LD_LIBRARY_PATH ${MCRROOT}/runtime/glnxa64:${MCRROOT}/bin/glnxa64:${MCRROOT}/sys/os/glnxa64:${MCRROOT}/sys/opengl/lib/glnxa64

# Copy your compiled program to the image
#COPY /groups/denk/home/rickgauerj/smap_v2.1 /opt/smap/
COPY . /opt/smap/

# Set the working directory
WORKDIR /opt/smap

# Start a shell so you can run the script manually
ENTRYPOINT ["/bin/bash"]


