##################################################################
# Dockerfile for pHfit
# Based on python 3.12-bookworm
##################################################################

FROM python:3.12-bookworm

# File Author / Maintainer
LABEL maintainer="Naoto Kubota <naotok@ucr.edu>"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get -qq update && \
	apt-get -qq -y install --no-install-recommends \
    wget less vim curl && \
    rm -rf /var/lib/apt/lists/*

# Install Arial font (msttcorefonts)
RUN apt-get -qq update && \
    apt-get -qq -y install --no-install-recommends \
    fontconfig cabextract && \
    wget -q http://ftp.de.debian.org/debian/pool/contrib/m/msttcorefonts/ttf-mscorefonts-installer_3.8_all.deb && \
    dpkg -i ttf-mscorefonts-installer_3.8_all.deb || true && \
    apt-get -qq -y --fix-broken install && \
    fc-cache -fv && \
    rm -f ttf-mscorefonts-installer_3.8_all.deb && \
    rm -rf /var/lib/apt/lists/*

# Install pHfit from build context
COPY . /tmp/pHfit
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /tmp/pHfit && \
    rm -rf /tmp/pHfit

# Set working directory
WORKDIR /home

# bash
CMD ["bash"]
