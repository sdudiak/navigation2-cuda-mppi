FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV RCUTILS_COLORIZED_OUTPUT=1
ARG DEBIAN_FRONTEND=noninteractive

# Basic software
RUN apt update && \
    apt install -y \
    build-essential \
    apt-utils \
    software-properties-common \ 
    unzip \
    curl \
    cmake \
    git 

# Requirements for the Nsight Systems
RUN apt update -y && \
    apt-get install -y --fix-missing --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    dbus \
    fontconfig \
    gnupg \
    libasound2 \
    libfreetype6 \
    libglib2.0-0 \
    libopengl0 \
    libnss3 \
    libsqlite3-0 \
    libx11-xcb1 \
    libxcb-glx0 \
    libxcb-xkb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxml2 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libxkbfile1 \
    openssh-client \
    wget \
    xcb \
    xkb-data && \
    apt-get clean

# Nsight Systems
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    qtcreator \
    qtbase5-dev \
    qt5-qmake \
    cuda-nsight-systems-12-1 && \
    apt-get clean

# ROS2 Humble installation
RUN apt install software-properties-common && \
    add-apt-repository universe && \
    apt update && apt install curl -y && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt update -y && \
    apt install -y ros-humble-desktop && \
    apt install -y ros-dev-tools

WORKDIR /home/user/ros_ws/src
CMD [ "sleep", "infinity" ]