# Use a base image with Python 
FROM python:3.12-slim


# Set the working directory in the container
WORKDIR /app


# Install OpenCV and X11 dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    v4l-utils \
    x11-apps \
    libx11-6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


# Copy the 4. CLI_version_for_docker.py file into the container
COPY 4.CLI_version_for_Docker.py /app/

# Copy the model_saved/Siamese_model directory into the container
COPY model_saved/fully_siamese_network.h5 /app/

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install opencv-python



# Add the default user to the video group
RUN usermod -aG video root

# Set environment for display
ENV QT_X11_NO_MITSHM=1

# # Make sure the entrypoint allows parameters to be passed
# ENTRYPOINT ["python3", "4.CLI_version_for_Docker.py"]

ENTRYPOINT ["python3", "4.CLI_version_for_Docker.py"]

# How to mkae docker file:

# 1. Ensure you have enable the display service, we will need this to open the camera windown of OpenCV
# xhost +local:*
# In case command not found, install with
# sudo pacman -S xorg-xhost

# 2. Build with the following command
# docker build -t IMAGE_NAME .

# 3. Run the image
# docker run --rm -it --privileged   --device /dev/video0:/dev/video0   -v /tmp/.X11-unix:/tmp/.X11-unix:rw   -e DISPLAY=$DISPLAY   IMAGE_NAME --device CAMERA_ID_TO_TRY 
# Some parameters:
# --device /dev/video0:/dev/video0 : map your physical cameras in host machine to the virtual one inside docker container. 
# you can map serveral cameras: --device /dev/video0:/dev/video0  --device /dev/video1:/dev/video1. To discover all the camera ID
# in your machine, use command: v4l2-ctl --list-devices or command: ls /dev/video*

#### CURRENTLY ISSUE ####

# (venv) [cta@CTA-ThinkPad Biometric_IT4432E]$ docker run --rm -it --privileged   --device /dev/video0:/dev/video0   -v /tmp/.X11-unix:/tmp/.X11-unix:rw   -e DISPLAY=$DISPLAY   test --device 0
# Cannot open device /dev/video0, exiting.
# Error listing devices
# Attempting to open camera at index 0
# [ WARN:0@0.015] global cap_v4l.cpp:999 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
# [ERROR:0@0.015] global obsensor_uvc_stream_channel.cpp:158 getStreamChannelGroup Camera index out of range
# Failed to open camera at index 0


# Can not solve why Docker still can not access to the host physical camera, any help is appreciated !!!!!!!