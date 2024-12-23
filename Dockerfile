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

# Build with the following command
# docker build -t image_name .
# docker run --rm -it --privileged   --device /dev/video0:/dev/video0   -v /tmp/.X11-unix:/tmp/.X11-unix:rw   -e DISPLAY=$DISPLAY   image_name