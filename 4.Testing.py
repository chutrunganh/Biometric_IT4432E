import cv2
import os
import subprocess
import sys
import argparse

def list_cameras():
    try:
        output = subprocess.check_output(['v4l2-ctl', '--list-devices']).decode()
        print("Available cameras:")
        print(output)
    except subprocess.CalledProcessError:
        print("Error listing devices")

    video_devices = [f"/dev/video{i}" for i in range(10)]
    for device in video_devices:
        if os.path.exists(device):
            print(f"Found device: {device}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Camera viewer with configurable device index')
    parser.add_argument('--device', type=int, help='Camera device index (e.g., 4 for /dev/video4)')
    args = parser.parse_args()

    # List available cameras first
    list_cameras()

    # Use provided device index or default to 0
    camera_index = args.device if args.device is not None else 0
    print(f"Attempting to open camera at index {camera_index}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open camera at index {camera_index}")
        exit(1)

    print(f"Successfully opened camera at index {camera_index}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# v4l2-ctl --list-devices / ls /dev/video*
#  xhost +local:*
# sudo pacman -S xorg-xhost
# sudo usermod -aG docker $USER
# docker run --rm -it --privileged   --device=/dev:/dev/   -v /tmp/.X11-unix:/tmp/.X11-unix:rw   -e DISPLAY=$DISPLAY   test --device 1
