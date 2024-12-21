#-----------------------------------------------------------------------
#  Demo for face verification app
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# 1. Dependencies and CAM ID setup
# We have completed training some model and store those models to `model_saved` folder. Now we load those models and use them to verify the face of a person.

# The process will be as follow:
# 1. User register their face to the system through sacnning process. After we get sanning images, extract face then extract face embeddings and store them to database.
# 2. When user want to verify their face, we open the camera, capture the image, extract face embeddings and compare with the embeddings in database (with correspoding name use provide when login).
#-----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
import uuid
from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import savez_compressed
import pickle
import json
import os

# For the Facenet model
import torch  # Ensure torch is imported here to avoid circular import issues
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

''' 
Each device has a different camera IDs/index, so we need to find the correct camera ID for our device. We try to loop throught a range of 
camera IDs and ask user to check if the camera is working. Each devices can have many webcams, so we have many corresponding camera IDs, we ask user to 
choose their prefered camera ID as well.

Store these congifuration to the `application_data/setting.json` file. **Next time, when user open the app, these setting will be loaded 
without asking user again.**
'''

# Create a application_data folder to store all app related data
os.makedirs('application_data', exist_ok=True)

# Setting.jpg file path
SETTINGS_FILE_PATH = os.path.join('application_data', 'settings.json')

def detect_cameras():
   
    print("Scanning for available cameras")
    detected_cameras = []

    # Test cameras 0-9
    for cam_id in range(10):
        print(f"\nTesting camera {cam_id}")
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f'Camera {cam_id}', frame)
                cv2.waitKey(1000)
                
                response = input(f"Can you see camera ID {cam_id}? (Y/n): ").lower()
                if response == 'y' or response == 'Y':
                    detected_cameras.append(cam_id)
                else:
                    continue
                
                cv2.destroyAllWindows()
            cap.release()

    if not detected_cameras:
        print("No cameras detected!")
        return [], None

    # Select preferred camera
    preferred = None
    if len(detected_cameras) > 1:
        while preferred not in detected_cameras:
            try:
                print("\nAll available cameras:", detected_cameras)
                preferred = int(input("Enter the camera ID you want to use (ID 0 is often RGB camera, ID 2 is often IR camera): "))
            except ValueError:
                print("Please enter a valid number")
    else:
        preferred = detected_cameras[0]

    # Save settings
    settings = {
        "camera_list": detected_cameras,
        "preferred_camera": preferred
    }
    
    with open(SETTINGS_FILE_PATH, 'w') as f:
        json.dump(settings, f)
        print(f"\nSettings saved to {SETTINGS_FILE_PATH}")

    return detected_cameras, preferred


CAM_ID = 0 # Default camera ID

# Check if settings file exists, if no then call the detect camera function
if not os.path.exists(SETTINGS_FILE_PATH):
    cameras, preferred = detect_cameras()
    print(f"\nDetected cameras: {cameras}")
    print(f"Preferred camera: {preferred}")
else:
    print(f"Settings file found at {SETTINGS_FILE_PATH}, if you want to rescan for cameras delete this file.")

#Load the settings file
with open(SETTINGS_FILE_PATH, 'r') as f:
    settings = json.load(f)
    CAM_ID = settings['preferred_camera']
    
print("Running app with camera ID:", CAM_ID)