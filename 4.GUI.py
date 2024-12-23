# ============================================================
# Import dependencies
# ============================================================

# For functions related to Deep Learning model and image processing
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

# For Kivy App
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
# For UI components
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage # Prevent name conflict with PIL Image
from kivy.uix.textinput import TextInput
# For others kivy components
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.core.window import Window


# ============================================================
# Set up some constants, paths and variables
# ============================================================

# Create a application_data folder to store all app related data
os.makedirs('application_data', exist_ok=True)
os.makedirs(os.path.join('application_data', 'settings.json'), exist_ok=True)
SETTINGS_FILE_PATH = os.path.join('application_data', 'settings.json')
os.makedirs(os.path.join('application_data', 'validation_images'), exist_ok=True)
VALIDATION_PATH = os.path.join('application_data', 'validation_images')


# Some constants
CAM_ID = 0 # Defined the Camera ID to use
DETECTION_THRESHOLD = 0.5 # Metric above which the prediction is considered as positive
VERIFICATION_THRESHOLD = 0.6 # Proportion of positive detections/ total positive samples
LIMIT_IMAGES_TO_COMPARE = 4 # Limit the number of images to compare in the verification process



