# For functions related to Deep Learning model and image processing
import cv2
import os
import numpy as np
from numpy import savez_compressed
from PIL import Image
from keras.layers import Layer
import tensorflow as tf

# For Kivy App
from kivy.app import App
# For UI components
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage # Prevent name conflict with PIL Image
# For others kivy components
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.core.window import Window




# Some constants
CAM_ID = 0 # Defined the Camera ID to use
VALIDATION_PATH = "application_data/validation_images" # Path to the validation images folder
DETECTION_THRESHOLD = 0.5 # Metric above which the prediction is considered as positive
VERIFICATION_THRESHOLD = 0.6 # Proportion of positive detections/ total positive samples
LIMIT_IMAGES_TO_COMPARE = 4 # Limit the number of images to compare

# Write code to auto detect the camera ID later