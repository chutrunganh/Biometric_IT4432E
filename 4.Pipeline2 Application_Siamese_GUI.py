"""---------------------------------------------------------Import all the necessary libraries---------------------------------------------------------"""
# pip install kivy[base] kivy[full]

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


"""---------------------------------------------------------Load functions for the model and preprocess the image---------------------------------------------------------"""
# These functions are just copy paste from Pipeline2 DataPreprocessing.ipynb and Pipeline2 Siamense_Network.ipynb with slightly modify, so don care about these functions, just focus on the kivy app
# code in the next part


#Since our Siamese model uses a custom L1 distance layer, this layer is not saved with the model itself. We rewrite the L1Dist layer here to load the model successfully.
class L1Dist(Layer):
    def __init__(self, **kwargs):
         super(L1Dist, self).__init__(**kwargs)
    
    def call(self,input_embedding, validation_embedding):
        
        # Convert inputs to tensors otherwise will meet error: unsupported operand type(s) for -: 'List' and 'List'
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        input_embedding = tf.squeeze(input_embedding, axis=0)  # Remove potential first dimension
        validation_embedding = tf.squeeze(validation_embedding, axis=0)

        # Calculate and return the L1 distance
        return tf.math.abs(input_embedding - validation_embedding)
    

# Define the preprocess function
def gaussian_blur(image, kernel_size=(3,3), sigma=0.1):
    """
    Apply Gaussian blur to an image using TensorFlow with auto-determined sigma.
    
    Args:
    - image: Input image tensor
    - kernel_size: Size of the Gaussian kernel (height, width)
    
    Returns:
    - Smoothed image
    """
    
    # Ensure the image is a tensor
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image)
    
    # Ensure 4D tensor [batch, height, width, channels]
    if len(image.shape) == 3:
        image = image[tf.newaxis, :, :, :]
    
    # Create Gaussian kernel for each channel
    def create_gaussian_kernel(size, sigma=1.0):
        """Generate a 2D Gaussian kernel"""
        size = int(size)
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g / g.sum()
    
    # Create kernel
    kernel_height, kernel_width = kernel_size
    kernel = create_gaussian_kernel(kernel_height, sigma)
    
    # Expand kernel for all channels
    num_channels = image.shape[-1]
    kernel_4d = np.expand_dims(kernel, axis=-1)
    kernel_4d = np.repeat(kernel_4d, num_channels, axis=-1)
    kernel_4d = np.expand_dims(kernel_4d, axis=-1)
    
    # Convert kernel to float32 tensor
    kernel_tensor = tf.convert_to_tensor(kernel_4d, dtype=tf.float32)
    
    # Apply convolution
    blurred = tf.nn.depthwise_conv2d(
        input=image, 
        filter=kernel_tensor, 
        strides=[1, 1, 1, 1], 
        padding='SAME'
    )
    
    # Remove batch dimension if it was added
    return blurred[0] if blurred.shape[0] == 1 else blurred


def preprocess(input_data):
    """
    Preprocess image data from various input formats into a standardized tensor.
    
    Args:
    input_data: Can be a file path (str), bytes tensor, numpy array, or PIL Image
    
    Returns:
    A preprocessed tensor of shape (100, 100, 3) with values in [0,1]
    """
    try:
        # Handle PIL Image input
        if isinstance(input_data, Image.Image):
            input_data = np.array(input_data)
        
        # Image decoding and initial processing
        if isinstance(input_data, (str, bytes)) or (isinstance(input_data, tf.Tensor) and input_data.dtype == tf.string):
            # Convert tensor to string if needed
            if isinstance(input_data, tf.Tensor):
                input_data = input_data.numpy()
            if isinstance(input_data, bytes):
                input_data = input_data.decode('utf-8')
            
            # Read and decode the image
            byte_image = tf.io.read_file(input_data)
            image = tf.image.decode_jpeg(byte_image, channels=3)
        else:
            # Handle numpy array or TensorFlow tensor input
            image = tf.convert_to_tensor(input_data)
        
        # Convert to float32
        image = tf.cast(image, tf.float32)
        
        # Ensure shape is correct
        if len(image.shape) != 3:
            raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}")
        
        # Resize the image
        image = tf.image.resize(image, (100, 100))
        
        # Smooth the image
        image = gaussian_blur(image, kernel_size=(3,3), sigma=0.1)
        
        # Normalize the image
        # With deep learing, it is ensential to normalize, so can improve model 
        # performance by ensuring that input data is within a smaller, consistent range, which can help with stability during training.
        image = image / 255.0  # Normalize to [0,1]

        '''
        However, scaling might make the image look lower quality because of the smaller numerical range (0-1), even though 
        this does not actually affect its visual structure when used in a deep learning model. This step is not 
        meant for direct visualization, but rather for preparing data for model input.

        If you are trying to visually inspect the image to verify it after scaling, you can:
        '''
        
        return image
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print(f"Input type: {type(input_data)}")
        if isinstance(input_data, (str, bytes)):
            print(f"Input path: {input_data}")
        raise
# Note that our preprocess function return a Tensorflow tensor, not a numpy array, so when need  to  perform image 
# with OpenCV, we need to convert it to numpy array

# Wrap the preprocess function in a tf.py_function to deal with Frame objects in Opencv
def preprocess_wrapper(input_data):
    """Wrapper function to use with tf.py_function if needed"""
    return tf.py_function(preprocess, [input_data], tf.float32)



def verify (frame, name , siamese_model, detection_threshold, verfication_threshold, LIMIT_IMAGES_TO_COMPARE):
    # Detection Threshold: Metric above which the prediction is considered as positive
    # Verification Threshold: Proportion of positive detections/ total positive samples

    # Example, it te out comes prediction is 0.7, and the detection threshold is 0.5, then the prediction is positive
    # If 30 / 50 images pass the detection threshold, then it pass the verification threshold

    # Create result array
    results = []

    # Load the input image directly from the Webcam, preprocess it
    input_img = preprocess(frame).numpy()

    # Process when the name is not existed in the validation_images folder
    if not os.path.exists(os.path.join(VALIDATION_PATH, name)):
        print("The name does not exist in the system")
        return results, False

    # Loop through all the images in the validation_images folder (with crossponding name)
    path_of_validation_subfolder = os.path.join(VALIDATION_PATH,name)
    print("Compare with images in foler:", path_of_validation_subfolder)

    # Load the preprocessed faces from the .npz file
    data = np.load(os.path.join(path_of_validation_subfolder, 'faces.npz'))

    # Get each validation image preprocess function from Part 3
    # The 'name' user input will be used to named the folder in the validation_images folder
        
    # validation_images  alreadly preprocessed at the enrollment process, so we just need to load the image 9actually, just need
    # to load the array inside faces.npz file, not the image itself)

    # Why need to preprocess at the enrollment process, but not here? -> reduce response time in real time


    validation_faces = data['arr_0']
    for face in validation_faces[:LIMIT_IMAGES_TO_COMPARE]: # Limit the number of images to compare
        # Pass two of these images to the model, with  and store preditcion to the array
        result = siamese_model.predict(list(np.expand_dims([input_img, face], axis=1)))
        results.append(result) 


    verification = np.sum(np.array(results) > detection_threshold) / len(results)
    if verification > verfication_threshold:
        verification = True
    else:
        verification = False

    # Return the verification result for futher processing
    return results, verification




"""---------------------------------------------------------Build the Kivy App---------------------------------------------------------"""
# Make change to the UI here
# The main layout of the app is a vertical box layout with a webcam feed, a button to verify the face, and a label to show the verification result

# Build app layout
class VerificationApp(App):

    def build(self):
        ### Main layout components ###

        # Create the webcam, button, and verification label
        self.webcam = KivyImage(size_hint=(1, .8)) # 1 mean full width, 0.8 mean 80% of the height
        self.button = Button(text="Verify", size_hint=(1, .1), on_press=self.on_verify_button_press)
        self.verification = Label(text="Verification Ready ", size_hint=(1, .1), markup=True) # Enable markup to use color in the text
        self.capture_frame = None # Store the frame when user press the button, this frame will be used as login image to compare with the validation images

        # Add components to the layout
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.webcam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        # Capture video from the camera
        self.capture = cv2.VideoCapture(CAM_ID) # Modify the index to use different camera match your device
        Clock.schedule_interval(self.update, 1.0/30.0) # 30 fps

        ### Load needed model ###

        # Load the Siamese model
        try:
            self.siamese_model = tf.keras.models.load_model('model_saved/fully_siamese_network.h5', 
                                   custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy},
                                   compile=False)
            # Without complie=false cause Warning: WARNING:absl:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.
            
            Logger.info("siamese_model loaded successfully")
        except Exception as e:
            Logger.error(f"Failed to load siamese_model: {e}")
            return None
        
        # Load the Haar Cascade Classifier for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            Logger.error("Error: Could not load face cascade classifier")
            return None

    
        return layout
    
    # Run continuously this function to get the video feed, the number of fps = number of times this function is called per second
    def update(self, *args):

        # Read frame from openCV
        ret, frame = self.capture.read()
        
        if frame is not None:
            self.capture_frame = frame.copy()

            # Flip horizontally and convert image to texture object in Kivy
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') #Since default color format in OpenCV is BGR
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.webcam.texture = texture

    

    # Function to handle the button press, when press, take the current frame, crop the face, and pass to the verify() function to check
    def on_verify_button_press(self, capture_frame):

        if self.capture_frame is None:
            self.verification.text = "No frame captured"
            return

        
        # Load the Haar Cascade Classifier for face detection
        gray_frame = cv2.cvtColor(self.capture_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cropped_face = self.capture_frame[y:y+h, x:x+w]
            resized_face = cv2.resize(cropped_face, (250, 250))
        
            # Run verification
            # with the input image take directly from the webcam, the validation images are taken from the validation_images/name_that_user_input folder
            # Here we limit the number of images to compare to 4
            name = input("Who are trying to login ? ")
            # search if the name is existed in the validation_images folder
            validation_folder = os.path.join(VALIDATION_PATH, name)
            if not os.path.exists(validation_folder):
                self.verification.text = f"[color=FF0000]User {name} does not exist in system[/color]"
                return

            results, verification = verify(resized_face, name, self.siamese_model, DETECTION_THRESHOLD, VERIFICATION_THRESHOLD, LIMIT_IMAGES_TO_COMPARE)

            # Print the raw results and their shape
            for result in results:
                Logger.info(f"Result: {result}")

            Logger.info(f"Verification: {verification}")
        
    
            # Bind the verification result to the label
            if verification:
                self.verification.text = "[color=00FF00]Verification Successful[/color]"
            else:
                self.verification.text = "[color=FF0000]Verification Failed[/color]"
        

        else:
            # Show a dialog if no faces are detectedq
            print("No faces detected, look at the camera and cpature the image again")
            self.verification.text = "No faces detected, look at the camera and capture the image again"
   
# Lauch the app
if __name__ == '__main__':
    VerificationApp().run()