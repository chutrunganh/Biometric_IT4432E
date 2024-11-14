# First, instell Kivy by this:
#pip install kivy[full] kivy_examples


# Imprt Kivy dependencies
from kivy.app import App

# For UI
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage # Prevent name conflict with PIL Image

# For others

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.core.window import Window

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist # Import the custom layer
import os
import numpy as np
from PIL import Image


### Need to update to ask user for their camera ID before running the app

# Build app layout
class CamApp(App):

    def build(self):
        # Main layout components


        self.icon = './resources/images/icon.png'  # Supports .png or .ico formats
        
        # Optional: Set window icon (for desktop)
        Window.set_icon('./resources/images/icon.png')

         # Set app title
        self.title = 'TruelyYou'
        
        # Optionally set window title directly
        Window.set_title('TruelyYou')


        # Create the webcam, button, and verification label


        self.webcam = KivyImage(size_hint=(1, .8)) # 1 mean full width, 0.8 mean 80% of the height
        self.button = Button(text="Verify", size_hint=(1, .1), on_press=self.on_verify_button_press)
        self.verification = Label(text="Verification Starting ... ", size_hint=(1, .1), markup=True) #Enable markup to use color in the text
        self.capture_frame = None # Store the frame when user press the button

        # Add components to the layout
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.webcam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        # Capture video from the camera
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30.0) # 30 fps

         # Load the Siamese model
        self.model = tf.keras.models.load_model('model_saved/fully_siamese_network.h5', custom_objects={'L1Dist': L1Dist})


        # When the user press the button, the app will capture the current frame and verify the face, specify
        # on this line: self.button = Button(text="Verify", size_hint=(1, .1), on_press=self.on_verify_button_press)
        


        
       
    


        return layout
    
    def on_verify_button_press(self, instance):

        """Handle verify button press"""
        if self.capture_frame is None:
            self.verification.text = "No frame captured"
            return


        # After user press the button, get the current frame, crop the face, and verify the face
        # Load the Haar Cascade Classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(self.capture_frame, scaleFactor=1.1, minNeighbors=5)
        resized_face = None
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cropped_face = self.capture_frame[y:y+h, x:x+w]
            resized_face = cv2.resize(cropped_face, (250, 250))
        else:
            self.verification.text = "No face detected"
            Logger.warning("No face detected in the frame")
            return

        # Specify arguments for the verify function
        name = "cta2"
        model = self.model
        detection_threshold = 0.5
        verification_threshold = 0.5
        LIMIT_IMAGES_TO_COMPARE = 5



         # Call the verify function
        _, verification = self.verify(resized_face, name, model, detection_threshold, verification_threshold, LIMIT_IMAGES_TO_COMPARE)

        # Change the label text based on the verification result
        if verification:
            self.verification.text = '[color=00ff00]Verification Successful[/color]'  # Green
        else:
            self.verification.text = '[color=ff0000]Verification Failed[/color]'  # Red

    
    # Run continuously to get the video feed
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



    # Function for preprocessing the image (just copy from the Siamese_Network.ipynb)
    def preprocess(self, input_data):
        try:
            # Input validation
            if input_data is None:
                raise ValueError("Input data cannot be None")
                
            # Handle OpenCV BGR format
            if isinstance(input_data, np.ndarray):
                # Convert BGR to RGB if input is from OpenCV
                if len(input_data.shape) == 3 and input_data.shape[2] == 3:
                    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            
            # Handle PIL Image input
            elif isinstance(input_data, Image.Image):
                input_data = np.array(input_data)
            
            # Handle string/bytes input
            elif isinstance(input_data, (str, bytes)) or (isinstance(input_data, tf.Tensor) and input_data.dtype == tf.string):
                if isinstance(input_data, tf.Tensor):
                    input_data = input_data.numpy()
                if isinstance(input_data, bytes):
                    input_data = input_data.decode('utf-8')
                byte_image = tf.io.read_file(input_data)
                input_data = tf.image.decode_jpeg(byte_image, channels=3)
                
            # Convert to tensor
            image = tf.convert_to_tensor(input_data)
            
            # Convert to float32
            image = tf.cast(image, tf.float32)
            
            # Ensure shape is correct
            if len(image.shape) != 3:
                raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}")
            
            # Resize the image
            image = tf.image.resize(image, (100, 100))
            
            # Smooth the image
            image = self.gaussian_blur(image, kernel_size=(3,3), sigma=0.1)
            
            # Normalize to [0,1]
            image = image / 255.0
            
            return image
            
        except Exception as e:
            Logger.error(f"Preprocessing error: {str(e)}")
            return None




    # Note that our preprocess function return a Tensorflow tensor, not a numpy array, so when need  to  perform image 
    # with OpenCV, we need to convert it to numpy array

    # Wrap the preprocess function in a tf.py_function to deal with Frame objects in Opencv
    def preprocess_wrapper(self, input_data):
        """Wrapper function to use with tf.py_function if needed"""
        return tf.py_function(self.preprocess, [input_data], tf.float32)
    
    def gaussian_blur(self, image, kernel_size=(3,3), sigma=0.1):
    
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
    

    # Function to verify the image
    def verify (self, frame, name ,model, detection_threshold, verfication_threshold, LIMIT_IMAGES_TO_COMPARE):
        VALIDATION_PATH = 'application_data/validation_images'

        # Detection Threshold: Metric above which the prediction is considered as positive
        # Verification Threshold: Proportion of positive detections/ total positive samples

        # Example, it te out comes prediction is 0.7, and the detection threshold is 0.5, then the prediction is positive
        # If 30 / 50 images pass the detection threshold, then it pass the verification threshold

        # Create result array
        results = []


        processed_frame = self.preprocess(frame) # from the input frame (after cropping the face), we preprocess it
        if processed_frame is None:
            Logger.error("Failed to preprocess the frame") # NO face found in the frame from camera, so we cannot preprocess it
            return [], False

        # Load the input image directly from the Webcam, preprocess it
        try:
            input_img = processed_frame.numpy()
        except Exception as e:
            Logger.error(f"Failed to convert preprocessed frame to numpy: {str(e)}")
            return [], False

        # Process when the name is not existed in the validation_images folder
        if not os.path.exists(os.path.join(VALIDATION_PATH, name)):
            print("The name does not exist in the system")
            return results, False

        # Loop through all the images in the validation_images folder (with crossponding name)
        path_of_validation_subfolder = os.path.join(VALIDATION_PATH,name)
        print("Compare with images in foler:", path_of_validation_subfolder)

        for image in os.listdir(path_of_validation_subfolder)[:LIMIT_IMAGES_TO_COMPARE]: #Limit to only comapre LIMIT_IMAGES_TO_COMPARE images instead of all images inside folder
            
            # Get each validation image
            # preprocess function from Part 3
            # The 'name' user input will be used to named the folder in the validation_images folder
            
            # validation_images  alreadly preprocessed at the enrollment process, so we just need to load the image

            # Why need to preprocess at the enrollment process, but not here? -> reduce response time in real time
            validation_img = cv2.imread(os.path.join(path_of_validation_subfolder, image), cv2.COLOR_BGR2GRAY)

            if validation_img is None:
                Logger.warning(f"Failed to load validation image: {validation_img}")
                continue


            # Ensure both images have the same shape and number of channels
            if input_img.shape != validation_img.shape:
                print(f"Shape mismatch: input_img shape {input_img.shape}, validation_img shape {validation_img.shape}")
                continue


            

            # Pass two of these images to the model, with  and store preditcion to the array
            result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
            

        verification = np.sum(np.array(results) > detection_threshold) / len(results)
        if verification > verfication_threshold:
            verification = True
        else:
            verification = False


        # Log out the confidence level (taken from results list)
        Logger.info(f"total of iamges matched: {np.sum(np.array(results) > detection_threshold)}")
        

        # Return the verification result for futher processing
        return results, verification


 




if __name__ == '__main__':
    CamApp().run()

