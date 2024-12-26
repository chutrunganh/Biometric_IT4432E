# ==============================================================================================================
# Import dependencies
# ==============================================================================================================
import sys
import cv2
import numpy as np

# In this project, we will use PyQt6 for the GUI
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                          QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                          QTabWidget, QMessageBox, QDialog, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# Import dependencies for face recognition
import torch
from facenet_pytorch import InceptionResnetV1
from mtcnn.mtcnn import MTCNN
import os
import uuid
from PIL import Image
import pickle
import json
from numpy import savez_compressed
from torchvision import transforms


# ==============================================================================================================
# Set up some constants, paths and variables
# ==============================================================================================================

# Create a application_data folder to store all app related data
os.makedirs('application_data', exist_ok=True)
# Make the image validation folder
os.makedirs(os.path.join('application_data', 'validation_images'), exist_ok=True)
VALIDATION_PATH = os.path.join('application_data', 'validation_images')

# Path to the settings file, do not create the file yet
SETTINGS_FILE_PATH = os.path.join('application_data', 'settings.json')



# Some constants
CAM_ID = 0 # Defined the Camera ID to use
DETECTION_THRESHOLD = 0.5 # Metric above which the prediction is considered as positive
VERIFICATION_THRESHOLD = 0.6 # Proportion of positive detections/ total positive samples
LIMIT_IMAGES_TO_COMPARE = 4 # Limit the number of images to compare in the verification process


'''
There are two windows in the application, corresponding to two classes. The first window is the 
CameraDetectionDialog class, which is a QDialog window that allows the user to select a camera from a list of detected cameras. 
The second window is the FaceVerificationUI class, which contains two tabs: one for registering a user and another 
for verifying a user.
'''

# =============================================================================================================
# Camera Detection Window
# =============================================================================================================
class CameraDetectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.detected_cameras = [] # List of detected cameras
        self.preferred_camera = None # Between detected cameras, store the camera that user select to use
        global SETTINGS_FILE_PATH 
        self.setup_ui()

        
    def setup_ui(self):
        # UI components are createad top to bottom inside a QVBoxLayout
        self.setWindowTitle("Camera Detection")
        layout = QVBoxLayout(self)

        # Creates a display area for camera feed
        self.camera_feed_display = QLabel() # creates a widget that can display text/images - here used as video preview area  
        self.camera_feed_display.setMinimumSize(640, 480)  # Sets minimum dimensions in pixels, ensures display area won't shrink smaller than VGA resolution
        layout.addWidget(self.camera_feed_display)
        # Setup preview timer to update the camera feed
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        
        # Control ultilities : a horizontal layout to hold the camera combo box (drop down list show all detected cameras), 
        # scan button (to start detect cameras), select button (to select a camera among detected cameras)
        controls_layout = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.scan_btn = QPushButton("Scan for Cameras")
        self.select_btn = QPushButton("Select this camera")
        
        # Connect the buttons to their respective functions        
        self.scan_btn.clicked.connect(self.scan_cameras)
        self.select_btn.clicked.connect(self.select_this_camera)
        self.camera_combo.currentIndexChanged.connect(self.switch_camera)
        # Add the controls to the main layout
        controls_layout.addWidget(QLabel("All cameras available:"))
        controls_layout.addWidget(self.camera_combo)
        controls_layout.addWidget(self.scan_btn)
        controls_layout.addWidget(self.select_btn)
        layout.addLayout(controls_layout)

        # Add label to see how many cameras are detected
        self.detection_camera_result_label = QLabel("Press 'Scan for Cameras' to detect available cameras")
        layout.addWidget(self.detection_camera_result_label)

    ###===============Ultility functions serving the UI (Logical Part)==========================###

    def update_preview(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                # Update to PyQt6's format enum
                image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.camera_feed_display.setPixmap(QPixmap.fromImage(image))

       
    def scan_cameras(self):
        # Clean up existing camera
        if self.cap is not None:
            self.cap.release()
            self.timer.stop()
        
        # Reset lists
        self.detected_cameras = []
        self.camera_combo.clear()
        
        # Scan for cameras: try to open camera index in a range.
        # With each index, try to open the camera using openCV. In case of success, add the camera index to the detected cameras list
        # and add the camera index to the combo box
        # In case of failure, openCV will through error and the camera will not be added to the detected cameras list
        for cam_id in range(10):
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.detected_cameras.append(cam_id)
                    self.camera_combo.addItem(f"Camera {cam_id}")
                cap.release()

        # The scannnig process will loop throught all camera, make them just open then close immadiately. Untill user
        # click the select button, the camera will be open and start the preview. During that time (before user select the camera),
        # just diplay the first camera in the detected camera list to the preview area
        if self.detected_cameras:
            self.cap = cv2.VideoCapture(self.detected_cameras[0])
            self.timer.start(30)

        # Update status for the camera detection result
        if self.detected_cameras:
            self.detection_camera_result_label.setText(f"Found {len(self.detected_cameras)} cameras on the system. Please select one.")
        else:
            self.detection_camera_result_label.setText("No cameras detected in the system!")

    # switch camera when user select a camera from the combo box
    def switch_camera(self, index):
        # Release current camera if exists/ close currently camera connection
        if self.cap is not None:
            self.cap.release()
            self.timer.stop()
        
        # Start new camera if index valid
        if index >= 0 and index < len(self.detected_cameras):
            self.cap = cv2.VideoCapture(self.detected_cameras[index])
            self.timer.start(30) # 30ms refresh rate

    # when user click of select this camera button, store all the detected cameras and the selected camera to a json file
    def select_this_camera(self):
        # Get selected camera ID from combo box
        selected_index = self.camera_combo.currentIndex()
        
        if selected_index >= 0 and self.detected_cameras:
            # Get camera ID from detected list
            preferred_camera = self.detected_cameras[selected_index]
            
            # Save settings
            settings = {
                "camera_list": self.detected_cameras,
                "preferred_camera": preferred_camera
            }
            
            # Save to file
            try:
                with open(SETTINGS_FILE_PATH, 'w') as f:
                    json.dump(settings, f)
                QMessageBox.information(self, "Success", "Camera settings saved successfully to " + SETTINGS_FILE_PATH)
                # Update global CAM_ID
                global CAM_ID
                CAM_ID = preferred_camera
                super().accept()  # Close dialog
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save settings: {str(e)}")
        else:
            QMessageBox.warning(self, "Error", "No cameras detected or selected!")

    # When user clicks the close button, application shuts down,... clean up the camera resources, stop the timer to prevent memory leak
    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()
        event.accept()

# =============================================================================================================
# Registration and Verification Window
# =============================================================================================================

class FaceVerificationUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Verification System")
        # Set minimum size instead of fixed geometry
        self.setMinimumSize(640, 480)  # Minimum usable size for camera feed
        
        global SETTINGS_FILE_PATH 
        global VALIDATION_PATH 
        global CAM_ID
        global DETECTION_THRESHOLD
        global VERIFICATION_THRESHOLD
        global LIMIT_IMAGES_TO_COMPARE

        # Setup main UI components 
        self.setup_ui()
        
        # Initialize camera
        self.setup_camera()

        # Load all our models
        self.init_models()

  
    # =========================================================================================================
    # Set up the main UI 
    # ==========================================================================================================
    def setup_ui(self):
        # UI components are created top to bottom inside a QVBoxLayout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create two tabs, one for registration and one for verification
        self.tabs = QTabWidget()
        self.register_tab = self.create_register_tab()
        self.verify_tab = self.create_verify_tab()
        self.tabs.addTab(self.register_tab, "Register")
        self.tabs.addTab(self.verify_tab, "Verify")
        layout.addWidget(self.tabs)

        # Status bar to display information about the app execution like verification result, registration result, error messages,...
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

    # Create the register tab
    def create_register_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Username input
        self.register_username = QLineEdit()
        self.register_username.setPlaceholderText("Enter username to register")
        layout.addWidget(self.register_username)

        # Camera preview feed
        self.register_camera_label = QLabel()
        layout.addWidget(self.register_camera_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.clicked.connect(self.capture_image)
        self.register_btn = QPushButton("Complete Registration")
        self.register_btn.clicked.connect(self.complete_registration)
        btn_layout.addWidget(self.capture_btn)
        btn_layout.addWidget(self.register_btn)
        layout.addLayout(btn_layout)

        return tab

    # Create the verify tab
    def create_verify_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Username input
        self.verify_username = QLineEdit()
        self.verify_username.setPlaceholderText("Enter username to verify")
        layout.addWidget(self.verify_username)

        # Camera preview feed
        self.verify_camera_label = QLabel()
        layout.addWidget(self.verify_camera_label)

        # Verify button
        self.verify_btn = QPushButton("Verify")
        self.verify_btn.clicked.connect(self.verify_user)
        layout.addWidget(self.verify_btn)

        return tab
    
    #=============================================================================================================
    # Utility functions for the main UI
    #=============================================================================================================
    
    ### UI inside the register tab ###
    def capture_image(self):
        username = self.register_username.text()
        if not username:
            QMessageBox.warning(self, "Error", "Please enter a username")
            return

        ret, frame = self.cap.read()
        if ret:
            # Store user image to the validation folder, inside a subfolder named after the username
            user_folder = os.path.join(VALIDATION_PATH, username)
            os.makedirs(user_folder, exist_ok=True)
            img_path = os.path.join(user_folder, f"{uuid.uuid4()}.jpg")
            cv2.imwrite(img_path, frame)
            self.status_label.setText(f"Image captured and store inside: {img_path}")
    def complete_registration(self):
        username = self.register_username.text()
        if not username:
            QMessageBox.warning(self, "Error", "Please enter a username")
            return

        try:
            self.status_label.setText("Waiting while the embeddings are generated...")
            # From images captured, process the images to extract the face, then generate embeddings for the faces
            self.face_detection(username)
            self.generate_embeddings(username)
            QMessageBox.information(self, "Success", "Registration completed successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Registration failed: {str(e)}")

    ### UI inside the verify tab ###
    def verify_user(self):
        username = self.verify_username.text()
        if not username:
            QMessageBox.warning(self, "Error", "Please enter a username")
            return

        ret, frame = self.cap.read()
        if ret:
            try:
                result = self.verify_face(frame, username)
                if result:
                    self.status_label.setText("Verification successful!")
                    QMessageBox.information(self, "Success", "User verified successfully")
                else:
                    self.status_label.setText("Verification failed!")
                    QMessageBox.warning(self, "Failed", "Verification failed")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Verification error: {str(e)}")

    #=============================================================================================================
    # Set up the camera
    #=============================================================================================================

    # Check if the setting.json file exists, if not, open the camera detection window to let user select the camera
    # If the file exists, show up a message box to inform user that the setting file is already there
    def check_camera_settings(self):

        if os.path.exists(SETTINGS_FILE_PATH):

            # Show message box to inform user that the setting file is already there
            QMessageBox.information(
            self,
            "Settings Found",
            f"Settings file already exists at:\n{SETTINGS_FILE_PATH}\n\nIf you wish to change the camera settings, please delete the file and restart the application."
        )
    
        else:
            dialog = CameraDetectionDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                sys.exit(1)
                
        # Open the settings file and read the preferred camera, assign the value to the global CAM_ID
        global CAM_ID # cahnge the golbal CAM_ID to the selected camera
        with open(SETTINGS_FILE_PATH, 'r') as f:
            settings = json.load(f)
            CAM_ID = settings['preferred_camera']
            self.status_label.setText(f"Using camera {CAM_ID}")
    
    def setup_camera(self):
        
        # Get the preferred camera from the settings file
        self.check_camera_settings()

        self.cap = cv2.VideoCapture(CAM_ID)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # Update the camera feed, show the camera feed in the label
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            if self.tabs.currentIndex() == 0:
                self.register_camera_label.setPixmap(pixmap.scaled(640, 480))
            else:
                self.verify_camera_label.setPixmap(pixmap.scaled(640, 480))
        
    # ==============================================================================================================
    # Set up model and the enrollment/recognition, verification process backend
    # ==============================================================================================================

    # Try to load models, if failed, show a message box to inform user that the model is not found
    def init_models(self):
        try:
            self.detector = MTCNN()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            with open('./model_saved/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('./model_saved/svm_model.pkl', 'rb') as f:
                self.svm_model = pickle.load(f)
                
        except FileNotFoundError:
            self.status_label.setText("Error: Required model files not found in ./model_saved/")
            QMessageBox.critical(
                self,
                "Model Files Not Found",
                "Required model files not found in ./model_saved/\n\nPlease ensure that the model files are placed in the correct directory."
            )
            raise
        except Exception as e:
            self.status_label.setText(f"Error initializing models: {str(e)}")
            raise

    
    ### Enrollemnt/ Registration Process ###
    # Detect and crop images by faces
    def face_detection(self, username):
        user_folder = os.path.join(VALIDATION_PATH, username)
        face_folder = os.path.join(user_folder, 'face') 
        os.makedirs(face_folder, exist_ok=True) # create the face folder inside the user name  folder
        
        try:
            # Get list of images from user folder
            image_files = [f for f in os.listdir(user_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                raise Exception("No images found in user folder")
                
            self.status_label.setText("Processing images...")
            detector = MTCNN()
            faces = []
            
            # Process each image
            for image_file in image_files:
                image_path = os.path.join(user_folder, image_file)
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(image_rgb)
                
                if detections:
                    x, y, width, height = detections[0]['box']
                    face = image_rgb[y:y+height, x:x+width]
                    face_image = Image.fromarray(face).resize((160, 160))
                    faces.append(np.array(face_image))
            
            if not faces:
                raise Exception("No faces detected in any images")
                
            # Save processed faces
            faces_array = np.array(faces)
            savez_compressed(os.path.join(face_folder, 'faces.npz'), faces_array)
            self.status_label.setText("Face processing completed!")
            return True
            
        except Exception as e:
            self.status_label.setText(f"Error processing images: {str(e)}")
            return False

    # With from all the images, generate the embeddings for the faces, store the embeddings in the user folder
    def generate_embeddings(self, username):
        try:
            user_folder = os.path.join(VALIDATION_PATH, username)
            face_folder = os.path.join(user_folder, 'face')
            embedding_folder = os.path.join(user_folder, 'embeddings')
            os.makedirs(embedding_folder, exist_ok=True) # create the embedding folder inside the user name  folder
            
            # Load FaceNet model to extract embeddings
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            
            # Load processed faces
            data = np.load(os.path.join(face_folder, 'faces.npz'))
            faces = data['arr_0']
            
            # Prepare image transformation
            transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225]),
            ])
            
            embeddings = []
            for face in faces:
                image = Image.fromarray(face)
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    embedding = facenet_model(img_tensor).cpu().numpy()
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
            savez_compressed(os.path.join(embedding_folder, 'embeddings.npz'), embeddings)
            self.status_label.setText("Embeddings generated successfully!")
            return True
            
        except Exception as e:
            self.status_label.setText(f"Error generating embeddings: {str(e)}")
            return False

    ###  Verification Process ###

    # Similiar to the generate_embeddings function, but design to only work with a single image, instead of a folder of images
    def generate_single_embedding(self, face_array):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            
            # Convert to PIL image and apply transforms
            image = Image.fromarray(face_array)
            transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = facenet_model(img_tensor).cpu().numpy()
            return embedding[0]  # Return the first (and only) embedding
            
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")

    # From login image, extract the face, generate the embedding, then compare the embedding with the stored embeddings
    def verify_face(self, frame, username):
        try:
            # Check if user exists
            user_path = os.path.join(VALIDATION_PATH, username)
            if not os.path.exists(user_path):
                QMessageBox.warning(self, "Error", "User not found in the system")
                raise Exception("User not found in the system")

            # Extract face and generate embedding
            detector = MTCNN()
            image = Image.fromarray(frame).convert('RGB')
            image_np = np.array(image)
            detections = detector.detect_faces(image_np)
            
            if not detections:
                raise Exception("No face detected in the input image")
                
            # Process detected face
            x, y, width, height = detections[0]['box']
            face = image_np[y:y+height, x:x+width]
            face_image = Image.fromarray(face).resize((160, 160))
            face_array = np.array(face_image)
            
            # Generate embedding for input face using the new method
            input_embedding = self.generate_single_embedding(face_array)
            
            # Load user embeddings
            embedding_path = os.path.join(user_path, 'embeddings', 'embeddings.npz')
            data = np.load(embedding_path)
            stored_embeddings = data['arr_0']
            
            # Compare embeddings
            results = []
            for stored_embedding in stored_embeddings[:LIMIT_IMAGES_TO_COMPARE]:
                # Prepare pair for comparison
                input_flat = input_embedding.flatten()
                stored_flat = stored_embedding.flatten()
                pair = np.concatenate((input_flat, stored_flat))
                
                # Scale features
                pair_scaled = self.scaler.transform([pair])
                
                # Get prediction with probability
                prediction = self.svm_model.predict(pair_scaled)[0]
                probability = self.svm_model.predict_proba(pair_scaled)[0]

                # Add detection threshold
                global DETECTION_THRESHOLD# Minimum confidence required for a match
                if probability[1] >= DETECTION_THRESHOLD:  # probability[1] is confidence for positive class
                    results.append(prediction)

                # Calculate final verification with validation threshold
                global VERIFICATION_THRESHOLD # Minimum proportion of positive matches required
                positive_matches = sum(results)
                total_comparisons = len(results)

                verification_result = (positive_matches / total_comparisons) > VERIFICATION_THRESHOLD if total_comparisons > 0 else False
                return verification_result
            
        except Exception as e:
            raise Exception(f"Verification error: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceVerificationUI()
    window.show()
    sys.exit(app.exec())