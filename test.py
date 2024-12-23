# main_app.py
import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                          QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                          QTabWidget, QMessageBox, QDialog, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import torch
from facenet_pytorch import InceptionResnetV1
from mtcnn.mtcnn import MTCNN
import os
import uuid
from PIL import Image
import pickle
import json
from numpy import savez_compressed

class CameraDetectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.detected_cameras = []
        self.preferred_camera = None
        self.setup_ui()
        self.SETTINGS_FILE_PATH = 'application_data/settings.json'

    def setup_ui(self):
        self.setWindowTitle("Camera Detection")
        layout = QVBoxLayout(self)

        # Camera preview
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(640, 480)
        layout.addWidget(self.preview_label)

        # Controls
        controls_layout = QHBoxLayout()
        
        self.camera_combo = QComboBox()
        self.scan_btn = QPushButton("Scan for Cameras")
        self.select_btn = QPushButton("Select Camera")
        
        controls_layout.addWidget(QLabel("Camera:"))
        controls_layout.addWidget(self.camera_combo)
        controls_layout.addWidget(self.scan_btn)
        controls_layout.addWidget(self.select_btn)
        
        layout.addLayout(controls_layout)

        # Add status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Connect signals
        self.scan_btn.clicked.connect(self.scan_cameras)
        self.select_btn.clicked.connect(self.accept)
        self.camera_combo.currentIndexChanged.connect(self.switch_camera)

        # Setup preview timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        
    def scan_cameras(self):
        self.detected_cameras = []
        self.camera_combo.clear()
        
        for cam_id in range(10):
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.detected_cameras.append(cam_id)
                    self.camera_combo.addItem(f"Camera {cam_id}")
                cap.release()
        
        if self.detected_cameras:
            self.switch_camera(0)
        
    def switch_camera(self, index):
        if self.cap is not None:
            self.cap.release()
            
        if index >= 0 and index < len(self.detected_cameras):
            self.cap = cv2.VideoCapture(self.detected_cameras[index])
            self.timer.start(30)
            
    def scan_cameras(self):

        self.status_label.setText("Scanning for cameras...")
        self.detected_cameras = []
        self.camera_combo.clear()
        
        for cam_id in range(10):
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.detected_cameras.append(cam_id)
                    self.camera_combo.addItem(f"Camera {cam_id}")
                cap.release()
        
        if self.detected_cameras:
            self.switch_camera(0)
            self.status_label.setText(f"Found {len(self.detected_cameras)} cameras")
        else:
            self.status_label.setText("No cameras detected!")

    def save_settings(self):
        if not self.detected_cameras:
            return False
            
        settings = {
            "camera_list": self.detected_cameras,
            "preferred_camera": self.get_selected_camera()
        }

    
        
        os.makedirs(os.path.dirname(self.SETTINGS_FILE_PATH), exist_ok=True)
        with open(self.SETTINGS_FILE_PATH, 'w') as f:
            json.dump(settings, f)
            
        return True
    
    def update_preview(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                # Update to PyQt6's format enum
                image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.preview_label.setPixmap(QPixmap.fromImage(image))

    def get_selected_camera(self):
        if self.camera_combo.currentIndex() >= 0:
            return self.detected_cameras[self.camera_combo.currentIndex()]
        return 0
    
    def accept(self):
        if self.save_settings():
            super().accept()
        else:
            QMessageBox.warning(self, "Error", "No cameras detected or selected!")

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()
        event.accept()

class FaceVerificationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Verification System")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize models
        self.init_models()
        
        # Setup main UI components 
        self.setup_ui()
        
        # Initialize camera
        self.setup_camera()
        
        # Create required folders
        os.makedirs('application_data/validation_images', exist_ok=True)

    def check_camera_settings(self):
        SETTINGS_FILE_PATH = 'application_data/settings.json'
        if not os.path.exists(SETTINGS_FILE_PATH):
            dialog = CameraDetectionDialog(self)
            if dialog.exec() == QDialog.accepted:
                settings = {
                    "camera_list": dialog.detected_cameras,
                    "preferred_camera": dialog.get_selected_camera()
                }
                with open(SETTINGS_FILE_PATH, 'w') as f:
                    json.dump(settings, f)

    def init_models(self):
        self.detector = MTCNN()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        with open('./model_saved/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open('./model_saved/svm_model.pkl', 'rb') as f:
            self.svm_model = pickle.load(f)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tabs
        self.tabs = QTabWidget()
        self.register_tab = self.create_register_tab()
        self.verify_tab = self.create_verify_tab()
        self.tabs.addTab(self.register_tab, "Register")
        self.tabs.addTab(self.verify_tab, "Verify")
        layout.addWidget(self.tabs)

        # Status bar
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

    def create_register_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Username input
        self.register_username = QLineEdit()
        self.register_username.setPlaceholderText("Enter username to register")
        layout.addWidget(self.register_username)

        # Camera preview
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

    def create_verify_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Username input
        self.verify_username = QLineEdit()
        self.verify_username.setPlaceholderText("Enter username to verify")
        layout.addWidget(self.verify_username)

        # Camera preview
        self.verify_camera_label = QLabel()
        layout.addWidget(self.verify_camera_label)

        # Verify button
        self.verify_btn = QPushButton("Verify")
        self.verify_btn.clicked.connect(self.verify_user)
        layout.addWidget(self.verify_btn)

        return tab

    def setup_camera(self):
        
        self.check_camera_settings()
        
        # Read camera settings

        with open('application_data/settings.json', 'r') as f:
            settings = json.load(f)
            self.cam_id = settings['preferred_camera']

        self.cap = cv2.VideoCapture(self.cam_id)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

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

    def capture_image(self):
        username = self.register_username.text()
        if not username:
            QMessageBox.warning(self, "Error", "Please enter a username")
            return

        ret, frame = self.cap.read()
        if ret:
            user_folder = f'application_data/validation_images/{username}'
            os.makedirs(user_folder, exist_ok=True)
            img_path = os.path.join(user_folder, f"{uuid.uuid4()}.jpg")
            cv2.imwrite(img_path, frame)
            self.status_label.setText(f"Image captured: {img_path}")

    def complete_registration(self):
        username = self.register_username.text()
        if not username:
            QMessageBox.warning(self, "Error", "Please enter a username")
            return

        try:
            self.process_user_images(username)
            self.generate_embeddings(username)
            QMessageBox.information(self, "Success", "Registration completed successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Registration failed: {str(e)}")

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

    def process_user_images(self, username):
        # Implementation of face detection and processing
        pass

    def generate_embeddings(self, username):
        # Implementation of embedding generation
        pass

    def verify_face(self, frame, username):
        # Implementation of face verification
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceVerificationUI()
    window.show()
    sys.exit(app.exec())