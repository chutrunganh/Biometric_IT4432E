The `application_data` folder contains several subfolders, with `validation_images` being one of the most important ones.

The `validation_images` folder stores images used for the authentication process. When users register, their facial images are scanned and stored in a personalized subfolder named after their username within the validation_images directory. 

During authentication, the system captures an image of the user's face via webcam and compares it with all stored images in their corresponding `validation_images/user_name` folder.

The structure of the application_data folder is organized as follows:

```plaintext
application_data
│
├───validation_images
│   │
│   ├───user1
│   │   │   image1.jpg
│   │   │   image2.jpg
│   │   │   ...
│   │
│   ├───user2
│   │   │   image1.jpg
│   │   │   image2.jpg
│   │   │   ...
```