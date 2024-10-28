This folder may contains some subfolders, one of the most important is the `validation_images` subfoler.

This folder contains the images that will be used fro the authentication process. With each user that registers then scan their images on the system, store these images in a subfolder with the user's name, inside the `validation_images` folder. Later, when the user wants to authenticate, the system capture a picture of user's face  from webcam and compare it with all the images stored in the `validation_images/user_name` folder.

THe structure of the `application_data` folder should be like this:

```plaintext
application_data
│
|───validation_images
|   │
│   └───user1
│   │   │   image1.jpg
│   │   │   image2.jpg
│   │   │   ...
│   │
│   └───user2
│       │   image1.jpg
│       │   image2.jpg
│       │   ...
```