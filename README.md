# Deep Learning for Computer Vision - Final Project

This project is part of the course Deep Learning for Computer Vision (20600) at Bocconi University.
Creators: Group 8 (Botti, A.; Holzach, N.; Lorenzetti, S.; Mueller, J.; Puschiasis, P.; Schrock, F.)

Please refer to 'face_recognition_presentation.pdf' and 'face_recognition_notebook.ipynb' for an overview of the project.

If you have any problems, questions, or suggestions, please contact me at jens.mueller@studbocconi.it
	

## Project description

Our final project combines several deep learning models (face detection, facial recognition, emotion recognition)
to create a marketing application that can recognize reccuring customers in a store and analyze the effect 
of marketing activities by recognizing the customers' emotions at different points in the store.


## Repository structure

The project repository contains four folders. The first three comprise the individual project parts face detection, 
facial recognition, and emotion recognition. The fourth folder combines all three individual parts into a single 
fully-integrated program. The repository structure with a brief description of file contents is shown below.
We use relative paths in our project so please leave the repository folder structure as it is.

```
face_recognition/
│   README.txt                            ---introduction to the project
│   requirements.txt                      ---required packages
│   face_recognition_notebook.ipynb       ---main code file linking all code and data
│   face_recognition_presentation.pdf     ---presentation about our models and business application
│
└───face_detection/
│       configs/                          ---contains configuration files of RetinaFaceModel
│       checkpoints/                      ---training checkpoints
│       data/                             ---used for training, benchmarking, and illustration
│       .py modules                       ---various functionalities related to face detection
│       FaceDetector_training.ipynb       ---model training
│
└───facial_recognition/
│       data/                             ---used for training, benchmarking, and illustration
│       models/                           ---face recognition models (OpenFace pretrained weights)
│       trained_model/                    ---our trained network
│       openface/                         ---the openface GitHub repository
│       .py modules                       ---various functionalities related to data handling
│       facial_recognition_training.ipynb ---model training and benchmarking
│
└───emotion_recognition/
│       models/                           ---emotion recognition models and weights
│       example_images/                   ---example images for illustration in our notebook
│       Emotion_Training.ipynb            ---emotion recognition training notebook (for reconstruction unzip Emotion_Training_full.zip)
│       Emotion_Benchmarking.ipynb        ---emotion recognition benchmarking (for reconstruction unzip Emotion_Training_full.zip)
│       Emotion_Training_full.zip         ---all files required to reconstruct emotion recognition training and benchmarking
│
└───full_project_integration/
        data/                             ---used for examples and illustration
        full_model.py                     ---combination of the three part into one integrated program
        database_mgmt.py                  ---functionalities related to database handling
        db_example.json                   ---exemplary database
```


## Requirements

For the repository to run correctly and completely, all libraries from the requirements.txt file must be installed.
If you are using Windows, you may experience difficulties installing dlib and the corresponding face_recognition library 
(not to be confused with our facial_recognition folder in the repository). To install dlib successfully on Windows, 
it must be build using Cmake and Visual Studio. Once dlib is installed face_recognition can simply be installed using pip.
