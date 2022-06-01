#!/usr/bin/env python

"""
This file combines our three individual models to a single integrated program.

First the three individual parts are set up. Then the full_model function combines all three 
so that it can take in an image and output the faces, embeddings, matches, and emotions 
in a format that can be added to our database.
"""

import os
import time
import numpy as np
import cv2
import tensorflow as tf
from absl import logging
from keras.models import model_from_json


### PART 1 - FACE DETECTION ###

# Import relevant modules
from face_detection.models import RetinaFaceModel
from face_detection.utils import (load_yaml, set_memory_growth,  pad_input_image, recover_pad_output, draw_bbox_landm, align_face)

# Set working device
set_memory_growth() # Avoid memory problems
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # use just gpu, uses cpu if no CUDA compatible device is found

# Get rid of the ugly tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)

# Define network (if run for the first time will download MobileNetv2/ResNet50 weights for the backbone)
cfg = load_yaml('face_detection/configs/retinaface_mbv2.yaml')
detector = RetinaFaceModel(cfg, training=False, iou_th=0.25, score_th=0.95)

# Load last checkpoint with weights
checkpoint = tf.train.Checkpoint(model = detector)
checkpoint.restore(tf.train.latest_checkpoint('face_detection/checkpoints/' + cfg['sub_name']))

def p1_model(image):
    '''Returns all faces in an image as a list of numpy arrays.'''
    
    # Reduce size in case of very large images
    if image.shape[0] > 1280:
        rescale_factor = 1280 / image.shape[0]
        image = cv2.resize(image, (int(image.shape[1] * rescale_factor), 1280))

    # Preprocess image
    frame = np.float32(image.copy())   # convert to float32 
    frame = frame[..., ::-1]           #convert BGR to RGB
    frame, pad_params = pad_input_image(frame, max_steps=max(cfg['steps'])) # pad input image to avoid unmatched shape problems

    # Detect faces
    det_faces = detector(frame[np.newaxis, ...]).numpy()
    det_faces = recover_pad_output(det_faces, pad_params) # recover padding effect

    # Align faces
    aligned_faces = [al_face / 255.0 for al_face in (align_face(image, face, min_face_size = 32, 
        o_size = (96,96), eyes_wh = (0.25,0.15)) for face in det_faces) if al_face is not None]

    return aligned_faces, det_faces


### PART 2 - FACIAL RECOGNITION ###

# Define pretrained network
# from facial_recognition.models.openface_model import model
# model.load_weights('facial_recognition/models/openface_weights.h5')

# Load our own model trained with transfer learning
from tensorflow import keras
model = keras.models.load_model('facial_recognition/trained_model')

def p2_model(p1_output, database):
    '''For each face in a list, it returns an embedding and a label/id.'''
    
    # get embedding for each face in image
    faces = np.array(p1_output)
    embeddings = model.predict(faces) 

    # compare embeddings to database entries
    matches = []
    for emb in embeddings:
        distances = []
        for emb_db in database.loc[:,'Embedding']:
            distances.append(np.linalg.norm(emb - emb_db))
        try:
            dist, idx = min((dist, idx) for (idx, dist) in enumerate(distances))
            # threshold = 0.85 for better demonstration. If unknown UserID: create a new ID
            matches.append(database.loc[idx,'UserID'] if dist < 0.85 else 'Unkown') 
        except:
            matches.append('Unkown')

    return embeddings, matches


### PART 3 - EMOTION RECOGNITION ###

class FacialExpressionModel(object):

    EXPRESSIONS_LIST = ['neutral','happy','surprise', 'sad','anger', 'disgust','fear']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)

        return FacialExpressionModel.EXPRESSIONS_LIST[np.argmax(self.preds)], self.preds

# Create instance of model with our weights
pic_to_exp = FacialExpressionModel("emotion_recognition/models/xception_model.json",
                                   "emotion_recognition/models/xception_20.h5")

def p3_model(p1_output):
    '''For each face in a list, it returns an emotion prediction and probabilities.'''
    
    emotion = []
    emotion_prob = []

    for face in p1_output:
        # preprocess image for this part
        image = tf.convert_to_tensor(face, dtype=tf.float32)
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, tf.float32)

        # predict emotion
        y_pred, y_probabilities = pic_to_exp.predict_emotion(image)
        emotion.append(y_pred)
        emotion_prob.append(y_probabilities)

    return emotion, emotion_prob


### FULL MODEL ###

def full_model(image, database):
    '''Combines three models.'''

    timestamp = time.time()

    aligned_faces, det_faces = p1_model(image)

    embeddings, matches = p2_model(aligned_faces, database)

    emotion, emotion_prob = p3_model(aligned_faces)

    # Combine the individual model outputs into a single database entry
    full_model_output = []
    for idx, face in enumerate(aligned_faces):
        full_model_output.append([
            timestamp, 
            aligned_faces[idx],
            embeddings[idx], 
            matches[idx], 
            emotion[idx], 
            emotion_prob[idx]])

    return full_model_output, det_faces

