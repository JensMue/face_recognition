#!/usr/bin/env python

"""
This file contains a function to perform common preprocessing operations.
"""

# import relevant modules
import numpy as np
import cv2
import dlib
import facial_recognition.openface.openface.align_dlib as openface
import tensorflow as tf
import os


def dict_to_model_input(data, grey2rgb=False, normalize=True, alignment=True, new_size=None):
    """
    Function that takes the output of the merge_data function in data_creation.py as input
    and applies several pre-processing steps to the data so that it can be fed into a neural network.
     
    Arguments:
    data: dict - dictionary object containing labels as keys and image arrays as values
    grey2rgb: bool - whether to convert images from greyscale to rgb. Default = False
    normalize: bool - whether to normalize the arrays to lie between 0 and 1. Default = True
    alignment: bool - whether to align the faces around facial landmarks. Default = True
    new_size: tupel - if not None, input tupel (width, height) to resize the image. Default = None
    
    Returns:
    tupel (X, y) containing 3D numpy arrays and their corresponing label
    """

    # initiliaze outputs
    image_output = []
    label_output = []

    #print(os.getcwd())

    # initialize aligning tool
    face_aligner = openface.AlignDlib('facial_recognition/models/shape_predictor_68_face_landmarks.dat') #use openface landmarks

    # iterate through dictionary keys
    progress = 0
    for label in data.keys():

        # iterate through list within each label
        for idx in data[label]:

            # iterate through images for each label
            for img in idx:

                # if specified, convert greyscale to RGB
                if grey2rgb == True:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # if specified, align the face in the image
                if alignment == True:		
                    bounding_box = dlib.rectangle(left=0, top=0, right=img.shape[1], bottom=img.shape[0]) #use whole image as bounding box
                    img = face_aligner.align(new_size[0], img, bounding_box, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE) #align face

                # if specified, scale image between 0 and 1
                img = img.astype('float32') #convert from integers to floats
                img /= 255.0 #normalize to the range 0-1

                # if specified, resize the image
                if new_size is not None and alignment is not True:
                    img = cv2.resize(img, new_size, interpolation = cv2.INTER_AREA)

                # add processed image and corresponding label to output
                image_output.append(img)
                label_output.append(label)

                progress += 1
                if progress % 968 == 0:
                    print('Progress:', 100*progress/19054, '%')

    # return output
    return (np.array(image_output), label_output)