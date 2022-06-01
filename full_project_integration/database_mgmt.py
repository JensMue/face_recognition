#!/usr/bin/env python

"""
This file contains all operations to manage our database.

It includes the following functions:
    create_database    -> creates a new database (either empty or from a specified folder)
    update_database    -> updates the current database with a new entry
    visualize_database -> visualizes the contents of the current database
    save_database_json -> saves the current database to a json-file
    load_database_json -> loads an existing database from a json-file
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from full_project_integration import full_model as fm


def create_database(folder=None):
    """ Creates a new database (from a specified folder of images or empty otherwise)"""
    
    # create an empty database
    columns = ['Timestamp', 'Image', 'Embedding', 'UserID', 'Emotion', 'Emotion_prob'] #, 'Camera'
    database = pd.DataFrame(columns=columns) 
    
    # return if no folder is specified
    if folder == None:
        return database

    # otherwise read in folder contents
    for file_name in [file for file in sorted(os.listdir(folder)) if not file.startswith('.')]:
        img = mpimg.imread(os.path.join(folder,file_name))
        full_model_output = fm.full_model(img, database)[0]
        database = update_database(database, full_model_output, file_name)
        
    return database


def update_database(database, full_model_output, UserID=None):
    """Updates the current database with a new entry"""

    for entry in full_model_output:
        if UserID:
            entry[3] = UserID
        database.loc[len(database)] = entry

    return database


def visualize_database(database):
    """Visualizes the contents of the current database"""

    rows = int((database.shape[0]-1)/6+1)
    fig, ax = plt.subplots(rows, 6, figsize=(16, rows*4))#, constrained_layout=True)
    axes = ax.flatten()
    fig.suptitle(f'Database Visualization', fontsize=20)
    for idx, val in enumerate(database.loc[:,"Image"]):
        axes[idx].imshow(val)
        name = database.loc[idx,"UserID"]
        emotion = database.loc[idx,"Emotion"]
        axes[idx].set_title(f'Name: {name}\nEmotion: {emotion}', fontsize=14)
    plt.show()

    return None


def save_database_json(database, filepath):
    """Saves the current database to a json-file"""
    database.to_json(filepath)
    print(f'Database saved to {filepath}')
    return None


def load_database_json(filepath):
    """Loads an existing database from a json-file"""
    return pd.read_json(filepath)

