#!/usr/bin/env python

"""
This file contains functions to create a single large dataset for face recognition.

It includes the following functions:
	load_data_FacePix -> loads the FacePix dataset
	load_data_uficrop -> loads the uficrop dataset
	load_data_lfwcrop -> loads the lfwcrop dataset
	merge_data -> merges different datasets
"""

# import relevant modules
import os
import re
import numpy as np
import cv2
from unidecode import unidecode
from PIL import Image


def load_data_FacePix(data_folder):
    """Function to load images from FacePix dataset."""
    
    # adjust folder path
    folder_FacePix = data_folder + '/FacePix' 
    
    # create output dictionary
    data_FacePix = {} 
    
    # get alphabetically sorted list of files in folder
    file_list = [file for file in sorted(os.listdir(folder_FacePix)) if not file.startswith('.')]
    progress = 0

    # loop through files
    for filename in file_list:  
        # create dictionary entry for new labels
        label = filename[0:2]
        if label not in data_FacePix.keys():
            data_FacePix[label] = []
        # thin out data to reduce redundancy and exclude images with strong angle
        if int(filename[-6]) % 2 == 0 and len(data_FacePix[label]) < 30:
            # read images
            img = cv2.imread(os.path.join(folder_FacePix,filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                data_FacePix[label].append(img)
                progress += 1
                if progress % 120 == 0:
                	print('Progress:', 100 * progress / 900,'%')  
    
    # format output as numpy arrays
    for label, images in data_FacePix.items():
        data_FacePix[label] = np.array(images)
    
    # return output
    return data_FacePix


def load_data_uficrop(data_folder):
	"""Function to load images from ufi dataset."""

	# adjust folder path
	folder_ufi = data_folder + '/ufi-cropped' 

	# initialize dictionaries
	folder_name_dict = {} #store folder names and real name of person
	data_ufi = {} #store final dataset

	# prepare regex
	regex = re.compile('[^a-zA-Z]')

	# go into train folder and extract labels and images
	train_folder = folder_ufi + '/' + 'train'

	progress = 0
	for person in os.listdir(train_folder):
		image_array = [] # initialize image array
		if person != '.DS_Store': #check for MacOS
			# open .txt file to extract label
			txt_path = train_folder + '/' + person + '/info.txt'

			#print(f'FILE NAME {txt_path} ')

			with open(txt_path, 'r', encoding="utf8") as f:

				#print(f'LABEL {f.readline().strip()} ')

				label = f.readline().strip()

			# split label to get first and last name
			names = label.split('_')
			# clean first and last name
			names = [unidecode(name) for name in names] #transform non-English letters to English ones
			names = [regex.sub('', name) for name in names] #remove non-letter characters
			names = [name for name in names if name != ''] #remove empty strings
			# correctly order first and last names
			first_names = [name for name in names if not name.isupper()] #info.txt files use capitalization for last names
			last_names = [name for name in names if name.isupper()]
			first_names.extend(last_names) #join names
			names = first_names
			names = [name.title() for name in names] #capitalize correctly
			# combine names to create label
			label = ('_').join(names)
			if label == 'George_Walker_Bush': #fix George W Bush edge case to align with lfw dataset
				label = 'George_W_Bush'
			# add name to dictionary
			folder_name_dict[person] = label

			# iterate through person's folder and add each image to dictionary
			for img in os.listdir(train_folder + '/' + person):
				if img[-3:] == 'pgm': #only consider image files
					# open image and add to array
					image_path = train_folder + '/' + person + '/' + img
					image = np.array(Image.open(image_path))
					image_array.append(image)
					progress += 1
					if progress % 500 == 0:
						print('Progress:', 100 * progress / 5000,'%')
			# add images to output dictionary
			data_ufi[label] = np.array(image_array)

	# go into test folder and extract remaining images
	test_folder = folder_ufi + '/' + 'test'

	for person in os.listdir(test_folder):
		# initialize image array
		image_array = []
		if person != '.DS_Store': #check for MacOS
			# iterate through person's folder and add each image to dictionary
			for img in os.listdir(test_folder + '/' + person):
				if img[-3:] == 'pgm': #only consider image files
					# open image and add to array
					image_path = test_folder + '/' + person + '/' + img
					image = np.array(Image.open(image_path))
					image_array.append(image)
					progress += 1
					if progress % 500 == 0:
						print('Progress:', 100 * progress / 5000,'%')
			# add images to output dictionary
			train_images = data_ufi[folder_name_dict[person]]
			all_images = np.vstack((train_images, np.array(image_array))) #combine train and test images
			data_ufi[folder_name_dict[person]] = all_images #modify data_ufi dictionary

	# return output
	return data_ufi


def load_data_lfwcrop(data_folder):
	"""Function to load images from lfwcrop dataset."""

	# initialize output
	data_lfwcrop = {}

	# prepare regex
	regex = re.compile('[^a-zA-Z]')

	# go faces folder and extract images and labels
	faces_folder = data_folder
	file_list = [file for file in os.listdir(faces_folder) if not file.startswith('.')] #remove hidden files

	progress = 0
	for file in file_list:
		# get filepath of image
		image_path = faces_folder + '/' + file

		# split filename and clean it to extract names for label
		names = file.split('_')
		names = names[:-1] #cut off file extension and number
		names = [unidecode(name) for name in names] #transform non-English letters to English ones
		names = [regex.sub('', name) for name in names] #remove non-letter characters
		names = [name.title() for name in names] #capitalize names

		# combine names to create label
		label = '_'.join(names)

		# extract image
		img = np.array(Image.open(image_path))

		# check if label already in dataset
		if label not in data_lfwcrop:
			data_lfwcrop[label] = np.array([img]) #if not, add image array to output
		elif label in data_lfwcrop:
			existing_values = data_lfwcrop[label] #otherwise extract existing image arrays
			new_values = np.vstack((existing_values, np.array([img])))
			data_lfwcrop[label] = new_values #and add old+new image arrays to the output

		progress += 1
		if progress % 1300 == 0:
			print('Progress:', 100 * progress / 13500,'%')
	# return output
	return data_lfwcrop


# define function to merge dictionaries containing image data from different datasets
def merge_data(list_of_dictionaries):
	"""
	Function to merge dictionaries containing image data.
	Dictionaries are structured the same way with
	keys representing the label/name of the person and
	values being a numpy array containing the image arrays.

	Argument:
	list_of_dictionaries: list object containing the dictionaries that should be merged

	Returns:
	dictionary object containing all the merged image data
	"""

	# use first dictionary in list as base
	data = {key : [values] for key, values in list_of_dictionaries[0].items()}

	# loop through remaining dictionaries
	for d in list_of_dictionaries[1:]:

		# loop through each key in dictionary
		for key, values in d.items():

			# check if label exists in base dictionary
			if key not in data:
				data[key] = [values] #if not, add the image arrays
			elif key in data:
				data[key].append(values) #and add old+new image arrays to the base dictionary

	# return the final dictionary
	return data