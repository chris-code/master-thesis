
# coding: utf-8

# # Generating adversarial examples (AEs)
# 
# This file generates adversarial examples on a dataset with the imagenet network provided by caffe. The data is expected to reside in *data_root*, with the following structure:
# - The images of each class reside in their own folder, named according to the imagenet label (e.g. 'n02112018')
# - There are no other folders or files in *data_root*
# - There are no other folders or files in *data_root*'s class-specific subfolders
# 
# For each class, a random representative will be selected. Adversarial examples pretending to belong to each class will be generated for it. The program tries to achive *CONFIDENCE_TARGET* certainty of the network, but will not spend longer than *MAX_ITERATIONS* on generating one AE. The number of considered classes can be restricted with *CLASS_LIMIT* (*None* means cross-generate AEs for all classes)
# 
# The output is saved to *OUTPUT_ROOT*, with the following structure:
# - For each class, there is a directory with the class name
# - The directory contains the AEs as .npy files, with the adversarial class as the filename
# - There is a history.csv file in each directory that records the final confidence and the required iterations for each AE.
# - The history.csv file has the format (class_representative_filename, adversarial_class_name, confidence, iterations)
# 
# We start with imports and by setting paths and constants.

# In[1]:

import random
import sys
import os
import csv
import numpy as np
import caffe
import adex
import adex.core
import adex.googlenet

CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
DATA_ROOT = '/media/sf_Masterarbeit/data/ILSVRC2012_img_train'
OUTPUT_ROOT = '/media/sf_Masterarbeit/data/ILSVRC2012_img_train_AE_10_3.0'
BATCH_SIZE = 1

AE_GRAD_COEFF = 3.0
CONFIDENCE_TARGET = 0.9
MAX_ITERATIONS = 500
CLASS_LIMIT = 10


# Next, we load the network, the labels and a transformer for i/o.

# In[2]:

net = adex.googlenet.load_model(CAFFE_ROOT, BATCH_SIZE)
labels = adex.googlenet.load_labels(CAFFE_ROOT)
transformer = adex.googlenet.build_transformer(net)


# Lastly, it is handy to have a function that turns the representative of one class into an AE of another class, adhering to the defined file structure.

# In[3]:

def make_ae_for_paths(infile, outfile, net, labels, target_class_name,
                                   AE_GRAD_COEFF, CONFIDENCE_TARGET, MAX_ITERATIONS):
    # Load class representative
    image = caffe.io.load_image(infile)
    image = transformer.preprocess('data', image)
    image = np.expand_dims(image, 0)
    
    target_label = adex.googlenet.get_label_from_class_name(labels, target_class_name)
    target_label = np.array([target_label]) # Caffe-friendly format for labels
    
    adversarial_image, confidence, iterations = adex.core.make_adversarial(net, image, target_label, AE_GRAD_COEFF,
                                                                               CONFIDENCE_TARGET, MAX_ITERATIONS)
    np.save(outfile, adversarial_image)
    return confidence, iterations


# Finally, we generate AEs.

# In[4]:

class_directories = os.listdir(DATA_ROOT) # The list of classes (conincide with their directory names)
random.shuffle(class_directories)
if CLASS_LIMIT is not None: # Make sure we only consider CLASS_LIMIT many classes
    class_directories = class_directories[:CLASS_LIMIT]

for directory_index, class_directory in enumerate(class_directories): # Iterate over all classes / their directories
    print('\nGenerating AEs for class ({0}/{1}): {2} '.format(directory_index + 1, len(class_directories), class_directory))
    
    random_class_representative = random.choice(os.listdir(DATA_ROOT + '/' + class_directory))
    
    # Prepare output directory
    try:
        os.mkdir(OUTPUT_ROOT + '/' + random_class_representative[:-5])
    except OSError:
        pass # Directory already exists
    
    history = [] # Keeps track of confidence and iterations
    for target_class_name in class_directories: # Iterate over all classes again
        sys.stdout.write('.')
        sys.stdout.flush()
        
        #Generate AEs
        infile = DATA_ROOT + '/' + class_directory + '/' + random_class_representative
        outfile = OUTPUT_ROOT + '/' + random_class_representative[:-5] + '/' + target_class_name + '.npy'
        confidence, iterations = make_ae_for_paths(infile, outfile, net, labels, target_class_name,
                                                   AE_GRAD_COEFF, CONFIDENCE_TARGET, MAX_ITERATIONS)
        
        # Add data to history
        history.append( [random_class_representative, target_class_name, confidence, iterations] )
    
    # Write history to file
    with open(OUTPUT_ROOT + '/' + class_directory + '_' + 'history.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for target_class_history in history:
            writer.writerow(target_class_history)

