
# coding: utf-8

# # Generating adversarial examples (AEs)
# 
# This file generates adversarial examples on a dataset with the imagenet network provided by caffe. The data is expected to reside in *data_root*, with the following structure:
# - The images of each class reside in their own folder, named according to the imagenet label (e.g. 'n02112018')
# - There are no other folders or files in *data_root*
# - There are no other folders or files in *data_root*'s class-specific subfolders
# 
# For each class, a random representative will be selected. Adversarial examples pretending to belong to each class will be generated for it. The program tries to make the network believe the image to belong to this class with *CONFIDENCE_TARGET* certainty, but will not spend longer than *MAX_ITERATIONS* on generating one AE.
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
import adex.googlenet

CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
DATA_ROOT = '/media/sf_Masterarbeit/data/ILSVRC2012_img_train_t3'
OUTPUT_ROOT = '/media/sf_Masterarbeit/data/ILSVRC2012_img_train_t3_adversarial'
BATCH_SIZE = 1

AE_GRAD_COEFF = 0.9
CONFIDENCE_TARGET = 0.9
MAX_ITERATIONS = 5


# Next, we load the network, the labels and a transformer for i/o.

# In[2]:

net = adex.googlenet.load_model(CAFFE_ROOT, BATCH_SIZE)
labels = adex.googlenet.load_labels(CAFFE_ROOT) # TODO do we need those?
transformer = adex.googlenet.build_transformer(net)


# Since we only have the name of the class, we need to get the actual network label. The following function does that.

# In[3]:

def get_label_from_classname(labels, classname):
    classname = classname.strip()
    
    for index, l in enumerate(labels):
        if l[0].strip() == classname:
            return index
    else:
        raise KeyError('Class name not found')


# In[5]:

class_directories = os.listdir(DATA_ROOT)
for directory_index, class_directory in enumerate(class_directories): # Iterate over all classes / their directories
    class_files = os.listdir(DATA_ROOT + '/' + class_directory)
    random_class_representative = random.choice(class_files)
    
    # Load the representative image for this class
    image = caffe.io.load_image(DATA_ROOT + '/' + class_directory + '/' + random_class_representative)
    image = transformer.preprocess('data', image)
    image = np.expand_dims(image, 0)
    
    print('Transforming class {0} ({1}/{2}):'.format(class_directory, directory_index, len(class_directories)))
    
    # Prepare output directory
    try:
        os.mkdir(OUTPUT_ROOT + '/' + class_directory)
    except OSError:
        pass # Directory already exists
    
    history = [] # Keeps track of confidence and iterations
    for target_class_directory in class_directories: # Iterate over all classes again
        #if class_directory == target_class_directory: # Do not transform a class into itself
        #    continue
        sys.stdout.write('.')
        
        label = get_label_from_classname(labels, target_class_directory) # Get the target label
        label = np.array([label])
        
        adversarial_image, confidence, iterations = adex.core.make_adversarial(net,
                                                                               image, label, AE_GRAD_COEFF,
                                                                               CONFIDENCE_TARGET, MAX_ITERATIONS)
        
        history.append( [random_class_representative, target_class_directory, confidence, iterations] )
        np.save(OUTPUT_ROOT + '/' + class_directory + '/' + target_class_directory + '.npy', adversarial_image)
    
    # Write history to file
    with open(OUTPUT_ROOT + '/' + class_directory + '/' + 'history.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for target_class_history in history:
            writer.writerow(target_class_history)
    
    break # FIXME remove this

