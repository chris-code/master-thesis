
# coding: utf-8

# In[2]:

import random
import csv
import sys
import os
import numpy as np
import caffe
import adex.core
import adex.gtsrb

#CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
LAYOUT_PATH = '/media/sf_Masterarbeit/master-thesis/gtsrb/network_reprod_deploy.prototxt'
WEIGHT_PATH = '/media/sf_Masterarbeit/master-thesis/gtsrb/snapshots/reprod_iter_548926.caffemodel'
DATA_ROOT = '/media/sf_Masterarbeit/data/GTSRB_TRAIN_PREPROCESSED'
OUTPUT_ROOT = '/media/sf_Masterarbeit/data/GTSRB_TRAIN_PREPROCESSED_AE_0.037'
BATCH_SIZE = 1

AE_GRAD_COEFF = 0.037
CONFIDENCE_TARGET = 0.9
MAX_ITERATIONS = 500

net = adex.gtsrb.load_model(LAYOUT_PATH, WEIGHT_PATH, BATCH_SIZE)
shape = list(net.blobs['data'].data.shape)
shape[0] = BATCH_SIZE
net.blobs['data'].reshape(*shape)
net.blobs['prob'].reshape(BATCH_SIZE, )
net.reshape()
transformer = adex.gtsrb.build_transformer(net)


# In[ ]:

class_directories = os.listdir(DATA_ROOT) # The list of classes (conincide with their directory names)
class_directories = [cs for cs in class_directories if os.path.isdir(DATA_ROOT + '/' + cs)]
random.shuffle(class_directories)

for class_dir in class_directories:
    random_class_representative = random.choice(os.listdir(DATA_ROOT + '/' + class_dir))
    
    # Prepare output directory
    output_dir = OUTPUT_ROOT + '/' + class_dir + '_' + random_class_representative[:-4]
    try:
        os.mkdir(output_dir)
    except OSError:
        pass # Directory already exists
    
    history = []
    for target_class_dir in class_directories:
        #Generate AEs
        infile = DATA_ROOT + '/' + class_dir + '/' + random_class_representative
        #confidence, iterations = make_ae_for_paths(infile, outfile, net, labels, target_class_name,
        #                                           AE_GRAD_COEFF, CONFIDENCE_TARGET, MAX_ITERATIONS)
        
        # Generate an AE
        image = adex.gtsrb.load_image(transformer, infile)
        target_label = np.array([int(target_class_dir)])
        adversarial_image, confidence, iterations = adex.core.make_adversarial(net, image, target_label, AE_GRAD_COEFF,
                                                                               CONFIDENCE_TARGET, MAX_ITERATIONS)
        
        # Save AE to disk
        outfile = output_dir + '/' + target_class_dir + '.npy'
        np.save(outfile, adversarial_image)
        
        # Add data to history
        history.append( [class_dir, random_class_representative, target_class_dir, confidence, iterations] )
    
    # Write history to file
    with open(OUTPUT_ROOT + '/' + class_dir + '_' + 'history.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for target_class_history in history:
            writer.writerow(target_class_history)
    
    # A bit of progress feedback
    sys.stdout.write('.')
    sys.stdout.flush()

