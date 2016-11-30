
# coding: utf-8

# In[1]:

import random
import csv
import sys
import glob
import os
import numpy as np
import caffe
import adex.core
import adex.coil

AE_GRAD_COEFF = 0.1
CONFIDENCE_TARGET = 0.9
MAX_ITERATIONS = 500

#CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
LAYOUT_PATH = '/media/sf_Masterarbeit/master-thesis/coil-100/network_normal_deploy.prototxt'
WEIGHT_PATH = '/media/sf_Masterarbeit/master-thesis/coil-100/snapshots/normal_iter_75600.caffemodel'
DATA_ROOT = '/media/sf_Masterarbeit/data/COIL100'
OUTPUT_ROOT = '/media/sf_Masterarbeit/data/COIL100_fullres_AE_{0}'.format(AE_GRAD_COEFF)
BATCH_SIZE = 1

net = adex.coil.load_model(LAYOUT_PATH, WEIGHT_PATH, BATCH_SIZE)
shape = list(net.blobs['data'].data.shape)
shape[0] = BATCH_SIZE
net.blobs['data'].reshape(*shape)
net.blobs['prob'].reshape(BATCH_SIZE, )
net.reshape()
transformer = adex.coil.build_transformer(net)


# In[2]:

def get_image_dict(path):
    image_dict = {}
    for coil_img_path in glob.glob(path + '/*.png'):
        img_class, _, instance_id = coil_img_path.split('/')[-1].split('_')
        img_class = int(img_class[3:]) - 1
        instance_id = int(instance_id[:-4])
        
        try:
            image_dict[img_class].append((coil_img_path, instance_id))
        except KeyError:
            image_dict[img_class] = [(coil_img_path, instance_id)]
    return image_dict

image_dict = get_image_dict(DATA_ROOT)


# In[3]:

try:
    os.mkdir(OUTPUT_ROOT)
except OSError:
    pass # Directory already exists

for orig_class in image_dict.keys():
    class_representative_path, class_representative_instance_id = random.choice(image_dict[orig_class])
    
    # Prepare output directory
    output_dir = OUTPUT_ROOT + '/' + str(orig_class) + '_' + str(class_representative_instance_id)
    try:
        os.mkdir(output_dir)
    except OSError:
        pass # Directory already exists
    
    history = []
    for target_class in image_dict.keys():
        image = adex.coil.load_image(transformer, class_representative_path)
        target_label = np.array([target_class])
        
        adversarial_image, confidence, iterations = adex.core.make_adversarial(net, image, target_label, AE_GRAD_COEFF,
                                                                               CONFIDENCE_TARGET, MAX_ITERATIONS)
        
        out_path = output_dir + '/' + str(target_class) + '.npy'
        np.save(out_path, adversarial_image)
        
        # Add data to history
        history.append( [orig_class, class_representative_path.split('/')[-1], target_class, confidence, iterations] )
    
    with open(OUTPUT_ROOT + '/' + str(orig_class) + '_' + 'history.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for target_class_history in history:
            writer.writerow(target_class_history)
    
    # A bit of progress feedback
    sys.stdout.write('.')
    sys.stdout.flush()

