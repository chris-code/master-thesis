
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
MAX_ITERATIONS = 10
ORIG_CLASS_LIMIT = 10
TARGET_CLASS_LIMIT = 10

#CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
LAYOUT_PATH = '/media/sf_Masterarbeit/master-thesis/coil-100/network_small_deploy.prototxt'
WEIGHT_PATH = '/media/sf_Masterarbeit/master-thesis/coil-100/snapshots/small_iter_75600.caffemodel'
#DATA_ROOT = '/media/sf_Masterarbeit/data/COIL100'
ORIGINAL_LIST_PATH = '/media/sf_Masterarbeit/data/COIL100/train_images_labeled.txt'
OUTPUT_ROOT = '/media/sf_Masterarbeit/data/COIL100_halfres_AE_{0}'.format(AE_GRAD_COEFF)
BATCH_SIZE = 1

net = adex.coil.load_model(LAYOUT_PATH, WEIGHT_PATH, BATCH_SIZE)
shape = list(net.blobs['data'].data.shape)
shape[0] = BATCH_SIZE
net.blobs['data'].reshape(*shape)
net.blobs['prob'].reshape(BATCH_SIZE, )
net.reshape()
transformer = adex.coil.build_transformer(net)

sys.stdout.write('AE_GRAD_COEFF = {0}\nMAX_ITERATIONS = {1}\n'.format(AE_GRAD_COEFF, MAX_ITERATIONS))
sys.stdout.flush()


# In[2]:

# Returns a dict with the class id as key and (full_path, class, instance_name) as value
def get_image_dict(path):
    image_dict = {}
    with open(path) as image_list_file:
        for line in image_list_file:
            line = line.strip()
            img_class = line.split()[-1].strip()
            img_path = line[:-len(img_class)].strip()
            img_class = int(img_class)# - 1 # TODO -1 needed?
            img_instance = img_path[:-len(img_path.split('.')[-1]) - 1] # cut extension irrespective of its length
            img_instance = img_instance.split('/')[-1]
            
            try:
                image_dict[img_class].append((img_path, img_class, img_instance))
            except KeyError:
                image_dict[img_class] = [(img_path, img_class, img_instance)]
    return image_dict
image_dict = get_image_dict(ORIGINAL_LIST_PATH)

#image_dict = get_image_dict(DATA_ROOT)
sys.stdout.write('Found {0} images in {1} classes\n'.format(
        sum(map(lambda x: len(x), image_dict.values())), len(image_dict.keys())))

orig_class_list = image_dict.keys()[:]
target_class_list = image_dict.keys()[:]
random.shuffle(orig_class_list)
random.shuffle(target_class_list)
orig_class_list = orig_class_list[:ORIG_CLASS_LIMIT]
target_class_list = target_class_list[:TARGET_CLASS_LIMIT]
sys.stdout.write('Using {0} original classes\n'.format(len(orig_class_list)))
sys.stdout.write('Using {0} target classes\n'.format(len(target_class_list)))
sys.stdout.flush()


# In[3]:

try:
    os.mkdir(OUTPUT_ROOT)
except OSError:
    pass # Directory already exists

for orig_class in orig_class_list:
    class_representative_path, _, class_representative_instance_name = random.choice(image_dict[orig_class])
    
    # Prepare output directory
    output_dir = OUTPUT_ROOT + '/' + str(orig_class) + '_' + str(class_representative_instance_name)
    try:
        os.mkdir(output_dir)
    except OSError:
        pass # Directory already exists
    
    history = []
    for target_class in target_class_list:
        image = adex.coil.load_image(transformer, class_representative_path)
        target_label = np.array([target_class])
        
        adversarial_image, confidence, iterations = adex.core.make_adversarial(net, image, target_label, AE_GRAD_COEFF,
                                                                               CONFIDENCE_TARGET, MAX_ITERATIONS)
        
        out_path = output_dir + '/' + str(target_class) + '.npy'
        np.save(out_path, adversarial_image)
        
        # Add data to history
        history.append( [orig_class, class_representative_path.split('/')[-1], target_class, confidence, iterations] )
    
    with open(OUTPUT_ROOT + '/' + str(orig_class) + '_' + str(class_representative_instance_name) + '_' + 'history.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for target_class_history in history:
            writer.writerow(target_class_history)
    
    # A bit of progress feedback
    sys.stdout.write('.')
    sys.stdout.flush()

