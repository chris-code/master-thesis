
# coding: utf-8

# In[ ]:

import random
import sys
import os
import csv
import numpy as np
import caffe
import adex
import adex.core
import adex.coil
import adex.gtsrb
import adex.googlenet

PROGRESS_COUNT = 3
ITERATIONS = 5
AE_GRAD_COEFF = 0.9
DATASET_NAME = 'imagenet'
CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
IMAGE_LIST_PATH = '/media/sf_Masterarbeit/data/ILSVRC2012_img_train/images_labeled.txt'
OUTPUT_PREFIX = '/media/sf_Masterarbeit/data/AE_PROGRESS/{0}_{1}c_{2}iter_{3}samples'.format(
    DATASET_NAME, AE_GRAD_COEFF, ITERATIONS, PROGRESS_COUNT)

BATCH_SIZE = 1
net = adex.googlenet.load_model(CAFFE_ROOT + '/models/bvlc_googlenet/deploy.prototxt',
                                 CAFFE_ROOT + '/models/bvlc_googlenet/bvlc_googlenet.caffemodel',
                                 BATCH_SIZE)
transformer = adex.googlenet.build_transformer(net)

#net = adex.gtsrb.load_model('/media/sf_Masterarbeit/master-thesis/gtsrb/network_reprod_deploy.prototxt',
#                            '/media/sf_Masterarbeit/master-thesis/gtsrb/snapshots/reprod_iter_548926.caffemodel',
#                            BATCH_SIZE)
#transformer = adex.gtsrb.build_transformer(net)

#net = adex.coil.load_model('/media/sf_Masterarbeit/master-thesis/coil-100/network_normal_deploy.prototxt',
#                           '/media/sf_Masterarbeit/master-thesis/coil-100/snapshots/normal_iter_75600.caffemodel',
#                           BATCH_SIZE)
#transformer = adex.coil.build_transformer(net)

sys.stdout.write('Generating {0} AEs with c={1} for {2} iterations.\n'.format(PROGRESS_COUNT, AE_GRAD_COEFF, ITERATIONS))
sys.stdout.flush()


# In[ ]:

def get_image_list(path):
    image_list = []
    
    with open(path) as image_list_file:
        for line in image_list_file:
            line = line.strip()
            img_class = line.split()[-1]
            img_path = line[:-len(img_class)].strip()
            img_class = int(img_class.strip())
            
            image_list.append((img_path, img_class))
            
    return image_list

image_list = get_image_list(IMAGE_LIST_PATH)

# Determine valid target classes
classes = set()
for img_path, img_class in image_list:
    classes.add(img_class)
classes = list(classes)
sys.stdout.write('Found {0} classes\n'.format(len(classes)))

random.shuffle(image_list)
image_list = image_list[:PROGRESS_COUNT]
sys.stdout.write('Using {0} images\n'.format(len(image_list)))
sys.stdout.flush()


# In[ ]:

def make_ae(net, data, desired_labels, ae_grad_coeff, iterations):
    progress = np.zeros(shape=(iterations))
    ae_data = data.copy()
    
    for i in range(iterations):
        ae_data, confidence, _ = adex.core.make_adversarial(net, ae_data, desired_labels, ae_grad_coeff / iterations,
                                                            100, 1)
        progress[i] = confidence
    
    return ae_data, progress

csv_data = []
progress_record = np.empty(shape=(PROGRESS_COUNT, ITERATIONS))
for idx, line in enumerate(image_list):
    img_path, source_class = line
    
    image = caffe.io.load_image(img_path)
    image = transformer.preprocess('data', image)
    image = np.expand_dims(image, 0)
    
    target_class = random.choice(classes)
    
    _, progress = make_ae(net, image, np.array([target_class]), AE_GRAD_COEFF, ITERATIONS)
    progress_record[idx, :] = progress
    
    csv_data.append([source_class, img_path[IMAGE_LIST_PATH.rfind('/')+1:], target_class, '', progress[-1], ITERATIONS])
    
    sys.stdout.write('.')
    sys.stdout.flush()


# In[ ]:

try:
    os.makedirs(OUTPUT_PREFIX[:OUTPUT_PREFIX.rfind('/')])
except OSError:
    pass

np.save(OUTPUT_PREFIX + '_data.npy', progress_record)

fieldnames = ['SourceClass', 'SourcePath', 'TargetClass', 'TargetPath', 'Confidence', 'Iterations']

csv_data = [{x: y for x, y in zip(fieldnames, row)} for row in csv_data]
with open(OUTPUT_PREFIX + '_meta.csv', 'w') as outfile:
    csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    csv_writer.writeheader()
    for row in csv_data:
        csv_writer.writerow(row)

sys.stdout.write('done.\n')
sys.stdout.flush()

