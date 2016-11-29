
# coding: utf-8

# In[1]:

import math
import random
import sys
import glob
import csv
import os
import numpy as np
import caffe
import adex.core
import adex.data
import adex.googlenet
import adex.gtsrb

AE_BATCH_NAME = 'gtsrb-ae-0.05'
IS_IMAGENET = False
MIN_AE_CONFIDENCE = 0.5
MAX_ORIGINAL_IMAGES = 100 # Set to 0 for no limit

CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
DATA_ROOT = '/media/sf_Masterarbeit/data'
ORIG_ROOT = DATA_ROOT + '/GTSRB_TRAIN_PREPROCESSED'
AE_ROOT = DATA_ROOT + '/GTSRB_TRAIN_PREPROCESSED_AE_0.5'
SAVE_PATH_PREFIX = DATA_ROOT + '/spectra/{0}-minconfidence-{1}-maxorig-{2}'.format(
    AE_BATCH_NAME, MIN_AE_CONFIDENCE, MAX_ORIGINAL_IMAGES)

BATCH_SIZE = 1
net_imagenet = adex.googlenet.load_model(CAFFE_ROOT, BATCH_SIZE)
transformer_imagenet = adex.googlenet.build_transformer(net_imagenet)

net_gtsrb = adex.gtsrb.load_model('/media/sf_Masterarbeit/master-thesis/gtsrb/network_reprod_deploy.prototxt',
                                 '/media/sf_Masterarbeit/master-thesis/gtsrb/snapshots/reprod_iter_548926.caffemodel',
                                 BATCH_SIZE)
transformer_gtsrb = adex.gtsrb.build_transformer(net_gtsrb)


# In[2]:

def get_original_list(orig_root, max_original_images):
    original_list = []
    for cls_path in glob.glob(orig_root + '/*'):
        for cls_member in glob.glob(cls_path + '/*'):
            original_list.append(cls_member)
    random.shuffle(original_list)
    if max_original_images is not 0:
        original_list = original_list[:max_original_images]
    return(original_list)
    
original_list = get_original_list(ORIG_ROOT, MAX_ORIGINAL_IMAGES)
sys.stdout.write('Working with {0} original images\n'.format(len(original_list)))
sys.stdout.flush()

def get_ae_list_imagenet(ae_root, min_ae_confidence):
    ae_list = []
    for csv_path in glob.glob(ae_root + '/*.csv'):
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                row[2], row[3] = float(row[2]), int(row[3])
                ae_path = ae_root + '/' + row[0].split('.')[0] + '/' + row[1] + '.npy'
                if row[2] >= min_ae_confidence:
                    ae_list.append(ae_path)
    return ae_list

def get_ae_list_gtsrb(ae_root, min_ae_confidence):
    ae_list = []
    for csv_path in glob.glob(ae_root + '/*.csv'):
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                row[3], row[4] = float(row[3]), int(row[4])
                ae_path = ae_root + '/' + row[0] + '_' + row[1].split('.')[0] + '/' + row[2] + '.npy'
                if row[2] >= min_ae_confidence:
                    ae_list.append(ae_path)
    return ae_list

if IS_IMAGENET:
    ae_list = get_ae_list_imagenet(AE_ROOT, MIN_AE_CONFIDENCE)
else:
    ae_list = get_ae_list_gtsrb(AE_ROOT, MIN_AE_CONFIDENCE)
sys.stdout.write('Working with {0} AEs\n'.format(len(ae_list)))
sys.stdout.flush()


# In[3]:

def original_loader_imagenet(path):
    img = adex.googlenet.load_image(transformer_imagenet, path)
    img = adex.data.grayvalue_image(img)
    img = img[...,:-1,:-1] # Make dimensions odd, required for symmetry
    img /= math.sqrt(np.sum(img**2))
    return img

def original_loader_gtsrb(path):
    img = adex.gtsrb.load_image(transformer_gtsrb, path)
    img = adex.data.grayvalue_image(img)
    img = img[...,:-1,:-1] # Make dimensions odd, required for symmetry
    img /= math.sqrt(np.sum(img**2))
    return img

def ae_loader(path):
    img = np.load(path)
    img = adex.data.grayvalue_image(img)
    img = img[...,:-1,:-1] # Make dimensions odd, required for symmetry
    img /= math.sqrt(np.sum(img**2))
    return img

def compute_spectrum(image_list, image_loader):
    spectrum = None
    
    for idx, path in enumerate(image_list):
        img = image_loader(path)

        if spectrum is None:
            spectrum = adex.data.get_spectrum(img)
        else:
            spectrum += adex.data.get_spectrum(img)
        
        # A bit of progress feedback every 1000 images
        if idx % 1000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    
    spectrum /= len(image_list)
    return spectrum


# In[4]:

original_save_path = SAVE_PATH_PREFIX + '-orig.npy'
ae_save_path = SAVE_PATH_PREFIX + '-ae.npy'

def compute_if_not_exists(file_list, loader, path):
    if os.path.isfile(path):
        sys.stdout.write('Skipping original spectrum computation: {0} exists\n'.format(path))
    else:
        sys.stdout.write('Computing spectrum for {0}\n'.format(path))
        spectrum = compute_spectrum(file_list, loader)
        np.save(path, spectrum)
        sys.stdout.write('Saving spectrum to {0}\n'.format(path))
    sys.stdout.flush()

if IS_IMAGENET:
    compute_if_not_exists(original_list, original_loader_imagenet, original_save_path)
else:
    compute_if_not_exists(original_list, original_loader_gtsrb, original_save_path)
compute_if_not_exists(ae_list, ae_loader, ae_save_path)

