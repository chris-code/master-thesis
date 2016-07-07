
# coding: utf-8

# In[1]:

import sys
import random
import os
import csv
import glob
import numpy as np
import caffe
import adex.core
import adex.googlenet

CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
DATA_ROOT = '/media/sf_Masterarbeit/data/ILSVRC2012_img_train'
AE_ROOT = '/media/sf_Masterarbeit/data/ILSVRC2012_img_train_AE_50'
OUT_ROOT = '/media/sf_Masterarbeit/data/ILSVRC2012_img_train_AE_50_progress_500_2.0'

AE_GRAD_COEFF = 0.9
ITERATIONS = 10
BATCH_SIZE = 1

SAMPLE_COUNT = 2
SUCCESS_THRESHOLD = 0.5
FAILURE_THRESHOLD = 0.1

net = adex.googlenet.load_model(CAFFE_ROOT, BATCH_SIZE)
labels = adex.googlenet.load_labels(CAFFE_ROOT)
transformer = adex.googlenet.build_transformer(net)


# We are gonna prepare the data by reading all .csv files in the *AE_ROOT* folder. They will be split into a success and a failure set.

# In[2]:

csv_data = []
for csvpath in glob.glob(AE_ROOT + '/*history.csv'):
    with open(csvpath) as csv_file_desc:
        reader = csv.reader(csv_file_desc)
        for row in reader:
            csv_data.append(row)
csv_data = [[fname, target_cl, float(certainty), int(num_iter)] for fname, target_cl, certainty, num_iter in csv_data]
csv_data = [row for row in csv_data if row[0][:9] != row[1]] # Exclude if original class = target class
random.shuffle(csv_data) # bring data into random order

csv_successes, csv_failures = [], []
for row in csv_data:
    if row[2] >= SUCCESS_THRESHOLD:
        csv_successes.append(row)
    elif row[2] < FAILURE_THRESHOLD:
        csv_failures.append(row)

csv_successes = csv_successes[:SAMPLE_COUNT] # Ensure a maximum of SAMPLE_COUNT images each
csv_failures = csv_failures[:SAMPLE_COUNT]

print('Working with {0} successful and {1} failed transformations'.format(len(csv_successes), len(csv_failures)))


# *make_ae* is a wrapper function for *adex.core.make_adversarial* that performs a fixed number of steps, regardless of confidence, and returns the confidence progress over time. *record_progress* records these progesses for all images in *csv_list* in a 2-dimensional numpy array.

# In[3]:

def make_ae(net, data, desired_labels, ae_grad_coeff, iterations):
    progress = np.zeros(shape=(iterations))
    ae_data = data.copy()
    
    for i in range(iterations):
        ae_data, confidence, _ = adex.core.make_adversarial(net, ae_data, desired_labels, ae_grad_coeff / iterations,
                                                            100, 1)
        progress[i] = confidence
    
    return ae_data, progress

def record_progress(csv_list):
    progress_record = np.zeros(shape=(ITERATIONS))
    
    for orig_filename, target_class, _, _ in csv_list:
        sys.stdout.write('.')
        sys.stdout.flush()
        
        orig_class = orig_filename[:9]

        image = caffe.io.load_image(DATA_ROOT + '/' + orig_class + '/' + orig_filename)
        image = transformer.preprocess('data', image)
        image = np.expand_dims(image, 0)

        label = adex.googlenet.get_label_from_class_name(labels, target_class)
        label = np.array([label])

        ae_data, progress = make_ae(net, image, label, AE_GRAD_COEFF, ITERATIONS)
        progress_record = np.vstack([progress_record, progress])
        
        # Save AE data to disk
        os.mkdir(OUT_ROOT + '/' + orig_filename[:-5])
        np.save(OUT_ROOT + '/' + orig_filename[:-5] + '/' + target_class + '.npy', ae_data)
    
    print('')
    return progress_record[1:] # Skip the first because it is all zeros (initialization for np.vstack)

success_progress = record_progress(csv_successes)
failure_progress = record_progress(csv_failures)


# Finally, write the data to disk.

# In[4]:

with open(OUT_ROOT + '/' + 'success_progress.csv', 'w') as file_desc:
    writer = csv.writer(file_desc)
    for row in csv_successes:
        writer.writerow(row)
with open(OUT_ROOT + '/' + 'failure_progress.csv', 'w') as file_desc:
    writer = csv.writer(file_desc)
    for row in csv_failures:
        writer.writerow(row)
np.save(OUT_ROOT + '/' + 'success_progress.npy', success_progress)
np.save(OUT_ROOT + '/' + 'failure_progress.npy', failure_progress)

