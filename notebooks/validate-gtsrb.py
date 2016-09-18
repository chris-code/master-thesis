
# coding: utf-8

# In[1]:

import numpy as np
import caffe
import adex.core

CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
MODEL_PATH = '/home/chrisbot/Projects/master-thesis/gtsrb/network_reprod_deploy.prototxt'
WEIGHT_PATH = '/media/sf_Masterarbeit/master-thesis/gtsrb/snapshots/reprod_iter_548926.caffemodel'
DATA_ROOT_PATH = '/media/sf_Masterarbeit/data/GTSRB_TEST_PREPROCESSED'

BATCH_SIZE = 1

# Build network and reshape it to avoid batches of size 16
net = caffe.Net(MODEL_PATH, WEIGHT_PATH, caffe.TEST)

print('In shape: {0}'.format(net.blobs['data'].data.shape))
#print('Label shape: {0}'.format(net.blobs['label'].data.shape))
print('Out shape: {0}'.format(net.blobs['prob'].data.shape))

# Build transformer
transformer = caffe.io.Transformer( {'data': net.blobs['data'].data.shape} )
transformer.set_transpose('data', (2,0,1))
#transformer.set_raw_scale('data', 255) # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0)) # the reference model has channels in BGR order instead of RGB


# In[2]:

DATA_LIMIT = None
data_count = 0
success_count = 0

with open(DATA_ROOT_PATH + '/test_images_labeled.txt') as test_list:
    for line in test_list:
        filename = ''.join(line.strip().split(' ')[:-1])
        label = int(line.strip().split(' ')[-1])
        
        image = caffe.io.load_image(filename)
        image = transformer.preprocess('data', image)
        image = np.expand_dims(image, 0)
        
        preds, probs = adex.core.predict(net, image)
        
        if(preds[0][0] == label):
            success_count += 1
        
        data_count += 1
        if DATA_LIMIT is not None and data_count == DATA_LIMIT:
            break

print('Accuracy: {0}'.format(float(success_count) / data_count))

