
# coding: utf-8

# In[1]:

import sys
import numpy as np
import caffe
import adex
import adex.core
import adex.googlenet

CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
BATCH_SIZE = 1
net = adex.googlenet.load_model(CAFFE_ROOT, BATCH_SIZE)
labels = adex.googlenet.load_labels(CAFFE_ROOT)
transformer = adex.googlenet.build_transformer(net)


# In[2]:

IMAGE_PATH = '/media/sf_Masterarbeit/data/example_images/panda.jpg'
TARGET_CLASS_NAME = ('n12267677', 'acorn')
#TARGET_CLASS_NAME = ('n03016953', 'dresser')
#TARGET_CLASS_NAME = ('n03837869', 'obelisk')
#TARGET_CLASS_NAME = 'n02749479' #rifle

NORM_PERCENTILE = 98
AE_GRAD_COEFF = 0.9
ITERATIONS_RANGE = [25 * i for i in range(1, 20+1)]
ITERATIONS_RANGE.insert(0, 10)
sys.stdout.write('Iterations: {0}\n'.format(ITERATIONS_RANGE))
sys.stdout.flush()

image = caffe.io.load_image(IMAGE_PATH)
image = transformer.preprocess('data', image)
image = np.expand_dims(image, 0)

target_label = adex.googlenet.get_label_from_class_name(labels, TARGET_CLASS_NAME[0])
target_label = np.array([target_label]) # Caffe-friendly format for labels
sys.stdout.write(str(labels[target_label[0]]) + '\n')
sys.stdout.flush()


# In[3]:

adversarial_images = []
confidences = []
for i in ITERATIONS_RANGE:
    adversarial_image, _, _ = adex.core.make_adversarial(net, image, target_label, AE_GRAD_COEFF,
                                                                               1.1, i)
    predictions, probabilities = adex.core.predict(net, adversarial_image)
    predictions, probabilities = predictions[0], probabilities[0]
    confidence = probabilities[target_label[0]]
    sys.stdout.write(str(confidence) + ' ')
    sys.stdout.flush()
    
    adversarial_images.append(adversarial_image)
    confidences.append(confidence)


# In[4]:

sys.stdout.write('# Iterations\tConfidence\n')
for i, c in zip(ITERATIONS_RANGE, confidences):
    sys.stdout.write('{0}\t{1}\n'.format(i, c))
sys.stdout.flush()

