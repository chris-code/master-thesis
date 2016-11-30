
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
AE_GRAD_COEFF_RANGE = np.linspace(0.3, 3.0, 10)
ITERATIONS = 50

sys.stdout.write('Coeffs: {0}\n'.format(AE_GRAD_COEFF_RANGE))
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
for c in AE_GRAD_COEFF_RANGE:
    adversarial_image, _, _ = adex.core.make_adversarial(net, image, target_label, c,
                                                                               1.1, ITERATIONS)
    predictions, probabilities = adex.core.predict(net, adversarial_image)
    predictions, probabilities = predictions[0], probabilities[0]
    confidence = probabilities[target_label[0]]
    sys.stdout.write(str(confidence) + ' ')
    sys.stdout.flush()
    
    adversarial_images.append(adversarial_image)
    confidences.append(confidence)


# In[4]:

sys.stdout.write('# Coeff\tConfidence\n')
for ae_coeff, confid in zip(AE_GRAD_COEFF_RANGE, confidences):
    sys.stdout.write('{0}\t{1}\n'.format(ae_coeff, confid))
sys.stdout.flush()

