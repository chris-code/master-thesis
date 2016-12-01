import glob
import os
import adex
import adex.googlenet

CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
IMAGENET_PATH = '/media/sf_Masterarbeit/data/ILSVRC2012_img_train/'
OUTPUT_PATH = '/media/sf_Masterarbeit/data/ILSVRC2012_img_train/images_labeled.txt'

imagenet_labels = adex.googlenet.load_labels(CAFFE_ROOT)

image_list = []
for directory in glob.glob(IMAGENET_PATH + '/*'):
	if not os.path.isdir(directory):
		continue
	
	imagenet_class = directory.split('/')[-1]
	class_id = adex.googlenet.get_label_from_class_name(imagenet_labels, imagenet_class)

	for image_path in glob.glob(directory + '/*'):
		image_list.append((image_path, class_id))

with open(OUTPUT_PATH, 'w') as outfile:
	for image_path, class_id in image_list:
		outfile.write('{0} {1}\n'.format(image_path, class_id))
