import random
import os

COIL_PATH = '/media/sf_Masterarbeit/data/COIL100'
TRAIN_OUT_FILENAME = 'train_images_labeled.txt'
TEST_OUT_FILENAME = 'test_images_labeled.txt'
TEST_PORTION = 0.25

files = {}
for img_filename in os.listdir(COIL_PATH):
	if img_filename[-3:] != 'png':
		continue
	class_name = img_filename.split('_')[0][3:]
	
	img_filename = COIL_PATH + '/' + img_filename
	try:
		files[class_name].append((img_filename, class_name))
	except KeyError:
		files[class_name] = [(img_filename, class_name)]

train_files = []
test_files = []
for key in files:
	class_files = files[key]
	random.shuffle(class_files)
	pivot = int(round((1-TEST_PORTION) * len(class_files)))
	
	train_files.extend(class_files[:pivot])
	test_files.extend(class_files[pivot:])

random.shuffle(train_files)
random.shuffle(test_files)

with open(COIL_PATH + '/' + TRAIN_OUT_FILENAME, 'w') as outfile:
	for file_path, class_name in train_files:
		outfile.write('{0} {1}\n'.format(file_path, class_name))

with open(COIL_PATH + '/' + TEST_OUT_FILENAME, 'w') as outfile:
	for file_path, class_name in test_files:
		outfile.write('{0} {1}\n'.format(file_path, class_name))

print('Train files: {0}'.format(len(train_files)))
print('Test files : {0}'.format(len(test_files)))
