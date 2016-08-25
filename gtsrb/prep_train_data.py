import os
import csv
import PIL.Image
import PIL.ImageOps

IN_PATH = '/media/sf_Masterarbeit/data/GTSRB_TRAIN' # Where the original GTSRB Train data resides
OUT_PATH = '/media/sf_Masterarbeit/data/GTSRB_TRAIN_PREPROCESSED' # Where to put the cropped and rescaled images
NEW_RES = (48, 48)

try:
	os.mkdir(OUT_PATH) # Create output directory
except OSError: # Directory exists
	pass

# Used to generate a file with each line containing a filename and the class of the corresponding image.
file_list = []

for d in os.listdir(IN_PATH):
	if not os.path.isdir(IN_PATH + '/' + d):
		continue
	
	# Create output directory
	try:
		os.mkdir(OUT_PATH + '/' + d)
	except OSError: # Directory exists
		pass
	
	print('Working on class {0}'.format(d))
	
	csvpath = IN_PATH + '/' + d + '/GT-' + d + '.csv'
	with open(csvpath) as csvfile:
		for row in csv.DictReader(csvfile, delimiter=';'):
			in_file = IN_PATH + '/' + d + '/' + row['Filename']
			out_file = OUT_PATH + '/' + d + '/' + row['Filename']
			file_list.append((out_file, int(row['ClassId'])))
			
			image = PIL.Image.open(in_file)
			
			# Crop image
			roi_borders = (int(row['Roi.X1']), int(row['Roi.Y1']), int(row['Roi.X2']), int(row['Roi.Y2']))
			image = image.crop(roi_borders)
			
			# Enhance contrast
			image = PIL.ImageOps.autocontrast(image)
			
			# Resize
			image = image.resize(NEW_RES)
			
			image.save(out_file)

file_list_path = OUT_PATH + '/' + 'train_images_labeled.txt'
with open(file_list_path, 'w') as outfile:
	for file_path, class_id in file_list:
		outfile.write(file_path + ' ' + str(class_id) + '\n')
