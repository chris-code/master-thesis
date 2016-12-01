
# coding: utf-8

# In[1]:

import glob
import csv
import os
import shutil
import adex
import adex.googlenet

CAFFE_ROOT = '/home/chrisbot/Projects/caffe'
OLD_ROOT = '/media/sf_Masterarbeit/data/ILSVRC2012_img_train_AE_10_3.0'
NEW_ROOT = '/media/sf_Masterarbeit/data/AE/ILSVRC2012_img_train_AE_10_3.0'

if not os.path.exists(NEW_ROOT):
    os.makedirs(NEW_ROOT)

labels = adex.googlenet.load_labels(CAFFE_ROOT)


# In[2]:

fieldnames = ['SourceClass', 'SourcePath', 'TargetClass', 'TargetPath', 'Confidence', 'Iterations']

for old_csv_path in glob.glob(OLD_ROOT + '/*.csv'):
    csv_data = []
    csv_filename = old_csv_path.split('/')[-1]
    
    with open(old_csv_path) as old_file:
        old_reader = csv.reader(old_file)
        
        for row in old_reader:
            source_class = row[0].split('_')[0]
            source_path = source_class + '/' + row[0]
            source_class = adex.googlenet.get_label_from_class_name(labels, source_class)
            
            csv_filename = str(source_class) + '.csv'
            
            target_class = adex.googlenet.get_label_from_class_name(labels, row[1])
            target_path = source_path[:source_path.rfind('.')] + '_' + str(target_class) + '.npy'
            confidence = float(row[2])
            iterations = int(row[3])
            new_row = [source_class, source_path, target_class, target_path, confidence, iterations]
            csv_data.append({x: y for x, y in zip(fieldnames, new_row)})
    
    with open(NEW_ROOT + '/' + csv_filename, 'w') as new_file:
        new_writer = csv.DictWriter(new_file, fieldnames=fieldnames)
        new_writer.writeheader()
        for row in csv_data:
            new_writer.writerow(row)


# In[3]:

for old_csv_path in glob.glob(OLD_ROOT + '/*.csv'):
    with open(old_csv_path) as old_file:
        old_reader = csv.reader(old_file)
        
        for row in old_reader:
            infile = row[0][:row[0].rfind('.')] + '/' + row[1] + '.npy'
            infile = OLD_ROOT + '/' + infile
            
            source_class = row[0].split('_')[0]
            source_path = source_class + '/' + row[0]
            target_class = adex.googlenet.get_label_from_class_name(labels, row[1])
            outfile = source_path[:source_path.rfind('.')] + '_' + str(target_class) + '.npy'
            outfile = NEW_ROOT + '/' + outfile
            
            try:
                os.makedirs(outfile[:outfile.rfind('/')])
            except OSError:
                pass
            shutil.copy(infile, outfile)

