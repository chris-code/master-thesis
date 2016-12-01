
# coding: utf-8

# In[1]:

import glob
import csv
import os
import shutil

OLD_ROOT = '/media/sf_Masterarbeit/data/GTSRB_TRAIN_PREPROCESSED_AE_0.1'
NEW_ROOT = '/media/sf_Masterarbeit/data/AE/GTSRB_TRAIN_PREPROCESSED_AE_0.1'

if not os.path.exists(NEW_ROOT):
    os.makedirs(NEW_ROOT)


# In[2]:

fieldnames = ['SourceClass', 'SourcePath', 'TargetClass', 'TargetPath', 'Confidence', 'Iterations']

for old_csv_path in glob.glob(OLD_ROOT + '/*.csv'):
    csv_data = []
    csv_filename = str(int(old_csv_path.split('/')[-1].split('_')[0])) + '.csv'
    
    with open(old_csv_path) as old_file:
        old_reader = csv.reader(old_file)
        
        for row in old_reader:
            source_class = int(row[0])
            source_path = row[0] + '/' + row[1]
            target_class = int(row[2])
            target_path = source_path[:source_path.rfind('.')] + '_' + str(target_class) + '.npy'
            confidence = float(row[3])
            iterations = int(row[4])
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
            infile = row[0] + '_' + row[1][:row[1].rfind('.')] + '/' + row[2] + '.npy'
            infile = OLD_ROOT + '/' + infile
            
            source_path = row[0] + '/' + row[1]
            target_class = int(row[2])
            outfile = source_path[:source_path.rfind('.')] + '_' + str(target_class) + '.npy'
            outfile = NEW_ROOT + '/' + outfile
            
            try:
                os.makedirs(outfile[:outfile.rfind('/')])
            except OSError:
                pass
            shutil.copy(infile, outfile)

