import numpy as np
import pandas as pd
import os
import random 
from shutil import copyfile
import pydicom as dicom
import cv2
from PIL import Image


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)




if os.path.exists('covid-chestxray-dataset') is False:
	os.system("git clone https://github.com/ieee8023/covid-chestxray-dataset.git")



if os.path.exists('data') is False:
	os.mkdir('data')
	os.mkdir('data/test')
	os.mkdir('data/train')	
	os.mkdir('data/labeled')

for the_file in os.listdir('data/test'):
    file_path = os.path.join('data/test', the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)


for the_file in os.listdir('data/train'):
    file_path = os.path.join('data/train', the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

for the_file in os.listdir('data/labeled'):
    file_path = os.path.join('data/labeled', the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)





# set parameters here
savepath = 'data'
seed = 0
np.random.seed(seed) # Reset the seed so all runs are the same.
random.seed(seed)
MAXVAL = 255  # Range [0 255]

# path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
imgpath = 'covid-chestxray-dataset/images' 
csvpath = 'covid-chestxray-dataset/metadata.csv'

if os.path.exists('rsna-pneumonia-detection-challenge') is False:
	print("download the folder pneumonia detection challenge from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge")
	quit()


# path to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
kaggle_datapath = 'rsna-pneumonia-detection-challenge'
kaggle_csvname = 'stage_2_detailed_class_info.csv' # get all the normal from here
kaggle_csvname2 = 'stage_2_train_labels.csv' # get all the 1s from here since 1 indicate pneumonia
kaggle_imgpath = 'stage_2_train_images'

# parameters for COVIDx dataset
train = []
test = []
train_yolo = []
test_yolo = []
test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

mapping = dict()
mapping['COVID-19'] = 'COVID-19'
mapping['SARS'] = 'pneumonia'
mapping['MERS'] = 'pneumonia'
mapping['Streptococcus'] = 'pneumonia'
mapping['Normal'] = 'normal'
mapping['Lung Opacity'] = 'pneumonia'
mapping['1'] = 'pneumonia'

# train/test split
split = 0.1

# adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py#L814
csv = pd.read_csv(csvpath, nrows=None)
idx_pa = csv["view"] == "PA"  # Keep only the PA view
csv = csv[idx_pa]

pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus"]
pathologies = ["Pneumonia","Viral Pneumonia", "Bacterial Pneumonia", "No Finding"] + pneumonias
pathologies = sorted(pathologies)

# get non-COVID19 viral, bacteria, and COVID-19 infections from covid-chestxray-dataset
# stored as patient id, image filename and label
filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
for index, row in csv.iterrows():
    f = row['finding']
    if f in mapping:
        count[mapping[f]] += 1
        entry = [int(row["patientid"]), row["filename"], mapping[f]]
        filename_label[mapping[f]].append(entry)

print('Data distribution from covid-chestxray-dataset:')
print(count)

# add covid-chestxray-dataset into COVIDx dataset
# since covid-chestxray-dataset doesn't have test dataset
# split into train/test by patientid
# for COVIDx:
# patient 8 is used as non-COVID19 viral test
# patient 31 is used as bacterial test
# patients 19, 20, 36, 42, 86 are used as COVID-19 viral test

for key in filename_label.keys():
    arr = np.array(filename_label[key])
    if arr.size == 0:
        continue
    print('Key: ', key)
    #print('Test patients: ', test_patients)
    # 20 percent is test 80 percent is training
    split_count = 0
    # go through all the patients
    for patient in arr:
        if split_count %5 == 1:
            copyfile(os.path.join(imgpath, patient[1]), os.path.join(savepath, 'test', patient[1]))
            test.append(patient)
            test_count[patient[2]] += 1
        else:
            copyfile(os.path.join(imgpath, patient[1]), os.path.join(savepath, 'train', patient[1]))
            train.append(patient)
            train_count[patient[2]] += 1
        split_count = split_count +1
print('test count: ', test_count)
print('train count: ', train_count)


print('starting to update the list...')
# add normal and rest of pneumonia cases from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
csv_normal = pd.read_csv(os.path.join(kaggle_datapath, kaggle_csvname), nrows=None)
csv_pneu = pd.read_csv(os.path.join(kaggle_datapath, kaggle_csvname2), nrows=None)
patients = {'normal': [], 'pneumonia': []}
labeled = {'id': [], 'x': [], 'y': [], 'width': [], 'height': []}

for index, row in csv_normal.iterrows():
    if row['class'] == 'Normal':
        patients['normal'].append(row['patientId'])

for index, row in csv_pneu.iterrows():
    if int(row['Target']) == 1:
        patients['pneumonia'].append(row['patientId'])
	labeled['id'].append(row['patientId'])
	labeled['x'].append(row['x'])
	labeled['y'].append(row['y'])
	labeled['width'].append(row['width'])
	labeled['height'].append(row['height'])
    else :
	patients['normal'].append(row['patientId'])
print(labeled['id'])



#for key in labeled.keys():
#    arr = np.array(labeled[key])
#    if arr.size == 0:
#        continue
#    count = 0
    # np.save('rsna_test_patients_{}.npy'.format(key), np.array(test_patients))
#    for patient in arr:
#	print arr

print('starting to update the items...')

for key in patients.keys():
    arr = np.array(patients[key])  
    if arr.size == 0:
        continue

    # split by patients 
    # num_diff_patients = len(np.unique(arr))
    # num_test = max(1, round(split*num_diff_patients))
    #test_patients = np.load('rsna_test_patients_{}.npy'.format(key)) # random.sample(list(arr), num_test)
    count = 0
    # np.save('rsna_test_patients_{}.npy'.format(key), np.array(test_patients))
    for patient in arr:

        ds = dicom.dcmread(os.path.join(kaggle_datapath, kaggle_imgpath, patient + '.dcm'))
        pixel_array_numpy = ds.pixel_array
        imgname = patient + '.png'
	textname = patient + '.txt'
        if count%5 == 0:
            cv2.imwrite(os.path.join(savepath, 'test', imgname), pixel_array_numpy)
	    txt_outfile = open(savepath+'/test/'+textname, "w")
	    txt_outfile.close()
            test.append([patient, imgname, key])
            test_count[key] += 1
        else:
            cv2.imwrite(os.path.join(savepath, 'train', imgname), pixel_array_numpy)
	    txt_outfile = open(savepath+'/train/'+textname, "w")
	    txt_outfile.close()
            train.append([patient, imgname, key])
            train_count[key] += 1
        count = count +1
        print count

        

count = 0
TxtPath = "Txtfiles"
for name in labeled['id']:
	txtoutfilename = str(name)+".txt"
	imgoutfilename = str(name)+".png"
 	exists = os.path.isfile(savepath+'/test/'+imgoutfilename)
	if exists is True: 
		ImagesPath = savepath+'/test'
		SavePath = str(os.getcwd()) +'/'+ImagesPath+'/'+imgoutfilename
		if SavePath not in test_yolo: 
			test_yolo.append(SavePath)
	#exists = os.path.isfile(savepath+'/train/'+imgoutfilename)	
	#if exists is True:
        else:
		ImagesPath = savepath+'/train'
                SavePath = str(os.getcwd()) +'/'+ImagesPath+'/'+imgoutfilename
		if SavePath not in train_yolo:
			train_yolo.append(SavePath)
	#else:
	#	continue
	exists = os.path.isfile(ImagesPath+'/'+txtoutfilename)
	if exists is True: 
		 txt_outread = open(ImagesPath+'/'+txtoutfilename, "r")
		 lines_r = txt_outread.read().split("\r\n")
		 txt_outread.close()
		 txt_outfile = open(ImagesPath+'/'+txtoutfilename, "w")
		 for prevval in lines_r:
		       txt_outfile.write(prevval)
	else:
		 txt_outfile = open(ImagesPath+'/'+txtoutfilename, "w")

       

	xmin = labeled['x'][count]
	xmax = labeled['x'][count] +labeled['width'][count]
	ymin = labeled['y'][count]
	ymax = labeled['y'][count] +labeled['height'][count]
	width = labeled['width'][count]
	height = labeled['height'][count]

	exists = os.path.isfile('data/labeled/'+imgoutfilename)
	if exists is True:
		imglabel = cv2.imread('data/labeled/'+imgoutfilename)       	
        else :
		imglabel = cv2.imread(ImagesPath+'/'+imgoutfilename)
	cv2.rectangle(imglabel,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,255,255),1)		
	cv2.imwrite('data/labeled/'+imgoutfilename,imglabel)		
	im=Image.open(ImagesPath+'/'+imgoutfilename)
        w= int(im.size[0])
        h= int(im.size[1])
        print(w, h)
        b = (float(xmin), float(xmax), float(ymin), float(ymax))
        bb = convert((w,h), b)
        print(bb)
        txt_outfile.write(str('0') + " " + " ".join([str(a) for a in bb]) + '\n')
        txt_outfile.close();
	count = count +1

print('test count: ', test_count)
print('train count: ', train_count)

# final stats
print('Final stats')
print('Train count: ', train_count)
print('Test count: ', test_count)
print('Total length of train: ', len(train))
print('Total length of test: ', len(test))


# export to train and test csv
# format as patientid, filename, label, separated by a space
train_file = open("train.txt","a") 
for sample in train:
    info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    train_file.write(info)

train_file.close()

test_file = open("test.txt", "a")
for sample in test:
    info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    test_file.write(info)

test_file.close()


train_file = open("train_yolo.txt","a") 
for sample in train_yolo:
    info = str(sample) + '\n'
    print info
    train_file.write(info)

train_file.close()

test_file = open("test_yolo.txt", "a")
for sample in test_yolo:
    info = str(sample)+ '\n'
    print info
    test_file.write(info)

test_file.close()
