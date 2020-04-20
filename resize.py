#   Bayley King
#   SOFM MNIST Application 
#   Python 3.7.3
#   April 6 2020

####### Decerations #######
from PIL import Image  
import cv2
import os
import csv
###########################

width = 100
height = 100

for subdir, dirs, files in os.walk('./DataSets/covid/xray_dataset_covid19/train/NORMAL/'):
    for file in files:
        #print(file)
        new = './DataSets/covid/xray_dataset_covid19/train/NORMAL/' + file
        img = cv2.imread(new, cv2.IMREAD_UNCHANGED)
        
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        new = './DataSets/covid/resized/train/NORMAL/' + file
        cv2.imwrite(new,resized)

for subdir, dirs, files in os.walk('./DataSets/covid/xray_dataset_covid19/test/NORMAL/'):
    for file in files:
        #print(file)
        new = './DataSets/covid/xray_dataset_covid19/test/NORMAL/' + file
        img = cv2.imread(new, cv2.IMREAD_UNCHANGED)
        
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        new = './DataSets/covid/resized/test/NORMAL/' + file
        cv2.imwrite(new,resized)

for subdir, dirs, files in os.walk('./DataSets/covid/xray_dataset_covid19/train/PNEUMONIA/'):
    for file in files:
        #print(file)
        new = './DataSets/covid/xray_dataset_covid19/train/PNEUMONIA/' + file
        img = cv2.imread(new, cv2.IMREAD_UNCHANGED)
        
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        new = './DataSets/covid/resized/train/PNEUMONIA/' + file
        cv2.imwrite(new,resized)

for subdir, dirs, files in os.walk('./DataSets/covid/xray_dataset_covid19/test/PNEUMONIA/'):
    for file in files:
        #print(file)
        new = './DataSets/covid/xray_dataset_covid19/test/PNEUMONIA/' + file
        img = cv2.imread(new, cv2.IMREAD_UNCHANGED)

        dim = (width, height)
        
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        new = './DataSets/covid/resized/test/PNEUMONIA/' + file
        cv2.imwrite(new,resized)
'''
for subdir, dirs, files in os.walk('./DataSets/covid/xray_dataset_covid19/train/NORMAL/'):
    for file in files:
        #print(file)
        new = './DataSets/covid/resized/train/PNEUMONIA/' + file
        img = Image.open(new)
        img2 = img.convert('LA').convert('RGB')
        img2.save(new)
'''
print('resized')

test = []
train = []
for subdir, dirs, files in os.walk('./DataSets/covid/resized/train/NORMAL/'):
    normal = []
    for file in files:

        new = './DataSets/covid/resized/train/NORMAL/' + file
        img = Image.open(new).convert('L')  # convert image to 8-bit grayscale
        WIDTH, HEIGHT = img.size

        data = list(img.getdata()) # convert image data to a list of integers
        # convert that to 2D list (list of lists of integers)
        #data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

        # At this point the image's pixels are all in memory and can be accessed
        # individually using data[row][col].
        data.append('0')
        train.append(data)

for subdir, dirs, files in os.walk('./DataSets/covid/resized/train/PNEUMONIA/'):
    normal = []
    for file in files:

        new = './DataSets/covid/resized/train/PNEUMONIA/' + file
        img = Image.open(new).convert('L')  # convert image to 8-bit grayscale
        WIDTH, HEIGHT = img.size

        data = list(img.getdata()) # convert image data to a list of integers
        # convert that to 2D list (list of lists of integers)
        #data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

        # At this point the image's pixels are all in memory and can be accessed
        # individually using data[row][col].
        data.append('1')
        train.append(data)

for subdir, dirs, files in os.walk('./DataSets/covid/resized/test/NORMAL/'):
    normal = []
    for file in files:

        new = './DataSets/covid/resized/test/NORMAL/' + file
        img = Image.open(new).convert('L')  # convert image to 8-bit grayscale
        WIDTH, HEIGHT = img.size

        data = list(img.getdata()) # convert image data to a list of integers
        # convert that to 2D list (list of lists of integers)
        #data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

        # At this point the image's pixels are all in memory and can be accessed
        # individually using data[row][col].
        data.append('0')
        test.append(data)

for subdir, dirs, files in os.walk('./DataSets/covid/resized/test/PNEUMONIA/'):
    normal = []
    for file in files:

        new = './DataSets/covid/resized/test/PNEUMONIA/' + file
        img = Image.open(new).convert('L')  # convert image to 8-bit grayscale
        WIDTH, HEIGHT = img.size

        data = list(img.getdata()) # convert image data to a list of integers
        # convert that to 2D list (list of lists of integers)
        #data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

        # At this point the image's pixels are all in memory and can be accessed
        # individually using data[row][col].
        data.append('1')
        test.append(data)

#print(len(test))
#print(len(train))
print('generated yo')

with open("DataSets/covid/test.csv", 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',',quoting=csv.QUOTE_MINIMAL)
    for point in test:
        csv_writer.writerow(point)
print('almost there....')
with open("DataSets/covid/train.csv", 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',',quoting=csv.QUOTE_MINIMAL)
    for point in train:
        csv_writer.writerow(point)
