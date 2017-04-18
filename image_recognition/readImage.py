from scipy import misc
import glob
import numpy as np
import matplotlib.image as img
import os
import csv
import pandas as pd

trainLabels=pd.read_csv('trainLabels.csv')
#print(trainLabels['label'])
labelList=[]
for label in trainLabels['label']:
    labelList.append(label)

labelList=np.asarray(labelList)
print(labelList)
print(labelList.shape)

png = []

for image in glob.glob('test_images/*'):
    print(image)
    png.append(misc.imread(image))

im = np.asarray(png)
print(im[0])
