import os
import random
import numpy as np
import tensorflow as tf

from keras.utils import np_utils
from numpy import argmax
from sklearn.utils import shuffle


SIZE = 6667
SEGMENT_SIZE = 60
DATASET_TRAIN_RATIO = 0.6 

seldir = ["juhee_1","juhee_2","jaebong_2","jaebong_3","taegu_1", "taegu_2", "seonmin_1", "seonmin_2"]

#seldir = ["juhee_1","juhee_2"]

labels = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30"]
'''
labels = [
    "1",
    "6",
    "11",
    "16",
    "21",
    "26"]
'''
a = {}

for label in labels:
    a[label] = []

for dirname in seldir:
    for filename in labels:
        #print(filename)
        data = np.loadtxt(dirname + "/" + filename + '.log')
        data = data[(data.shape[0] - SIZE) / 2 : data.shape[0] - ((data.shape[0] - SIZE) / 2)]
        #print(np.min(data[:, 8]), np.max(data[:, 8]))

        name = filename.split(".")[0]
        for i in range(SEGMENT_SIZE, data.shape[0] + 1):
            a[name].append(data[i - SEGMENT_SIZE:i, :9])
#            a[name].append(np.concatenate((data[i-SEGMENT_SIZE:i,:1],data[i - SEGMENT_SIZE:i,2:3]), axis=1))


    #print("")


for label in labels:
    print(label, np.array(a[label]).shape)
print("")


b = {}
for i in range(len(labels)):
    b[labels[i]] = np.full(np.array(a[labels[i]]).shape[0], i)

x = np.concatenate([a[label] for label in labels], axis=0)

y = np.concatenate([b[label] for label in labels], axis=0)
print(x.shape)
print(y.shape)
print("")


x_avg = [512, 512, 512,0,0,0,0,0,0]
x_res = [512, 512, 512,32768,32768,32768,16384,16384,16384]

x = np.array(x).astype('float32')
for i in range(x.shape[2]):
    #x_min = np.min(x[:, :, i])
    #x_max = np.max(x[:, :, i])
    #x_avg = (x_max + x_min) / 2
    #x_res = (x_max - x_min) / 2
    x[:, :, i] -= x_avg[i]
    x[:, :, i] /= x_res[i]
#    print(x_min, x_max, x_avg, x_res, np.min(x[:, :, i]), np.max(x[:, :, i]))
print("")

y = np_utils.to_categorical(y)

ran_idx = np.random.choice(x.shape[0], x.shape[0])
x = x[ran_idx]
y = y[ran_idx]

#x = x[0:3][:][:]
#print(x.shape)

x_train = np.array(x[:int(x.shape[0] * DATASET_TRAIN_RATIO)]).reshape(-1, x.shape[1] * x.shape[2])
y_train = np.array(y[:int(x.shape[0] * DATASET_TRAIN_RATIO)])
x_test = np.array(x[int(x.shape[0] * DATASET_TRAIN_RATIO):]).reshape(-1, x.shape[1] * x.shape[2])
y_test = np.array(y[int(x.shape[0] * DATASET_TRAIN_RATIO):])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

np.save('x_train', x_train)
np.save('y_train', y_train)
np.save('x_test', x_test)
np.save('y_test', y_test)

