import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

import matplotlib.pyplot as plt
import scipy.misc



training_fingerprints = np.load('model/segmentTest/10/x_train.npy')
training_ground_truth = np.load('model/segmentTest/10/y_train.npy')
test_fingerprints = np.load('model/segmentTest/10/x_test.npy')
test_ground_truth = np.load('model/segmentTest/10/y_test.npy')

test_fingerprints *= (0x1 << 7)

f = open('input.h', 'wb')
f.close()

with open('input.h', 'a') as f:
    f.write('#define INPUT {')

    for i in range(89):
        f.write(str(int(test_fingerprints[0][i])))
        f.write(', ')
    f.write(str(int(test_fingerprints[0][89])))

    f.write('}\n')


print(test_fingerprints[0])
print(test_ground_truth[0])
