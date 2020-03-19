# -*- coding: utf-8 -*-
"""
Digit Recognition

@author: aman_pal
"""
# Importing classes
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import scipy.io

# download dataset
#features, labels = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
dataset = scipy.io.loadmat('D:/Bruhhhhh/Some_Data/mnist-original.mat')

features = np.array(dataset.get('data'), 'int16')
features = np.transpose(features)
labels = np.array(dataset.get('label'), 'int')
labels = np.transpose(labels)

# calculate HOG value
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28,28)), orientations=9, pixels_per_cell=(7,7), cells_per_block=(1,1), visualize=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# something
#hog_features = np.hstack((hog_features,np.ones((hog_features.shape[0],1))))

# create SVM object
#clf = LinearSVC(C=3)
clf = LinearSVC()

clf.fit(hog_features, labels)

# saving the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)