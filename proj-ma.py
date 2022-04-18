#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Run this cell and select the kaggle.json file downloaded
# from the Kaggle account settings page.
from google.colab import files
files.upload()


# In[ ]:


# Let's make sure the kaggle.json file is present.
get_ipython().system('ls -lha kaggle.json')
# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')

# This permissions change avoids a warning on Kaggle tool startup.
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


get_ipython().system(' mkdir train')
get_ipython().system(' unzip gtsrb-german-traffic-sign.zip -d gtsrb-german-traffic-sign')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import os
import warnings
import glob
from PIL import Image
import matplotlib.image as img
import cv2
import numpy as np
warnings.filterwarnings("ignore")


# In[ ]:


# Number of total classes
total_classes = 43

# Dimensions of our images
height = 32
width = 32
channels = 3


# In[ ]:


X = []
y = []

folder = '/content/gtsrb-german-traffic-sign'
for i in range (0, 43):
    path = folder + "/Train/" + str(i)
    images = glob.glob(path + '/*.png')
    for image in images:
        image = cv2.imread(image).T
        resize_image = np.resize(image, (3, height, width))
 
        currentImg = np.asarray(resize_image.flatten())
   
        X.append(currentImg)
        y.append(i)


# In[ ]:


X = np.array(X)


# In[ ]:


y = np.array(y)


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


# Shuffling data
shuffle_indexes = np.arange(X.shape[0])
np.random.shuffle(shuffle_indexes)

X = X[shuffle_indexes]
y = y[shuffle_indexes]


# In[ ]:


from sklearn.model_selection import train_test_split
# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                     random_state=42, shuffle=True)
# Scale the values between 0 and 1
X_train = X_train / 255
X_test = X_test / 255


# # get validation set

# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# **PCA**

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_trainn = sc.fit_transform(X_train)
X_testt = sc.transform(X_test)


# In[ ]:


pca = PCA()
X_trainpca = pca.fit_transform(X_train)
X_testpca = pca.transform(X_test)


# In[ ]:


explained_variance = pca.explained_variance_ratio_
print(explained_variance)


# **RandomForestClassifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy :', accuracy_score(y_test, y_pred))


# **DecisionTreeClassifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# **naive_bayes**

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train, y_train)
y_predict = model.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test, y_predict))
print('Accuracy :', accuracy_score(y_test, y_predict))


# **GradientBoostingClassifier**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))


# # **NN model** 

# In[ ]:


import os
import time as time

import numpy as np
np.random.seed(40)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image
import time
from datetime import timedelta
import pandas as pd
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import skimage.morphology as morp
from skimage.filters import rank

import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout # new!
from keras.layers.normalization import BatchNormalization # new!
from keras import regularizers # new! 
from keras.optimizers import SGD
from keras.layers import Flatten, Conv2D, MaxPooling2D # new!
from keras.callbacks import ModelCheckpoint

import cv2


# In[ ]:


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(32*32*3,)))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(43, activation='softmax'))


# In[ ]:


model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# In[ ]:


X_train_baseline = X_train.reshape(len(X_train), 32*32*3).astype('float32')
X_valid_baseline = X_test.reshape(len(X_test), 32*32*3).astype('float32')
y_train_baseline = keras.utils.to_categorical(y_train,43)
y_valid_baseline = keras.utils.to_categorical(y_test,43)


# In[ ]:


model.fit(X_train_baseline, y_train_baseline, batch_size=128, epochs=50, verbose=1, validation_data=(X_valid_baseline, y_valid_baseline))


# In[ ]:


X_test = np.asarray(X_test)
X_test_baseline = X_test.reshape(len(X_test), 32*32*3).astype('float32')
y_test_baseline = keras.utils.to_categorical(y_test,43)


# In[ ]:


for i in range(43):
  class_records = np.where(y_train==i)[0].size
  if class_records == 0:
    for j in range(10):
      temptemp = np.where(y_valid == i)
      tempindex = temptemp[j]
      templable = y_valid[tempindex]
      tempvar =X_valid[tempindex]
      np.delete(y_valid,tempindex)
      np.delete(X_valid,tempindex)
      X_train = np.append(X_train,tempvar)
      y_train = np.append(y_train,templable)


# In[ ]:


def data_augment(image):
    rows= image.shape[0]
    cols = image.shape[1]
    
    # rotation
    M_rot = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
    
    # Translation
    M_trans = np.float32([[1,0,3],[0,1,6]])
    
    
    img = cv2.warpAffine(image,M_rot,(cols,rows))
    img = cv2.warpAffine(img,M_trans,(cols,rows))
    #img = cv2.warpAffine(img,M_aff,(cols,rows))
    
    # Bilateral filtering
    img = cv2.bilateralFilter(img,9,75,75)
    return img


# In[ ]:


classes = 43

X_train_final = X_train
y_train_final = y_train
X_aug_1 = []
Y_aug_1 = []

for i in range(0,classes):
    
    class_records = np.where(y_train==i)[0].size
    max_records = 4000
    if class_records != max_records:
        ovr_sample = max_records - class_records
        samples = X_train[np.where(y_train==i)[0]]
        X_aug = []
        Y_aug = [i] * ovr_sample
        for x in range(ovr_sample):
            img = samples[int(round(x % class_records))]
            trans_img = data_augment(img)
            X_aug.append(trans_img)   
        X_train_final = np.concatenate((X_train_final, X_aug), axis=0)
        temp, = np.shape(Y_aug)
        Y_aug = np.reshape(Y_aug,(temp,1))
        y_train_final = np.concatenate((y_train_final, Y_aug)) 
        Y_aug_1.append( Y_aug)
        X_aug_1.append( X_aug)


#  **Unique elements**

# In[ ]:


unique_elements, counts_elements = np.unique(y_train_final, return_counts = True)
print(np.asarray((unique_elements, counts_elements)))

pyplot.bar( np.arange( 43 ), counts_elements, align='center',color='green' )
pyplot.xlabel('Class')
pyplot.ylabel('No of Training data')
pyplot.xlim([-1, 43])

pyplot.show()


# In[ ]:


X_train_baseline = X_train_final.reshape(len(X_train_final), 32*32*3).astype('float32')
X_valid_baseline = X_valid.reshape(len(X_valid), 32*32*3).astype('float32')
y_train_baseline = keras.utils.to_categorical(y_train_final,43)
y_valid_baseline = keras.utils.to_categorical(y_valid,43)


# In[ ]:


model.fit(X_train_baseline, y_train_baseline, batch_size=128, epochs=50, verbose=1, validation_data=(X_valid_baseline, y_valid_baseline))


# In[ ]:


Pred = model.evaluate(X_test_baseline, y_test_baseline, verbose=0)
print("Dense fully connected network results on the test data - Baseline ")
print(" ")
print("%s- %.2f" % (model.metrics_names[0], Pred[0]))
print("%s- %.2f" % (model.metrics_names[1], Pred[1]))


# # **AT&T Dataset**

# In[ ]:


get_ipython().system('ls -lha kaggle.json')
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('kaggle datasets download -d kasikrit/att-database-of-faces')


# In[ ]:


get_ipython().system(' unzip att-database-of-faces.zip -d att-database-of-faces')


# In[ ]:


import numpy as np
import cv2
import math
from PIL import Image

import matplotlib.image as img

from sklearn.metrics import accuracy_score


# In[ ]:


X = np.zeros((0, 10304))
y = np.empty((0,400))

folder = '/content/att-database-of-faces'
for j in range (1, 41):
    direction = folder + '/s' + str(j) + '/'
    for i in range(1, 11):
        directory = direction + str(i) + '.pgm'
        image = img.imread(directory).T
        #image = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
        currentImg = np.asmatrix(image.flatten())
        X = np.concatenate((X, currentImg))
        y = np.append(y, j)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=42)


# ## **Linear Regression**

# In[ ]:


from sklearn.linear_model import LinearRegression
LinearRegressionModel = LinearRegression()


# In[ ]:


LinearRegressionModel.fit(X_train, y_train)


# In[ ]:


y_pred = LinearRegressionModel.predict(X_test)


# In[ ]:


y_pred = np.rint(y_pred)


# In[ ]:


accuracy_score(y_test, y_pred)


# ## **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
LogisticRegressionModel = LogisticRegression(solver='liblinear', random_state=0)


# In[ ]:


LogisticRegressionModel.fit(X_train, y_train)


# In[ ]:


y_pred = LogisticRegressionModel.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# ## **Adaboost**

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


AdaBoostClassifierModel = AdaBoostClassifier(n_estimators=100)


# In[ ]:


AdaBoostClassifierModel.fit(X_train, y_train)


# In[ ]:


y_pred = AdaBoostClassifierModel.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# ## **SVM**

# In[ ]:


from sklearn import svm
SVModel = svm.SVC()


# In[ ]:


SVModel.fit(X_train, y_train)


# In[ ]:


y_pred = SVModel.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# ## **LDA**

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis()


# In[ ]:


LDA.fit(X_train, y_train)


# In[ ]:


y_pred = LDA.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# ## **KMeans**

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeans = KMeans(n_clusters=41)
kmeans.fit(X_train)


# In[ ]:


y_pred = kmeans.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)

