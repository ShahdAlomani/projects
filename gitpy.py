#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
from matplotlib import pyplot
import math
import random


# In[2]:


k=3    #ENN Flag
MAX_LIMIT=3000    #max no. of total images in dataset ... for faster processing
MAX_LIMIT_PER_CLASS=300  #max no. of images in each class ... 
CLASSES = 10 #no. of classes


# ## Help Functions

# In[3]:


def euclidean (a, b):
  d = 0
  for i in range(len(a)):
    d = d + math.pow((a[i] - b[i]), 2)
  return math.sqrt(d)


# In[4]:


# K Nearest Neighbours Function
#=============== updated ======================
def getKNN (image, array, k):
  distances = [] #array of all distances between points
  for x in range(len(array)):
    d = euclidean( image ,array[x][0]) #get euclidean distance between the 2 points
    distances.append((array[x], d))
  distances.sort(key=lambda x: x[1]) #sort in ascending order by the distance values not the image using a lambda expression

  neighbours = [] #list of neighbours
  for x in range(k):
      neighbours.append(distances[x][0]) #return neighbours of smallest k (first k in the distances array)
  return neighbours


# In[5]:


# To get Count of special lbl
def getcount(lbl,lst):
    return [x for a,x in lst].count(lbl)


# In[6]:


# comprehension list
from collections import Counter 
def most_frequent_test(array): 
    x=[item[1] for item in array]  
    occurence_count = Counter(x) 
    return occurence_count.most_common(1)[0][0]


# # Start Project Code

# In[7]:


fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels),(test_images,test_labels)=fashion_mnist.load_data()


# In[8]:


# Explain about the class 10, why not visiable
for i in range(9):
    pyplot.subplot(330+1+i)
    pyplot.imshow((train_images[i]),cmap=pyplot.get_cmap('gray'))
pyplot.show()


# In[9]:


train_images=train_images.reshape(train_images.shape[0],-1)
train_images1=list(train_images)[:MAX_LIMIT]
train_labels1=list(train_labels)[:MAX_LIMIT]


# In[10]:


train_images=train_images.reshape(train_images.shape[0],-1)
train_images1=train_images.tolist()[:MAX_LIMIT]
train_labels1=train_labels.tolist()[:MAX_LIMIT]


# In[11]:


print(len(train_images1[1])) #28 * 28 = size of image


# In[12]:


print(len(train_images1))


# In[13]:


print(len(train_labels1))


# In[14]:


#zipping and shuffling
zippedList=list(zip(train_images,train_labels))
random.shuffle(zippedList)


# In[15]:


# To get 300 Images of each class
print(len(zippedList))

labels =[]
images =[]
count =0
for i in range(CLASSES):
    count=0;
    for image,label in zippedList:
        if(label==i and count!=MAX_LIMIT_PER_CLASS):
            labels.append(label)
            images.append(image)
            count+=1
        
finalList=list(zip(images,labels))
print(len(finalList))


# In[16]:


# Visualization of the number of images in each class
for i in range(10):
    print("Class", i, ":", labels.count(i))


# In[17]:


tempMajority=[]  # get majority (with redundancy)
MainMajority=set() # set removes redundancies

limitGiven=10     # originally 3000
limitTotal=500  # it originally 60000

# shuffling
random.shuffle(finalList)

for img,lbl in finalList[:limitGiven]:
    tempMajority=getKNN(img,zippedList[:limitTotal],k)
    MainMajority.add(most_frequent_test(tempMajority))

print(MainMajority)


# In[18]:


# To Check the List with Majority Items, If the lbl not one of the Majority list, remove it

MainListMajority=[]

for item in finalList:
    if item[1] in MainMajority:
        MainListMajority.append(item)
      
print(len(MainListMajority))        
print(len(finalList))


# In[19]:


# Visualization of no. of images/class with class names 
import pandas as pd
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#Adding to pandas dataframe to be able to visualize the results
df = pd.DataFrame(MainListMajority, columns= ['Image', 'Label'])

classes = []
for x in range(len(df)):
  classes.append(class_names[df['Label'][x]])

df['Class Name'] = classes
df


# In[20]:


#get no. of unique classes of each group in df.

classes_present = df.pivot_table(index=['Class Name'], aggfunc='size')
classes_present

