#!/usr/bin/env python
# coding: utf-8

# In[2]:


#imports
import json
import codecs
import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt
import math


# In[3]:


#reading data from json file
data = json.load(codecs.open('Airports.json', 'r', 'utf-8-sig')) #read into a dictionary

df = pd.DataFrame(data) #convert dictionary into a df to understand the data better
df.head(n=6)


# In[4]:


len(df)


# ## **Cleaning the Data**

# In[5]:


#while converting longitude and latitude columns to numeric, if any str is present it will be converted to NaN
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')


# In[6]:


#detecting NaN ids
is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df[row_has_NaN]
rows_with_NaN.reset_index(drop=True, inplace=True)


# In[7]:


rows_with_NaN


# In[8]:


#putting NaN ids to remove in an array to remove from destinations later
ids_to_remove = []
for i in range(len(rows_with_NaN)):
  ids_to_remove.append(rows_with_NaN['Airport ID'][i])


# In[9]:


#dropping NaN rows and resetting dataframe index
df = df.dropna()
df.reset_index(drop=True, inplace=True)


# In[16]:


#adjusting all destinations to contain only ids present
for i in range(len(df)):
    destinations = df['destinations'][i]
    for d in destinations: 
      if  d not in df['Airport ID'].values:
        destinations.remove(d)
        df['destinations'][i] = destinations


# In[11]:


#renaming latitude and longitude columns to x and y
df = df.rename(columns = {'Longitude': 'x', 'Latitude': 'y'}, inplace = False)
df.head() 


# In[12]:


df.to_json('Airports2.json')


# ## **Project Code**

# In[13]:


class Airport:
  def __init__(self, id, x, y, destinations):
    self.isVisited = False
    self.distance = np.Infinity
    self.id = id
    self.x = x
    self.y = y
    self.destinations = destinations
    self.distances = []
    self.prev = ""

  def setDistances(self, distances_array):
    self.distances = distances_array

  def setDistance(self, d):
    self.distance = d
  
  def setVisited(self):
    self.isVisited = True
  
  def setPrev(self, prev):
    self.prev = prev

  


# In[14]:



class Dijkstra:
    def __init__(self, df):
    
        #initialize dict and ids array
        self.airport_dict = {}
        self.ids =[]
        
        #initialize graph/dict
        self.airport_dict = {}
        self.ids = []
        self.buildAirport(df)
        self.calDist()
       
        #set start and end 
        self.start = '7252'
        self.end = '45'
        print("start: ",self.start)
        print("end: ",self.end)
        print("__________")
        self.dijkstra()
    
    def buildAirport (self, df):
      for i in range(df.shape[0]):
       
        id = df['Airport ID'][i]
        airport = Airport(id, (df['x'][i]), (df['y'][i]), df['destinations'][i])
        self.airport_dict[id] = airport
        self.ids.append(id)
       
      print("Done building Airports graph")

    def EuclideanDistance(self, x2, x1, y2, y1):
      d = math.sqrt((x2-x1)**2 + (y2 - y1)**2) 
      return d

    #calculate distances between airports   
    def calDist(self):
        for i in range(len(self.ids)):
          destinations =  self.airport_dict[self.ids[i]].destinations
          distances = [] #initial array of weights
          for d in range(len(destinations)):
            source = self.ids[i]
            destination = destinations[d]
            try:
              distances.append(self.EuclideanDistance((self.airport_dict[source].x),
                                                (self.airport_dict[destination].x), 
                                                (self.airport_dict[source].y), 
                                                (self.airport_dict[destination].y)))
            except:
              print("a key error in id no.", ids[i], "and destination", destinations[d])
              continue
            self.airport_dict[self.ids[i]].setDistances(distances)
           
        print("Done calculating distances...")

    def shortestP(self, v, path):
      #calculate shortest path by recursion
      if v.prev:
          path.append(v.prev.id)
          self.shortestP(v.prev, path)
      return

    def dijkstra(self):
      print("______Dijkstra_______")
      #distance assigned to start vertex = 0
      self.airport_dict[self.start].setDistance(0)

      #adding each pair of id to object in a priority queue to get them in order of being pushed
      unvisitedQ = [(self.airport_dict[v].distance, v) for v in self.airport_dict]
      heapq.heapify(unvisitedQ)

      while len(unvisitedQ):
          # Pops a vertex with the smallest distance 
          uv = heapq.heappop(unvisitedQ)
          currentpos = self.airport_dict[uv[1]]
          currentpos.setVisited()
          
          adjacent = currentpos.destinations
          weights = currentpos.distances
          print("__________")
          print("current airport: " , currentpos.id)
          print("destinatios: " , adjacent)
          print("weights: " , weights)
          print("")
          #for next in adjacents:
          for i in range(len(adjacent)):
              # if visited, skip
              next = self.airport_dict[adjacent[i]]
              if next.isVisited:
                  continue
              new_dist = currentpos.distance + weights[i]
              #edge relaxation
              if new_dist < next.distance:
                  next.setDistance(new_dist)
                  next.setPrev(currentpos)
                  print ('updated : current = %s next = %s new_dist = %s'                           %(currentpos.id, next.id, next.distance))
              else:
                  print ('not updated : current = %s next = %s new_dist = %s'                           %(currentpos.id, next.id, next.distance))

          #rebuilding heap
          #pop all
          while len(unvisitedQ):
              heapq.heappop(unvisitedQ)
          #queue all v that is not yet visited
          unvisitedQ = [(self.airport_dict[v].distance,v) for v in self.airport_dict if not self.airport_dict[v].isVisited]
          heapq.heapify(unvisitedQ)

          target = self.airport_dict[self.end] #target airport object
          path = [target.id] #add id to path
          self.shortestP(target, path) #rollback by recursion to find shortest path 
          print ('The shortestP path : %s' %(path[::-1]))


# In[17]:


dijs = Dijkstra(df) 


# In[ ]:




