#!/usr/bin/env python
# coding: utf-8

# ### Swarm Optimization  Optimization(PSO)
# <br>
# <font color='purple' size='3.5'>
# In computational science, particle swarm optimization (PSO)[1] is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It solves a problem by having a population of candidate solutions, here dubbed particles, and moving these particles around in the search-space according to simple mathematical formulae over the particle's position and velocity. Each particle's movement is influenced by its local best known position, but is also guided toward the best known positions in the search-space, which are updated as better positions are found by other particles. This is expected to move the swarm toward the best solutions.</font>

# (attachment:Screen%20Shot%202020-11-03%20at%2011.44.47%20AM.png)

# ![Screen%20Shot%202020-11-03%20at%2011.39.53%20AM.png](attachment:Screen%20Shot%202020-11-03%20at%2011.39.53%20AM.png)

# ![Screen%20Shot%202020-11-03%20at%2011.51.16%20AM.png](attachment:Screen%20Shot%202020-11-03%20at%2011.51.16%20AM.png)

# ![Screen%20Shot%202020-10-28%20at%2010.07.01%20AM.png](attachment:Screen%20Shot%202020-10-28%20at%2010.07.01%20AM.png)

# 

# ### For more details:
# - Research paper: https://reader.elsevier.com/reader/sd/pii/S0898122111004299?token=FBDD045726FCB6A6EB52C4ABE434511AFAE5D435CD413894BACBC584E5A784D0009C7A33B0EA45B36E4A1B2ED2FFCFBD
# - powerpoint:
# https://www.slideshare.net/raafiubrian/particles-swarm-optimization
# - video:
# https://www.youtube.com/watch?v=uwXFnzWaCY0

# #### <font color='red'> Write Python program for applying PSP for solving any n variable optmization problem. Your program must follow the following:</font>

# # Shahd Alomani       421010088

# In[11]:


import sys
class pso:
    def __init__(self, no_of_particles):
        self.W = 0.7
        self.c1 = 0.2
        self.c2 = 0.6   
        self.no_of_particles=no_of_particles
        self.f = (x1 + 10*x2)**2 + 5*(x3 - x4)**2 + (x2 - 2*x3)**2 + 10*(x1-x4)**4 #given equation
        self.no_x = 4 #number of variables
        self.lst_particles=[]
        self.gBest = [0, 0, 0, 0]
      
    def create_particles(self):
        ## initialize parameters for each particle
        for i in range(self.no_of_particles):
          p=particle(self.f,self.no_x)
          self.lst_particles.append(p)
     

  

    def Fitness(self, x):
      x1, x2, x3, x4 = symbols('x1, x2, x3, x4')
      result = self.f.subs({x1:x[0], x2: x[1], x3:x[2], x4: x[3]})
      
      return result
  
 
    def calculate_new_velocity(self,particle, i):
  
      # generate random numbers
      r1 = random.random()
      r2 = random.random()
      
      return (self.W * particle.v[i]) + (self.c1 * r1 * (particle.pBest[i] - particle.x[i])) + (self.c2 * r2 * (self.gBest[i] - particle.x[i]))
      

    def start_calculation(self, iter=100):
      self.create_particles()
      gbest_fitness = 100000000000000000000
      for p in range(self.no_of_particles): #to find initial gBest
        print(self.lst_particles[p].x)
        
        new_fitness = self.Fitness(self.lst_particles[p].x)
        print(new_fitness)
        if new_fitness < gbest_fitness :
              self.gBest = self.lst_particles[p].x
              gbest_fitness = self.Fitness(self.gBest)
              
              
      print("initial gBest:",self.gBest, "gBest fitness:",self.Fitness(self.gBest))

      for i in range(iter): #apply second part in pseudo code
        for p in range(self.no_of_particles):
          for x in range(self.no_x):
            self.lst_particles[p].v[x] = self.calculate_new_velocity(self.lst_particles[p], x)
            self.lst_particles[p].x[x]=self.lst_particles[p].x[x]+self.lst_particles[p].v[x]
          #print("gbest before: ",self.Fitness(self.gBest))
          if self.Fitness(self.lst_particles[p].x) < self.lst_particles[p].fitness:
            self.lst_particles[p].pBest = self.lst_particles[p].x
            #self.lst_particles[p].fitness = new_fitness
            self.lst_particles[p].fitness = (self.Fitness(self.lst_particles[p].pBest))
      
          if self.Fitness(self.lst_particles[p].x) < self.Fitness(self.gBest):
            #print("gbest in if before: ",self.Fitness(self.gBest))
            self.gBest = self.lst_particles[p].x
            #print("gbest in if before: ",self.Fitness(self.gBest))
          
        print("iteration number", i+1)
        print("final gBest: ", self.gBest, "final fitness: ", self.Fitness(self.gBest))
        for p in range(self.no_of_particles):
          print("final pBest for particle",p,": ", self.lst_particles[p].pBest,  "fitness: ", self.Fitness(self.lst_particles[p].pBest))
        print("______________________________________________________________________________________")


# In[5]:


import numpy as np
import random 
from sympy import *
x1, x2, x3, x4 = symbols('x1, x2, x3, x4')
class particle:
    def __init__ (self,f,no_x):  
     
      self.x=[None] * no_x
      self.v=[None] * no_x
      self.f = f
      for i in range(no_x):
        self.x[i]=np.random.randint(-20,20)
        self.v[i]=np.random.randint(-300,30)
      self.pBest=self.x
      self.fitness = self.Fitness(self.x)

    def Fitness(self, x):
      x1, x2, x3, x4 = symbols('x1, x2, x3, x4')
      result = self.f.subs({x1:x[0], x2: x[1], x3:x[2], x4: x[3]})
      
      return result


# In[12]:


pso_object = pso(5) #initialize pso with number of particles


# In[13]:


pso_object.start_calculation()

