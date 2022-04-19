#!/usr/bin/env python
# coding: utf-8

# In[100]:


#Importing Libraries
import pandas as pd 
import seaborn as sns
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
warnings.filterwarnings('ignore')


# In[ ]:





# In[107]:


a=[]

grp_per_min = format(np.random.exponential(scale=5), ".4f") # .4f = 4 digit after . | for delta 1/5
grp_per_min = grp_per_min[0]
print("Number of groups", grp_per_min)

for m in range(0, int(grp_per_min)): 
    num_in_group = int(random.normal(loc=5 , scale=3, size=(1, 1)) )
    a.append(abs(num_in_group))
    print("number of visitors in this group",  num_in_group)
print(a)
num_vsts =0
for ele in range(0, len(a)):
    num_vsts = num_vsts + a[ele]
print("Total number of visitors is", num_vsts)

####

from_booth = int(num_vsts *0.4  )
from_booth = round(from_booth, 0)
print("from booth sales", from_booth)

ext = int(from_booth % 4)
print(ext)

from_online = int(num_vsts) *0.4  
from_online = round(from_online, 0)
print("from online sales", from_online)

from_staff_credentails = int(num_vsts) *0.2
from_staff_credentails = round(from_staff_credentails, 0)
print("from staff credentails", from_staff_credentails)
ext1 = ext
v = int(from_booth)
if v < 4:
    c = v
else:
    c = 4


# In[ ]:





# In[108]:


## For visitors who buy tickets from booth tables
#generating inter arrival times using exponential distribution
table_1 = []
table_2 = []
table_3 = []
table_4 = []
while c >= 1:
    print("c", c)
    if ext !=0 :
        ext1 = ext
    if ext == 0 :
        ext1 = ext +1
    if ext < 0:
        ext = ext + 1

    print("#####     Table       #####")
    bus_booth_time =  list(random.normal(loc=1 , scale=0.25 ,size=int(ext/ext1 + from_booth/4)).round(4))
    print("Bus to booth times for vistors are", bus_booth_time)
        

    arrival_times= []# list of arrival times of a person joining the queue
    finish_times = [] # list of finish times after waiting and being served

    arrival_times = [0 for i in range(int(ext/ext1 + from_booth/4))]
    finish_times = [0 for i in range(int(ext/ext1 + from_booth/4))]
    arrival_times[0]=round(bus_booth_time[0],4)#arrival of first customer



    # Generate arrival times
    for i in range(1,int(ext/ext1 + from_booth/4)):
        arrival_times[i]=round((arrival_times[i-1]+bus_booth_time[i]),4)      
    print("arrival times are ",arrival_times)


    # Generate random service times for each customer                      
    sale_times = list(random.normal(loc=1 , scale=0.2 ,size=int(ext/ext1 + from_booth/4)).round(4))
    print("sale times are ",sale_times)

    # Generate random sale to scan times for each customer                      
    sell_scan_time = list(random.normal(loc=0.5 , scale=1  ,size=int(ext/ext1 + from_booth/4)).round(4))
    print("sale to scan times are ",sale_times)  

    # Generate random scan times for each customer                      
    scan_times = list(random.normal(loc=0.05 , scale=0.01  ,size=int(ext/ext1 + from_booth/4)).round(4))
    print("scan times are ",scan_times)

    # Generate random scaniing-to-table times for each customer                      
    to_table_time = list(random.normal(loc=1 , scale=0.2 ,size=int(ext/ext1 + from_booth/4)).round(4))
    print("to the tables times are ",to_table_time)

    # Generate finish times
    finish_times[0]= round((arrival_times[0]+ sale_times[0] + sell_scan_time[0]
                                +scan_times[0] + to_table_time[0] + bus_booth_time[0]),4)

    for i in range(1, int(ext/ext1 + from_booth/4)):
        previous_finish=finish_times[:i]
        previous_finish.sort(reverse=True)
        previous_finish=previous_finish[:c]
        if i< c:
            finish_times[i] = round((arrival_times[i]+ sale_times[i] + sell_scan_time[i]
                                +scan_times[i] + to_table_time[i] + bus_booth_time[i]),4)
        else:
            finish_times[i]=round((max(arrival_times[i],min(previous_finish))+ sale_times[i] + sell_scan_time[i]
                                +scan_times[i] + to_table_time[i] + bus_booth_time[i]),4)

    print("finish times are ",finish_times)

    # Total time spent in the system by each customer
    total_times =[abs(round((finish_times[i]-arrival_times[i]),4)) for i in range( int(ext/ext1 + from_booth/4))]
    print("total times are ",total_times)

    for r in range(0, int(ext/ext1 + from_booth/4)):    
        if c is 4:
            table_1.append(total_times[r])
        elif c is 3:
            table_2.append(total_times[r])
        elif c is 2:
            table_3.append(total_times[r])
        elif c is 1:
            table_4.append(total_times[r])
        r = r+1
    ext = ext - 1
    c = c - 1    
    

    data = pd.DataFrame(list(zip(arrival_times, bus_booth_time, sale_times, sell_scan_time, scan_times, to_table_time, finish_times, total_times)), 
            columns =['arrival_times','bus_booth_time', 'sale_times', 'sell_scan_times', 'scan_times', 'to_table_times','finish_times','total_times']) 

data.head()
print(table_1)
print(table_2)
print(table_3)
print(table_4)


# In[109]:


if from_booth >= 4:
    fig = plt.figure(figsize = (len(table_1), max(table_1)+4))
    br1 = np.arange(len(table_1))
    plt.bar(br1, table_1, color ='maroon',  width = 0.4)

    plt.xlabel("visitors in table 1")
    plt.ylabel("total time ")
    plt.title("total time from the bus to getting to the table")
    plt.show()

    fig = plt.figure(figsize = (len(table_2), max(table_2)+4))
    br2 = np.arange(len(table_2))
    plt.bar(br2, table_2, color ='red',  width = 0.4)

    plt.xlabel("visitors in table 2")
    plt.ylabel("total time ")
    plt.title("total time from the bus to getting to the table")
    plt.show()

    fig = plt.figure(figsize = (len(table_3), max(table_3)+4))
    br3 = np.arange(len(table_3))
    plt.bar(br3, table_3, color ='blue',  width = 0.4)

    plt.xlabel("visitors in table 3")
    plt.ylabel("total time ")
    plt.title("total time from the bus to getting to the table")
    plt.show()

    fig = plt.figure(figsize = (len(table_4), max(table_4)+4))
    br4 = np.arange(len(table_4))
    plt.bar(br4, table_4, color ='green',  width = 0.4)

    plt.xlabel("visitors in table 4")
    plt.ylabel("total time ")
    plt.title("total time from the bus to getting to the table")
    plt.show()


# In[110]:


if from_booth >= 4:
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].bar(br1, table_1, color ='maroon',  width = 0.4)
    axs[0, 0].set_title('table 1')

    axs[0, 1].bar(br2, table_2, color ='red',  width = 0.4)
    axs[0, 1].set_title('table 2')

    axs[1, 0].bar(br3, table_3, color ='blue',  width = 0.4)
    axs[1, 0].set_title('table 3')

    axs[1, 1].bar(br4, table_4, color ='green',  width = 0.4)
    axs[1, 1].set_title('table 4')

    for ax in axs.flat:
        ax.set(xlabel='VISITORS ', ylabel='TOTAL TIME')
        


# In[111]:


plt.plot(br1, table_1, label = "table 1")
plt.plot(br2, table_2, label = "table 2")
plt.plot(br3, table_3, label = "table 3")
plt.plot(br4, table_4, label = "table 4")
plt.legend()
# Display a figure.
plt.show()


# In[112]:


if from_booth >= 4:
    average1 = sum(table_1)/ len(table_1)
    print("Average total time in table 1  =", round(average1, 2))
    average2 = sum(table_2)/ len(table_2)
    print("Average total time in table 2  =", round(average2, 2))
    average3 = sum(table_3)/ len(table_3)
    print("Average total time in table 3  =", round(average3, 2))
    average4 = sum(table_4)/ len(table_4)
    print("Average total time in table 4  =", round(average4, 2))

    comp = [average1, average2, average3, average4]
    print("avetage total time in tables is", comp)
    fast = min(comp)
    if fast is average4:
        print("TABLE 4 the fasest service")
    elif fast is average3:
        print("TABLE 3 the fasest service")
    elif fast is average2:
        print("TABLE 2 the fasest service")
    elif fast is average1:
        print("TABLE 1 the fasest service")


# In[113]:


## for visitors who bought online tickits and stuff creddintials
table_11 = []
table_22 = []
table_33 = []
table_44 = []
print(int(num_vsts*0.6))
ext2 =  int(num_vsts*0.6)%4
y = int(ext2 + num_vsts*0.6)
if y < 4:
    c = y
else:
    c = 4
while c >= 1:
    print(ext2)
    if ext2 !=0 :
        ext22 = ext2
    if ext2 == 0 :
        ext22 = ext2 +1
    if ext2 < 0:
        ext2 = ext2 + 1
    print("###### TABLE #######")
    print("'                           '")
    walk_from_bus_ToScan = list(random.normal(loc= 1.5 , scale=0.35  ,size=int(ext2/ext22 + num_vsts*0.6/4)).round(4))
    print("walk from bus ToScan are ",walk_from_bus_ToScan)
# Generate random scan times for each customer                      
    scan_time2 = list(random.normal(loc=0.05 , scale=0.01  ,size=int(ext2/ext22 +num_vsts*0.6/4)).round(4))
    print("scan times are ",scan_time2)
       
    to_table_time2 = list(random.normal(loc=1 , scale=0.2 ,size=int(ext2/ext22 +num_vsts*0.6/4)).round(4))
    print("to the tables times are ",to_table_time2)

    total_time = [0 for i in range(int(ext2/ext22 +num_vsts*0.6/4))]
    for i in range(0, int(ext2/ext22 + num_vsts*0.6/4)):
        total_time[i] = round((walk_from_bus_ToScan[i] + scan_time2[i] + to_table_time2[i] ),4)
    print("the total time is ",total_time)
    data2 = pd.DataFrame(list(zip(walk_from_bus_ToScan,  scan_time2, to_table_time2, total_time)), 
                columns =['walk_from_bus_ToScan','scan_time2', 'to_table_time2', 'total_time']) 
    data2.head()
    for rr in range(0, int(int(ext2/ext22 +num_vsts*0.6/4))):    
        if c is 4:
            table_11.append(total_time[rr])
        elif c is 3:
            table_22.append(total_time[rr])
        elif c is 2:
            table_33.append(total_time[rr])
        elif c is 1:
            table_44.append(total_time[rr])
        rr = rr+1
    c = c -1
    ext2 = ext2 -1
print(table_11)


# In[114]:



if int(ext2 +num_vsts*0.6) >= 4:
    fig, axs = plt.subplots(2, 2)
    br11 = np.arange(len(table_11))
    axs[0, 0].bar(br11, table_11, color ='maroon',  width = 0.4)
    axs[0, 0].set_title('table 1')

    br22 = np.arange(len(table_22))
    axs[0, 1].bar(br22, table_22, color ='red',  width = 0.4)
    axs[0, 1].set_title('table 2')
    
    br33 = np.arange(len(table_33))
    axs[1, 0].bar(br33, table_33, color ='blue',  width = 0.4)
    axs[1, 0].set_title('table 3')

    br44 = np.arange(len(table_44))
    axs[1, 1].bar(br44, table_44, color ='green',  width = 0.4)
    axs[1, 1].set_title('table 4')
    for ax in axs.flat:
        ax.set(xlabel='VISITORS ', ylabel='TOTAL TIME')


# In[115]:


if int(ext2 +num_vsts*0.6) >= 4:
    average11 = sum(table_11)/ len(table_11)
    print("Average total time in table 1  =", round(average11, 2))
    average22 = sum(table_22)/ len(table_22)
    print("Average total time in table 2  =", round(average22, 2))
    average33 = sum(table_33)/ len(table_33)
    print("Average total time in table 3  =", round(average33, 2))
    average44 = sum(table_44)/ len(table_44)
    print("Average total time in table 4  =", round(average44, 2))

    comp1 = [average11, average22, average33, average44]
    print("avetage total time in tables is", comp1)
    fast1 = min(comp1)
    if fast1 is average44:
        print("TABLE 4 the fasest service")
    elif fast1 is average33:
        print("TABLE 3 the fasest service")
    elif fast1 is average22:
        print("TABLE 2 the fasest service")
    elif fast1 is average11:
        print("TABLE 1 the fasest service")


# In[116]:


import pandas as pd
import sqlite3

cnn = sqlite3.connect('r1.db')
get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'sqlite:///r1.db')
data.to_sql('r1', cnn)

data


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('sql', '', 'SELECT * FROM r1')


# In[ ]:


get_ipython().run_cell_magic('sql', '', 'SELECT * FROM r where total_times = 5.1438')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


pip install latax

