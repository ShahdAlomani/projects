#!/usr/bin/env python
# coding: utf-8

# In[ ]:


**1.Import pandas under the alias pd.**


# In[ ]:


import pandas as  pd


# **2. Print the version of pandas that has been imported.**

# In[ ]:


pd.__version__


# **3. Print out all the version information of the libraries that are required by the pandas library.**

# In[ ]:


pd.show_versions()


# **4. Overview of the Data**

# In[ ]:


animesdf = pd.read_csv("animes.csv")
animesdf.head()


# **5. Display a summary of the basic information**

# In[ ]:


animesdf.info()


# **6. Return the first 3 rows of the DataFrame animesdf.**

# In[ ]:


animesdf.iloc[:3]


# **7. Nummber of Nan in all dataset**

# Check if any Nan in dataset

# In[ ]:


check_nan_in_df = animesdf.isnull()
print (check_nan_in_df)


# In[ ]:


animesdf.isnull().values.any() 


# In[ ]:


animesdf.isnull().sum().sum()


# **8. Check number of  Nan in columns**

# In[ ]:


animesdf[animesdf['episodes'].isnull()]


# In[ ]:


animesdf['episodes'].isnull().values.sum()


# In[ ]:


animesdf[animesdf['ranked'].isnull()]


# In[ ]:


animesdf['ranked'].isnull().values.sum()


# In[ ]:


animesdf[animesdf['score'].isnull()]


# In[ ]:


animesdf['score'].isnull().values.sum()


# **9. Select the rows the ranked is between 25 and 30 (inclusive).**

# In[ ]:


animesdf[animesdf['ranked'].between(25, 30)]


# **10. Select only the rows where the number of episodes is less than 30.**
# 
# 

# In[ ]:


animesdf[animesdf['episodes'] < 30]


# **11.count episodes< 30**

# In[ ]:


animesdf[animesdf['episodes'] < 30].count


# **12.Select the rows where the popularity is a 14113 and the eposides is less than 30.**

# In[ ]:


animesdf[(animesdf['popularity'] == 14113) & (animesdf['episodes'] < 30)]


# **13. Select only the rows where the score > than 6.**
# 
# 

# In[ ]:


animesdf[animesdf['score'] > 6]


# **14.count score >6**

# In[ ]:


animesdf[animesdf['score'] > 6].count


# **15. Find highest rated animes.**

# In[ ]:


#find the highest rating
animesdf['score'].max()


# **16.highest rated animes count**

# In[ ]:


#find number of highest rated animes
animesdf[animesdf['score'] == 9.23].count


# **17. Find lowest rated animes.**

# In[ ]:


#find the lowest rating
animesdf['score'].min()


# **18.count lowest rating**

# In[ ]:


#find number of lowest rated animes
animesdf[animesdf['score'] == 1.25].count


# **19. How many rows are duplicated?**

# In[ ]:


len(animesdf) 


# In[ ]:


len(animesdf.drop_duplicates(keep=False))


# **20.number of duplicated rows**

# In[ ]:


#no of duplicated rows
len(animesdf) - len(animesdf.drop_duplicates(keep=False))


# **21. Arrange data by score range**

# In[ ]:


import numpy as np
animesdf.groupby(pd.cut(animesdf['score'], np.arange(0, 11, 2))).count()


# **22. Sort data with respect to popularity**

# In[ ]:


animesdf.sort_values(by=['popularity'], ascending=[False])


# **23. Select genre and score**
# 

# In[ ]:


animesdf.loc[:, ['genre','score']]


# **24. Select the data in rows [43, 14, 118] and in columns ['title', 'genre']**

# In[ ]:


animesdf.loc[animesdf.index[[43, 14, 118]], ['title', 'genre']]


#  **25.Count the number of each type of  title in animesdf.**

# In[ ]:


animesdf['title'].value_counts()


# **26. For each title and each number of episodes, find the mean ranked. In other words, each row is a title, each column is a number of episodes and the values are the mean ranked.**

# In[ ]:


animesdf.pivot_table(index='title', columns='episodes', values='ranked', aggfunc='mean')


# **27. Drop Nan values**

# In[ ]:


newanimesdf = animesdf.dropna()
newanimesdf


#  **28. Plot histogram after drop nan values**

# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
newanimesdf.plot.hist()


# **29. Histograms visualizing each column individually**

# In[ ]:


newanimesdf.hist(color='Red',figsize= (10,10));


# **30. Adding release year column**

# In[ ]:


newanimesdf['release_year']=newanimesdf['aired'].str.extract(r'(\d{4})') #extracting first 4 digits 


# **31. Extracting count by grouping release_year column**

# In[ ]:


#number of movies per year 
animes_per_year = pd.DataFrame(newanimesdf.groupby('release_year').uid.nunique())
animes_per_year['No. of Animes Released'] = pd.DataFrame(newanimesdf.groupby('release_year').uid.nunique())
animes_per_year.drop('uid',axis='columns', inplace=True)
animes_per_year.head()


# **32. Visualizing release_year vs. count in a line graph.**

# In[ ]:


# plot line graph
animes_per_year.plot.line(title = 'Animes released per year',color='Red', figsize=(8, 8));


# **33. Extracting the mean score for each year.**

# In[ ]:


#number of movies per year 
rating_per_year = pd.DataFrame(newanimesdf.groupby('release_year').score.mean())
rating_per_year['Mean Rating'] = pd.DataFrame(newanimesdf.groupby('release_year').score.mean())
rating_per_year.drop('score',axis='columns', inplace=True)
rating_per_year.head()


# **34. Visualizing release_year vs. mean score in a line graph.**

# In[ ]:


# plot line graph
rating_per_year.plot.line(title = 'Mean Rating Per Year',color='Blue', figsize=(8, 8));


# **35. Extract genre titles individually.**

# In[ ]:


newanimesdf['genre'] = newanimesdf['genre'].str.extract('([a-zA-Z]+)')
genres = (newanimesdf.genre.str.split(',', expand=True)
            .stack()
            .to_frame(name='genre'))


genres.index = genres.index.droplevel(1)


# In[ ]:


genres


# **36. Bar plot the genre types vs no. of members interested in it.**

# In[ ]:


(genres.join(newanimesdf['members'])
       .groupby('genre')
       .sum()
       .plot(kind='bar'))


# **37. Bar plot the genre types vs its popularity.**

# In[ ]:


(genres.join(newanimesdf['popularity'])
       .groupby('genre')
       .sum()
       .plot(kind='bar'))


# **38. Number of animes with an unknown airing date.**

# In[ ]:


animesdf[animesdf['aired']=='Not available'].count


# In[ ]:


animesdf


# **39. Some values in the the ranked column are missing (they are NaN). These numbers are meant to increase by 10 with each row so 10055 and 10075 need to be put in place. Modify df to fill in these missing numbers and make the column an integer column (instead of a float column).**

# In[119]:


animesdf['ranked'] = animesdf['ranked'].interpolate().astype(int)
animesdf


#  **40. In the Members column, the values have been entered into the DataFrame as a list. We would like each first value in its own column, each second value in its own column, and so on. If there isn't an Nth value, the value should be NaN.**

# In[120]:


delays = animesdf['members'].apply(pd.Series)

delays.columns = ['delay_{}'.format(n) for n in range(1, len(delays.columns)+1)]

animesdf = animesdf.drop('members', axis=1).join(delays)

animesdf

