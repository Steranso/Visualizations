#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas

lyric_data = pandas.read_csv("./data/hip-hop-candidate-lyrics/genius_hip_hop_lyrics.csv", encoding= "ISO-8859-1")
trump_lyrics = lyric_data[lyric_data["candidate"] == "Donald Trump"]

trump_lyrics.shape

pandas.crosstab(trump_lyrics.album_release_date, trump_lyrics.sentiment)

import seaborn
import matplotlib.pyplot as plt

trump_crosstab = pandas.crosstab(trump_lyrics.album_release_date, trump_lyrics.sentiment)

trump_crosstab.plot.bar(stacked=True)
plt.legend
plt.show

print('max negative is', max(trump_crosstab['negative']))
print('max positive is',max(trump_crosstab['positive']))


# In[2]:


max(trump_crosstab['positive'])


# In[9]:


lyric_data


# In[11]:


Hey_papi_lyrics = lyric_data[lyric_data["song"] == "Hey Papi"]
Hey_papi_lyrics


# In[13]:


clinton_lyrics = lyric_data[lyric_data["candidate"] == "Hillary Clinton"]
clinton_crosstab = pandas.crosstab(clinton_lyrics.album_release_date, clinton_lyrics.sentiment)
print('max negative is', max(clinton_crosstab['negative']))
clinton_crosstab.plot.bar(stacked=True)
plt.legend
plt.show


# In[14]:


trump_crosstab


# In[16]:


clinton_lyrics.shape


# In[17]:


Sanders_lyrics = lyric_data[lyric_data["candidate"] == "Bernie Sanders"]

Sanders_lyrics.shape


# In[18]:


huckabee_lyrics = lyric_data[lyric_data["candidate"] == "Mike Huckabee"]

huckabee_lyrics.shape


# In[7]:


money_lyrics = lyric_data[lyric_data["theme"] == "money"]
money_lyrics.shape
money_crosstab = pandas.crosstab(money_lyrics.album_release_date, money_lyrics.theme)
trump_crosstab.plot.bar(stacked=True)


# In[10]:


lyric_data.candidate.unique()


# In[16]:


trump_negative_lyrics = lyric_data[(lyric_data["candidate"] == "Donald Trump") & (lyric_data["sentiment"] == 'negative')]
trump_negative_lyrics.shape


# In[17]:


trump_positive_lyrics = lyric_data[(lyric_data["candidate"] == "Donald Trump") & (lyric_data["sentiment"] == 'positive')]
trump_positive_lyrics.shape


# In[ ]:




