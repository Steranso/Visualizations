#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#download from https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset

df = pd.read_csv("https://thedatadoctor.io/wp-content/uploads/2019/10/london_merged.csv")

np.random.seed(1693)


# In[3]:


df.head()


# In[4]:


df["hour_of_day"] = pd.to_datetime(df["timestamp"]).dt.hour
df.pop('timestamp')
df.head()


# In[5]:


viz_df = df.select_dtypes(include=[np.number])
ax = pd.plotting.scatter_matrix(viz_df,alpha=0.75, figsize=[20,20], diagonal='kde')
plt.suptitle('Diagnostics')
plt.show()


# In[6]:


df = df.select_dtypes(include=[np.number])

from sklearn import preprocessing
scalingModel = preprocessing.StandardScaler().fit(df.values)
X_scaled = scalingModel.transform(df.values)

from sklearn.decomposition import PCA
df_pca = PCA(n_components = 2)
df_pca.fit(X_scaled)
X = pd.DataFrame(data=df_pca.transform(X_scaled), columns=["PC1","PC2"], index=df.index)
X_all = pd.concat([X,df], axis = 1)
X_all


# In[7]:


X_all.plot.scatter(x="PC1",y="PC2", s=0.15)

X_Sample = X_all.sample(n=1000,random_state=1693)
X_PCA = X_Sample[["PC1","PC2"]]
X_Sample.pop("PC1")
X_Sample.pop("PC2")
X_PCA.plot.scatter(x="PC1",y="PC2",s=0.5)


# In[8]:


from sklearn.cluster import SpectralClustering
spectralCluster = SpectralClustering(n_clusters=3,affinity="nearest_neighbors",n_neighbors =10).fit_predict(X_PCA)

fig,ax = plt.subplots()
plt.figure(figsize=(10,10))
scatter = ax.scatter(X_PCA['PC1'],X_PCA['PC2'],c=spectralCluster,label=spectralCluster.tolist()[0],s=0.5)
legend = ax.legend(*scatter.legend_elements(),loc="lower left",title="Classes")
ax.add_artist(legend)
plt.show()


# In[9]:


from sklearn.cluster import SpectralClustering
spectralCluster = SpectralClustering(n_clusters=3,affinity="nearest_neighbors",n_neighbors =10).fit_predict(X_PCA)
spectralresults = pd.DataFrame(data=spectralCluster,columns=["Cluster"],index=X_PCA.index)

from sklearn.tree import DecisionTreeClassifier
dTree = DecisionTreeClassifier(random_state = 1500, max_depth = 2).fit(X_Sample.values, spectralCluster)

from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import sklearn.tree

graph = Source(sklearn.tree.export_graphviz(dTree,
                                            out_file=None,
                                            feature_names=X_Sample.columns,
                                            class_names=True, filled = True))
display(SVG(graph.pipe(format='svg')))


# In[10]:


from sklearn.cluster import KMeans
kMeans = KMeans(n_clusters=3,random_state=1693).fit_predict(X_PCA)
plt.figure(figsize=(10,10))

fig,ax = plt.subplots()
plt.figure(figsize=(10,10))
scatter = ax.scatter(X_PCA['PC1'],X_PCA['PC2'],c=kMeans,label=kMeans.tolist()[0],s=0.5)
legend1 = ax.legend(*scatter.legend_elements(),loc="lower left",title="Classes")
ax.add_artist(legend1)
plt.show()


# In[11]:


from sklearn.cluster import MeanShift
ms = MeanShift(bandwidth = 1.379).fit_predict(X_PCA)


fig,ax = plt.subplots()
plt.figure(figsize=(10,10))
scatter = ax.scatter(X_PCA['PC1'],X_PCA['PC2'],c=ms,label=ms.tolist()[0],s=0.5)
legend1 = ax.legend(*scatter.legend_elements(),loc="lower left",title="Classes")
ax.add_artist(legend1)
plt.show()


# In[12]:


from sklearn.cluster import AgglomerativeClustering
agglom = AgglomerativeClustering(n_clusters=3).fit_predict(X_PCA)


fig,ax = plt.subplots()
plt.figure(figsize=(10,10))
scatter = ax.scatter(X_PCA['PC1'],X_PCA['PC2'],c=agglom,label=agglom.tolist()[0],s=0.5)
legend1 = ax.legend(*scatter.legend_elements(),loc="lower left",title="Classes")
ax.add_artist(legend1)
plt.show()


# In[13]:


from sklearn.cluster import DBSCAN
dbscan = DBSCAN().fit_predict(X_PCA)

fig,ax = plt.subplots()
plt.figure(figsize=(10,10))
scatter = ax.scatter(X_PCA['PC1'],X_PCA['PC2'],c=dbscan,label=dbscan.tolist()[0],s=0.5)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left",title="Classes")
ax.add_artist(legend1)
plt.show()


# In[14]:


from sklearn.cluster import OPTICS
opt = OPTICS(min_cluster_size=0.5).fit_predict(X_PCA)

fig,ax = plt.subplots()
plt.figure(figsize=(10,10))
scatter = ax.scatter(X_PCA['PC1'],X_PCA['PC2'],c=opt,label=opt.tolist()[0],s=0.5)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left",title="Classes")
ax.add_artist(legend1)
plt.show()


# In[15]:


from sklearn.cluster import Birch
birch = Birch().fit_predict(X_PCA)

fig,ax = plt.subplots()
plt.figure(figsize=(10,10))
scatter = ax.scatter(X_PCA['PC1'],X_PCA['PC2'],c=birch,label=birch.tolist()[0],s=0.5)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left",title="Classes")
ax.add_artist(legend1)
plt.show()


# In[16]:


#Metrics to contrast strength of clustering
#Davies-Bouldin is a good choice if you care about *seperation*
#between clusters. This is generally representative
#of visual breaks the human eye might percieve as difference.
#0 is the best score. Bigger is worse.
from sklearn import metrics
print("=====================")
print("DB Scores:")
print("=====================")
print("Optics:")
print(metrics.davies_bouldin_score(X_PCA,opt))
print("")
print("DB Scan:")
print(metrics.davies_bouldin_score(X_PCA,dbscan))
print("")
print("AHC:")
print(metrics.davies_bouldin_score(X_PCA,agglom))
print("")
print("k Means:")
print(metrics.davies_bouldin_score(X_PCA,kMeans))
print("")
print("Spectral Clustering:")
print(metrics.davies_bouldin_score(X_PCA,spectralCluster))
print("")
print("Mean Shift:")
print(metrics.davies_bouldin_score(X_PCA,ms))
print("")


# In[17]:


X_all_foranalysis = X_all.drop(["PC1","PC2"], axis=1)
X_all_foranalysis

from sklearn.cluster import SpectralClustering
spectralCluster = SpectralClustering(n_clusters=3,affinity="nearest_neighbors",n_neighbors =10).fit_predict(X_all_foranalysis)


# In[18]:


fig,ax = plt.subplots()
plt.figure(figsize=(20,20))
scatter = ax.scatter(X_all['PC1'],X_all['PC2'],c=spectralCluster,label=spectralCluster.tolist()[0],s=0.5)
legend1 = ax.legend(*scatter.legend_elements(),loc="lower left",title="Classes")
ax.add_artist(legend1)
plt.show()


# In[19]:


from sklearn.tree import DecisionTreeClassifier
dTree = DecisionTreeClassifier(random_state = 1693, max_depth = 2).fit(X_all_foranalysis.values, spectralCluster)

from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import sklearn.tree

graph = Source(sklearn.tree.export_graphviz(dTree,
                                            out_file=None,
                                            feature_names=X_Sample.columns,
                                            class_names=True, filled = True))
display(SVG(graph.pipe(format='svg')))


# In[20]:


Busy_not_X = X_all.drop(["PC1","PC2"], axis=1)

#Class 1 - Not Busy
c1 = (Busy_not_X["cnt"]<929.5).astype(int)*1

#Class 2 - Average/Medium Level of Bike Rentals
c2 = ((Busy_not_X["cnt"]>929.5).astype(int) & (Busy_not_X["cnt"]<2926.0).astype(int))*2

#Class 3 - Very Busy
c3 = (Busy_not_X["cnt"]>2926.0).astype(int)*3

y = c1 + c2 +c3

Busy_not_X.pop("cnt")

from sklearn.tree import DecisionTreeClassifier
analysisExampleTree = DecisionTreeClassifier(random_state = 1693, max_depth = 3).fit(Busy_not_X.values, y)

from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import sklearn.tree

graph = Source(sklearn.tree.export_graphviz(analysisExampleTree,
                                            out_file=None,
                                            feature_names=Busy_not_X.columns,
                                            #class_names=True,
                                            filled = True))
display(SVG(graph.pipe(format='svg')))


# In[21]:


y


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


#for question 7
from sklearn.cluster import SpectralClustering
spectralCluster = SpectralClustering(affinity="nearest_neighbors",n_neighbors =1000).fit_predict(X_PCA)

fig,ax = plt.subplots()
plt.figure(figsize=(10,10))
scatter = ax.scatter(X_PCA['PC1'],X_PCA['PC2'],c=spectralCluster,label=spectralCluster.tolist()[0],s=0.5)
legend = ax.legend(*scatter.legend_elements(),loc="lower left",title="Classes")
ax.add_artist(legend)
plt.show()


# In[ ]:




