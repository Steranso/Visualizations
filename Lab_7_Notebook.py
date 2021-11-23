#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
r = requests.post('https://cdsw00.geo.sciclone.wm.edu/api/altus-ds-1/models/call-model',
                 data = '{"accessKey":"m149rzguxkf56i4pnqsulvkmfx43zu5t", "request":{"timestamp_start":"9/1/2019 0:00", "timestamp_end": "9/2/2019 23:59"}}',
                 headers = {"Content-Type" : "application/json", "host":"cdsw.geo.sciclone.wm.edu"}, verify=False)
dta = pd.read_json(r.json()["response"], orient="index")


# In[2]:


get_ipython().system('conda install -y graphviz python-graphviz    #(Already downloaded dont need to do again)')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import numpy as np

dta_clean = dta.drop(["ApproachCount", "FemaleCount", "MaleCount", "Timestamp", "pdtimes",
                     "CNN1","CNN2","CNN3","CNN4","CNN5","CNN6","CNN7","CNN8","CNN9","CNN11","CNN12"], axis = 1)

y_continuous = dta_clean.pop("PersonCount")

y = (y_continuous > 30).astype(int)
print(y)

X = dta_clean

from sklearn import preprocessing
scalingModel = preprocessing.StandardScaler().fit(X)
X_scaled = scalingModel.transform(X)


# In[4]:


colors = ["red","blue"]
fig = plt.figure(figsize=(8,8))
plt.scatter(X["ImgBright"].values,
           X["CNN10"].values,
           c=y,
           label = "High Foot Traffic",
           s=4,
           cmap = cm.ListedColormap(colors))
plt.grid(True)
plt.legend()
plt.ylabel("Values From Convolutional Neural Network (Feature 10)")
plt.xlabel("Image Brightness")


# In[5]:


from sklearn.neighbors import KNeighborsClassifier
NNeighbors = KNeighborsClassifier(n_neighbors=10).fit(X_scaled,y)

xx1, xx2 = np.meshgrid(np.arange(0,30,5), np.arange(0, 100, 10))
XX = np.stack([xx1.ravel(), xx2.ravel()], axis=1)
XX_grid_scaled = scalingModel.transform(XX)

Z = NNeighbors.predict_proba(XX_grid_scaled)[:, 1]

Z = Z.reshape(len(xx1),len(xx2[0]))

colors = ["red","blue"]

fig = plt.figure(figsize=(8,8))
plt.contourf(xx1, xx2, Z, cmap=plt.cm.RdBu, alpha=.8)
plt.scatter(X["ImgBright"].values, X["CNN10"].values, c=y, label = "High Foot Traffic",s=4,cmap = cm.ListedColormap(colors))
plt.xlim(0,25)
plt.ylim(0,170)
plt.grid(True)
plt.legend()
plt.ylabel("Values From Convolutional Neural Network (Feature 10)")
plt.xlabel("Image Brightness")


#Why the fuck is it onl blue up until a vule of 90 for CNN instead of the whole way?


# In[6]:


def viz_classifier(model):
    xx1, xx2 = np.meshgrid(np.arange(0,30,5), np.arange(0, 100, 10))
    XX = np.stack([xx1.ravel(), xx2.ravel()], axis=1)
    global scalingModel
    XX_grid_scaled = scalingModel.transform(XX)

    Z = model.predict_proba(XX_grid_scaled)[:, 1]

    Z = Z.reshape(len(xx1),len(xx2[0]))

    colors = ["red","blue"]

    fig = plt.figure(figsize=(8,8))
    plt.contourf(xx1, xx2, Z, cmap=plt.cm.RdBu, alpha=.8)
    plt.scatter(X["ImgBright"].values, X["CNN10"].values, c=y, label = "High Foot Traffic",s=4,cmap = cm.ListedColormap(colors))
    plt.xlim(0,25)
    plt.ylim(0,170)
    plt.grid(True)
    plt.legend()
    plt.ylabel("Values From Convolutional Neural Network (Feature 10)")
    plt.xlabel("Image Brightness")
    plt.title(str(model))

viz_classifier(NNeighbors)


# In[7]:


from sklearn.svm import SVC
linear_svm = SVC(kernel = "linear", C=0.025, probability=True).fit(X_scaled,y)
viz_classifier(linear_svm)


# In[8]:


from sklearn.svm import SVC
radial_svm = SVC(gamma=2, C=1, probability=True).fit(X_scaled,y)
viz_classifier(radial_svm)


# In[9]:


from sklearn.tree import DecisionTreeClassifier
dTree = DecisionTreeClassifier(random_state = 1693, max_depth = 3).fit(X_scaled, y)
viz_classifier(dTree)

from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import sklearn.tree

graph = Source(sklearn.tree.export_graphviz(dTree, out_file=None, feature_names=X.columns,
                                            class_names=["Low Foot Traffic","High Foot Traffic"], filled = True))
display(SVG(graph.pipe(format='svg')))


# In[24]:


from sklearn.ensemble import RandomForestClassifier
rForest = RandomForestClassifier(n_estimators=1000, random_state = 1693, max_depth = 3).fit(X_scaled, y)
viz_classifier(rForest)


# In[25]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(alpha=1, max_iter=1000).fit(X_scaled,y)
viz_classifier(mlp)


# In[26]:


from sklearn.metrics import accuracy_score
import ipywidgets
from ipywidgets import interact, interact_manual

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import numpy as np

dta_clean = dta.drop(["ApproachCount", "FemaleCount", "MaleCount", "Timestamp", "pdtimes",
                     "CNN1","CNN2","CNN3","CNN4","CNN5","CNN6","CNN7","CNN8","CNN9","CNN11","CNN12"
                     ], axis = 1)

y_continuous = dta_clean.pop("PersonCount")

y = (y_continuous > 30).astype(int)

X = dta_clean

from sklearn import preprocessing
scalingModel = preprocessing.StandardScaler().fit(X)
X_scaled = scalingModel.transform(X)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

@interact_manual(
selection = ipywidgets.Dropdown(options=
                               ["Nearest Neighbor",
                               "Radial SVM",
                               "Decision Tree",
                               "Neural Network",
                               "Random Forest"],
                               value = "Nearest Neighbor",description = "Model"))

def viz_classifier(selection):
    if (selection == "Nearest Neighbor"):
        model = KNeighborsClassifier(n_neighbors=10).fit(X_scaled,y)
    if (selection == "Radial SVM"):
        model = SVC(gamma=2, C=1, probability=True).fit(X_scaled,y)
    if (selection == "Decision Tree"):
        model = DecisionTreeClassifier(random_state = 1693, max_depth = 3).fit(X_scaled, y)
    if (selection == "Neural Network"):
        model = MLPClassifier(alpha=1, max_iter=1000).fit(X_scaled,y)
    if (selection == "Random Forest"):
        model = RandomForestClassifier(n_estimators=1000, random_state = 1693, max_depth = 3).fit(X_scaled, y)
        
    xx1, xx2 = np.meshgrid(np.arange(0,30,5), np.arange(0, 100, 10))

    #stack our dummy data for the grid
    XX = np.stack([xx1.ravel(), xx2.ravel()], axis=1)
    global scalingModel
    XX_grid_scaled = scalingModel.transform(XX)

    Z = model.predict_proba(XX_grid_scaled)[:, 1]
    Z = Z.reshape(len(xx1),len(xx2[0]))

    colors = ["red","blue"]
    fig = plt.figure(figsize=(8,8))
    plt.contourf(xx1, xx2, Z, cmap=plt.cm.RdBu, alpha=.8)
    plt.scatter(X["ImgBright"].values, X["CNN10"].values, c=y, label = "High Foot Traffic",s=4,cmap = cm.ListedColormap(colors))
    plt.xlim(0,25)
    plt.ylim(0,170)
    plt.grid(True)
    plt.legend()
    plt.ylabel("Values From Convolutional Neural Network (Feature 10)")
    plt.xlabel("Image Brightness")
    plt.title(str(selection) + "Accuracy: " + str(accuracy_score(y, model.predict(X_scaled))))


# In[17]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import time
requests.packages.urllib3.disable_warnings()

cur_time = pd.Timestamp("9/1/2019 0:00")

pred_col_names = ["Timestamp", "trulyBusy", "estimateBusy", "estimatedProbabilityBusy"]
pred_df = pd.DataFrame(columns=pred_col_names)

fig, ax = plt.subplots()
fig.set_size_inches(11,8)
fig.show()
fig.canvas.draw()
legend_draw=0
first_run=0
end_time = pd.Timestamp("9/1/2019 23:59")
time_step_min = 30
ax.set_yticks([0.25,0.75])
ax.set_yticklabels(["", "Observed Busy"])
ax2 = ax.twinx()
ax2._sharex = ax

timestamp = 0

while (cur_time < end_time):
    r = requests.post('https://cdsw00.geo.sciclone.wm.edu/api/altus-ds-1/models/call-model',
                 data = '{"accessKey":"m149rzguxkf56i4pnqsulvkmfx43zu5t", "request":{"timestamp_start":"'+ str(cur_time) +'", "timestamp_end":"'+ str(cur_time) + '"}}',
                 headers = {"Content-Type" : "application/json", "host":"cdsw.geo.sciclone.wm.edu"}, verify=False)
    dta = pd.read_json(r.json()["response"], orient="index")
    
    timestamp = timestamp + time_step_min
    
    if (first_run==0):
        dta["Minute"] = dta["Timestamp"].dt.minute + dta["Timestamp"].dt.hour * 60
        timestamp = dta["Minute"].iloc[0].astype(float)
        first_run = 1
        
    trulyBusy = (dta["PersonCount"].iloc[0] > 30).astype(int)
    
    dta_clean = dta.drop(["ApproachCount", "FemaleCount", "MaleCount", "Timestamp", "pdtimes",
                     "CNN1","CNN2","CNN3","CNN4","CNN5","CNN6","CNN7","CNN8","CNN9","CNN11","CNN12"
                     ], axis = 1)
    
    scaled_X = scalingModel.transform(dta_clean[["CNN10", "ImgBright"]].values.reshape(1,-1))
    
    estimateBusy = NNeighbors.predict(scaled_X)[0]
    prob = NNeighbors.predict_proba(scaled_X)[0]
    
    pred_df.loc[len(pred_df)] = [timestamp,trulyBusy, estimateBusy, prob[1]]
    
    plt.bar(pred_df["Timestamp"], height=pred_df["trulyBusy"].values, width=30,color="blue")
    
    
    plt.xticks(np.arange(len(pred_df)), pred_df["Timestamp"])
    
    if(len(pred_df)<25):
        ax.xaxis.set_major_locator(MultipleLocator(30))
    if(len(pred_df)>25):
        ax.xaxis.set_major_locator(MultipleLocator(120))
    if(len(pred_df)>100):
        ax.xaxis.set_major_locator(MultipleLocator(360))    
        
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(color="w",linestyle="-",zorder=1.0)
    
    ax2.plot(pred_df["Timestamp"], pred_df["estimatedProbabilityBusy"], color="red")
    fig.canvas.draw()
    
    cur_time = cur_time + pd.Timedelta(minutes=time_step_min)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




