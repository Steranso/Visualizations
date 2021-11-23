#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!conda install -y anaconda graphviz
get_ipython().system('conda install -y graphviz python-graphviz    #(Already downloaded dont need to do again)')
#!pip install graphviz


# In[2]:


import requests
import pandas as pd
r = requests.post('https://cdsw00.geo.sciclone.wm.edu/api/altus-ds-1/models/call-model',
                 data = '{"accessKey":"m149rzguxkf56i4pnqsulvkmfx43zu5t", "request":{"timestamp_start":"9/1/2019 0:00", "timestamp_end": "9/2/2019 23:59"}}',
                 headers = {"Content-Type" : "application/json", "host":"cdsw.geo.sciclone.wm.edu"}, verify=False)
dta = pd.read_json(r.json()["response"], orient="index")


# In[3]:


dta


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
dta.head()


# In[5]:


dta["Minute"] = dta["Timestamp"].dt.minute + dta["Timestamp"].dt.hour * 60


# In[6]:


corr = dta.corr()


# In[7]:


corr


# In[8]:


ax = sns.heatmap(corr)


# In[9]:


dta_clean = dta.drop(["ApproachCount", "FemaleCount", "MaleCount", "Timestamp", "pdtimes"], axis = 1)


# In[10]:


ax = sns.heatmap(dta_clean.corr())


# In[11]:


dta_clean = dta.drop(["ApproachCount", "FemaleCount", "MaleCount", "Timestamp", "pdtimes"], axis = 1)
y = dta_clean.pop('PersonCount')
X = dta_clean

from sklearn import preprocessing
scalingModel = preprocessing.StandardScaler().fit(X)
X_scaled = scalingModel.transform(X)

print('Features for Observation 996 before Scaling;')
print(X.loc[996])

print('Features for Observation 996 after Scaling;')
print(X_scaled[996])

X_example = scalingModel.transform(X.iloc[996].values.reshape(1,-1))


# In[12]:


#Lasso Regression
from sklearn import linear_model
lasso_model = linear_model.Lasso(random_state = 1693)
lasso_model.fit(X_scaled, y)
print(list(zip(lasso_model.coef_, X.columns)))
lasso_model.predict(X_example)


# In[13]:


#OMP example (Orthganal Matching Pursuit)
from sklearn.linear_model import OrthogonalMatchingPursuit
orth_MatchingPursuit_model = OrthogonalMatchingPursuit().fit(X_scaled, y)

print(list(zip(orth_MatchingPursuit_model.coef_, X.columns)))
orth_MatchingPursuit_model.predict(X_example)


# In[14]:


from sklearn.linear_model import LinearRegression
linReg = LinearRegression().fit(X_scaled, y)
print(list(zip(linReg.coef_, X.columns)))
linReg.predict(X_example)


# In[15]:


#Ridge Regression
from sklearn.linear_model import Ridge
ridgeReg = Ridge(random_state = 1693)
ridgeReg.fit(X_scaled, y)
print(list(zip(ridgeReg.coef_, X.columns)))
ridgeReg.predict(X_example)


# In[16]:


#Support Vector Regression
from sklearn.svm import LinearSVR
SVR =LinearSVR(random_state = 1693, tol = 1e-5)
SVR.fit(X_scaled, y)
print(list(zip(SVR.coef_, X.columns)))
SVR.predict(X_example)


# In[17]:


from sklearn.linear_model import HuberRegressor
huber = HuberRegressor().fit(X_scaled, y)
print(list(zip(huber.coef_, X.columns)))
huber.predict(X_example)


# In[18]:


from sklearn.tree import DecisionTreeRegressor
treeRegressor = DecisionTreeRegressor(random_state = 1693, max_depth = 3).fit(X_scaled, y)

from IPython.display import SVG
from graphviz import Source
from IPython.display import display
import sklearn.tree

graph = Source(sklearn.tree.export_graphviz(treeRegressor, out_file=None, feature_names=X.columns, class_names=['o','1','2'], filled = True))
display(SVG(graph.pipe(format='svg')))

treeRegressor.predict(X_example)


# In[19]:


from sklearn.neighbors import RadiusNeighborsRegressor
radNeighbors = RadiusNeighborsRegressor(radius=2.5)
radNeighbors.fit(X_scaled, y)
print(radNeighbors.predict(X_example))


# In[20]:


from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(10,3), random_state = 1693, max_iter=2000)
mlp.fit(X_scaled, y)
mlp.predict(X_example)


# In[21]:


from sklearn import linear_model
lars = linear_model.Lars(n_nonzero_coefs=1)
lars.fit(X_scaled, y)
print(list(zip(lars.coef_, X.columns)))
print(lars.predict(X_example))


# In[22]:


from sklearn.linear_model import ElasticNet
elastic = ElasticNet(random_state = 1693)
elastic.fit(X_scaled, y)
print(list(zip(elastic.coef_, X.columns)))
print(elastic.predict(X_example))


# In[23]:


from sklearn.linear_model import PassiveAggressiveRegressor
passiveAgressiveModeling = PassiveAggressiveRegressor(max_iter=100, random_state = 1693, tol = 1e-3)
passiveAgressiveModeling.fit(X_scaled, y)
print(list(zip(passiveAgressiveModeling.coef_, X.columns)))
print(passiveAgressiveModeling.predict(X_example))


# In[24]:


from sklearn.linear_model import RANSACRegressor
RANSAC = RANSACRegressor(random_state = 1693).fit(X_scaled, y)
print(list(zip(RANSAC.estimator_.coef_, X.columns)))
print(RANSAC.predict(X_example))


# In[25]:


col_names = ["ModelType", "MAE_Historic"]
accuracy_df = pd.DataFrame(columns = col_names)

from sklearn.metrics import mean_absolute_error
mae_lasso = mean_absolute_error(y, lasso_model.predict(X_scaled))
accuracy_df.loc[len(accuracy_df)] = ["OrthMatchPursuit", mean_absolute_error(y, orth_MatchingPursuit_model.predict(X_scaled))]
accuracy_df.loc[len(accuracy_df)] = ["linReg", mean_absolute_error(y, linReg.predict(X_scaled))]
accuracy_df.loc[len(accuracy_df)] = ["ridgeReg", mean_absolute_error(y, ridgeReg.predict(X_scaled))]
accuracy_df.loc[len(accuracy_df)] = ["SVR", mean_absolute_error(y, SVR.predict(X_scaled))]
accuracy_df.loc[len(accuracy_df)] = ["huber", mean_absolute_error(y, huber.predict(X_scaled))]
accuracy_df.loc[len(accuracy_df)] = ["treeRegressor", mean_absolute_error(y, treeRegressor.predict(X_scaled))]
accuracy_df.loc[len(accuracy_df)] = ["radNeighbors", mean_absolute_error(y, radNeighbors.predict(X_scaled))]
accuracy_df.loc[len(accuracy_df)] = ["mlp", mean_absolute_error(y, mlp.predict(X_scaled))]
accuracy_df.loc[len(accuracy_df)] = ["lars", mean_absolute_error(y, lars.predict(X_scaled))]
accuracy_df.loc[len(accuracy_df)] = ["elastic", mean_absolute_error(y, elastic.predict(X_scaled))]
accuracy_df.loc[len(accuracy_df)] = ["passiveAgressiveModeling", mean_absolute_error(y, passiveAgressiveModeling.predict(X_scaled))]
accuracy_df.loc[len(accuracy_df)] = ["RANSAC", mean_absolute_error(y, RANSAC.predict(X_scaled))]

ax = sns.barplot(x="MAE_Historic",y="ModelType", data=accuracy_df)


# In[26]:


accuracy_df


# In[27]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import time
requests.packages.urllib3.disable_warnings()

cur_time = pd.Timestamp("9/1/2019 0:00")

pred_col_names = ["Timestamp", "TruePersonCount", "EstimatedPersonCount", "MAE"]
pred_df = pd.DataFrame(columns=pred_col_names)

fig = plt.gcf()
fig.set_size_inches(8,6)
fig.show
fig.canvas.draw()
legend_draw=0

while (cur_time < pd.Timestamp("9/3/2019 23:59")):
    r = requests.post('https://cdsw00.geo.sciclone.wm.edu/api/altus-ds-1/models/call-model',
                 data = '{"accessKey":"m149rzguxkf56i4pnqsulvkmfx43zu5t", "request":{"timestamp_start":"' + str(cur_time)+ '", "timestamp_end":"' + str(cur_time)+ '"}}',
                 headers = {"Content-Type" : "application/json", "host":"cdsw.geo.sciclone.wm.edu"}, verify=False)
    dta = pd.read_json(r.json()["response"], orient="index")
    dta["Minute"] = dta["Timestamp"].dt.minute + dta["Timestamp"].dt.hour * 60
    
    timestamp= dta["Timestamp"].iloc[0]
    truepersoncount = dta["PersonCount"].iloc[0]
    dta_clean = dta.drop(["ApproachCount", "FemaleCount", "MaleCount", "Timestamp", "pdtimes", "PersonCount"], axis = 1)
    
    scaled_X = scalingModel.transform(dta_clean.values.reshape(1,-1))
    
    estimated_person_count = radNeighbors.predict(scaled_X)[0]
    
    cur_time = cur_time + pd.Timedelta(minutes=240)

    MAE = abs(truepersoncount - estimated_person_count)
    
    pred_df.loc[len(pred_df)] = [timestamp, truepersoncount, estimated_person_count, MAE]
    
    plt.plot(pred_df["Timestamp"].values, pred_df["TruePersonCount"].values, "g^", label = "True Number of Persons")
    plt.plot(pred_df["Timestamp"].values, pred_df["EstimatedPersonCount"].values, "b^", label = "Estimated Number of Persons")
    plt.plot(pred_df["Timestamp"].values, pred_df["MAE"].values, "r-.", label = "Mean Absolute Error")
    
    if (legend_draw==0):
        plt.legend()
        legend_draw=1
    

    plt.xticks(rotation=90)
    
    fig.canvas.draw()


# In[ ]:





# In[28]:


#Question 6
from sklearn import linear_model
lars = linear_model.Lars(n_nonzero_coefs=3)
lars.fit(X_scaled, y)
print(list(zip(lars.coef_, X.columns)))
print(lars.predict(X_example))


# In[29]:


#Question 8
X_example2 = scalingModel.transform(X.iloc[900].values.reshape(1,-1))
from sklearn.neighbors import RadiusNeighborsRegressor
radNeighbors = RadiusNeighborsRegressor(radius=2.5)
radNeighbors.fit(X_scaled[900], y)
print(radNeighbors.predict(X_example2))


# In[ ]:





# In[ ]:





# In[ ]:




