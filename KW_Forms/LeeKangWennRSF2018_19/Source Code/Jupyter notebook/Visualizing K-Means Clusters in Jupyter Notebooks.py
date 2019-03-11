
# coding: utf-8

# # RFM Score Clustering

# Import libraries needed for clustering and visualisation

# In[1]:


# Initialize plotting library and functions for 3D scatter plots 
import pandas as pd
import numpy as np
import argparse
import json
import re
import os
import sys
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()
import tensorflow as tf


# Import cleaned data

# In[2]:


df = pd.read_csv("RFM.csv")
df.columns
R = df.drop(['user_id'],axis=1)
full_data_x = R.as_matrix()


# Split data into training set and testing set of
# **test-size** means percentage of data being split for testing

# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(full_data_x,test_size=0.1, random_state=42)


# In[5]:


def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(X_train, dtype=tf.float32), num_epochs=1)


# In[6]:


def input_fn2():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(X_test, dtype=tf.float32), num_epochs=1)


# In[7]:


num_clusters = 4
kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters=num_clusters, use_mini_batch=False)


# In[8]:


# train
num_iterations = 10
previous_centers = None
for _ in range(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  if previous_centers is not None:
    print ('delta:', cluster_centers - previous_centers)
  previous_centers = cluster_centers
  print ('score:', kmeans.score(input_fn))


# In[9]:


print ('cluster centers:', cluster_centers)


# In[10]:


# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(X_train):
  cluster_index = cluster_indices[i]
  center = cluster_centers[cluster_index]
  #print ('point:', point, 'is in cluster', cluster_index, 'centered at', center)


# In[11]:


predicted_cluster = list(kmeans.predict_cluster_index(input_fn2))
for i, point in enumerate(X_test):
  cluster_index = predicted_cluster[i]
  center = cluster_centers[cluster_index]
  #print ('point:', point, 'is in cluster', cluster_index, 'centered at', center)


# In[16]:


print(kmeans.score(input_fn2))


# In[12]:


r = X_train[:,0].tolist()
f = X_train[:,1].tolist()
m = X_train[:,2].tolist()
df2 = pd.DataFrame({'R':r})
df2['F'] = f
df2['M'] = m
df2['y'] = cluster_indices
df2.head(3)


# In[13]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[14]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = X_train[:,0]
y = X_train[:, 1]
z = X_train[:, 2]
ax.scatter(x,y,z, marker="s", c=cluster_indices, s=40, cmap="RdBu")
plt.show()


# In[15]:


# Visualize cluster shapes in 3d.

cluster1=df2.loc[df2['y'] == 0]
cluster2=df2.loc[df2['y'] == 1]
cluster3=df2.loc[df2['y'] == 2]
cluster4=df2.loc[df2['y'] == 3]

scatter1 = dict(
    mode = "markers",
    name = "Cluster 1",
    type = "scatter3d",    
    x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
    marker = dict( size=2, color='green')
)
scatter2 = dict(
    mode = "markers",
    name = "Cluster 2",
    type = "scatter3d",    
    x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
    marker = dict( size=2, color='blue')
)
scatter3 = dict(
    mode = "markers",
    name = "Cluster 3",
    type = "scatter3d",    
    x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
    marker = dict( size=2, color='red')
)
scatter4 = dict(
    mode = "markers",
    name = "Cluster 4",
    type = "scatter3d",    
    x = cluster4.as_matrix()[:,0], y = cluster4.as_matrix()[:,1], z = cluster4.as_matrix()[:,2],
    marker = dict( size=2, color='purple')
)
cluster1 = dict(
    alphahull = 1,
    name = "Cluster 1",
    opacity = .1,
    type = "mesh3d",    
    x = cluster1.as_matrix()[:,0], y = cluster1.as_matrix()[:,1], z = cluster1.as_matrix()[:,2],
    color='green', showscale = True
)
cluster2 = dict(
    alphahull = 1,
    name = "Cluster 2",
    opacity = .1,
    type = "mesh3d",    
    x = cluster2.as_matrix()[:,0], y = cluster2.as_matrix()[:,1], z = cluster2.as_matrix()[:,2],
    color='blue', showscale = True
)
cluster3 = dict(
    alphahull = 1,
    name = "Cluster 3",
    opacity = .1,
    type = "mesh3d",    
    x = cluster3.as_matrix()[:,0], y = cluster3.as_matrix()[:,1], z = cluster3.as_matrix()[:,2],
    color='red', showscale = True
)
cluster4 = dict(
    alphahull = 1,
    name = "Cluster 4",
    opacity = .1,
    type = "mesh3d",    
    x = cluster4.as_matrix()[:,0], y = cluster4.as_matrix()[:,1], z = cluster4.as_matrix()[:,2],
    color='purple', showscale = True
)
layout = dict(
    title = 'Interactive Cluster Shapes in 3D',
    scene = dict(
        xaxis = dict( zeroline=True ),
        yaxis = dict( zeroline=True ),
        zaxis = dict( zeroline=True ),
    )
)
fig = dict( data=[scatter1, scatter2, scatter3, scatter4, cluster1, cluster2, cluster3, cluster4], layout=layout )
# Use py.iplot() for IPython notebook
plotly.offline.iplot(fig, filename='mesh3d_sample')

