import numpy as np
import tensorflow as tf


import pandas as pd
df = pd.read_csv("RFM.csv")
df.columns
R = df.drop(['user_id'],axis=1)
full_data_x = R.as_matrix()

#num_points = 100
#dimensions = 3
#points = np.random.uniform(0, 1000, [num_points, dimensions])

def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(full_data_x, dtype=tf.float32), num_epochs=1)

num_clusters = 4
kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters=num_clusters, use_mini_batch=False)

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
print ('cluster centers:', cluster_centers)

# map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn))
for i, point in enumerate(full_data_x):
  cluster_index = cluster_indices[i]
  center = cluster_centers[cluster_index]
  #print ('point:', point, 'is in cluster', cluster_index, 'centered at', center)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(df['Recency_score'])
y = np.array(df['Frequency_score'])
z = np.array(df['Monetary_score'])
ax.scatter(x,y,z, marker="s", c=cluster_indices, s=40, cmap="RdBu")
plt.show()
