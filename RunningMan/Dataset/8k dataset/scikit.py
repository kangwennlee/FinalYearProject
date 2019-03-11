# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:13:13 2018

@author: kangw
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans


df = pd.read_csv("RFM.csv")
df.columns

### For the purposes of this example, we store feature data from our
### dataframe `df`, in the `f1` and `f2` arrays. We combine this into
### a feature matrix `X` before entering it into the algorithm.
f1 = df['Recency_score'].values
f2 = df['Frequency_score'].values
f3 = df['Monetary_score'].values

X = df.drop(['user_id'],axis=1).as_matrix()
#print(X)
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X,test_size=0.1, random_state=42)

kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=1000,algorithm='full',tol=0.0001).fit(X_train)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroids)
#print(labels)
count = Counter(labels)
print(count)

print("Group ",kmeans.predict([[5,5,5]]))

y = kmeans.predict(X_test)

estimators = [('k_means_iris_8', KMeans(n_clusters=3)),
              ('k_means_iris_3', KMeans(n_clusters=4)),
              ('k_means_iris_bad_init', KMeans(n_clusters=4, n_init=1,
                                               init='random'))]

fignum = 1
titles = ['3 clusters', '4 clusters', '4 clusters, random init']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X_train)
    labels = est.labels_
    centroids = est.cluster_centers_
    #print(centroids)

    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2],c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Recency score')
    ax.set_ylabel('Frequency score')
    ax.set_zlabel('Monetary score')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y.astype(np.float), edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Recency score')
ax.set_ylabel('Frequency score')
ax.set_zlabel('Monetary score')
ax.set_title('Predicted clusters')
ax.dist = 12

fig.show()
#colors = ["g.","r.","b."]
#for i in range(len(X)):
#    print("coordinate:",X[i],"label:",labels[i])
    #plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=10)
    
#plt.scatter(centroids[:,0],centroids[:,1],marker = "x",s=150,linewidths = 5, zorder = 10)
#plt.show()

#import tensorflow as tf
#from tensorflow.contrib.factorization import KMeans
## Input images
##X = tf.placeholder(tf.float32, shape=[None, num_features])
## Labels (for assigning a label to a centroid and testing)
##Y = tf.placeholder(tf.float32, shape=[None, num_classes])
#
#k = 3
#generation = 25
#data_points = tf.Variable(X)
#print(data_points)
#
## K-Means Parameters
#kmeans = KMeans(inputs=data_points, num_clusters=4, distance_metric='cosine',
#                use_mini_batch=True)
#
## Build KMeans graph
#training_graph = kmeans.training_graph()
#
#if len(training_graph) > 6: # Tensorflow 1.4+
#    (all_scores, cluster_idx, scores, cluster_centers_initialized,
#     cluster_centers_var, init_op, train_op) = training_graph
#else:
#    (all_scores, cluster_idx, scores, cluster_centers_initialized,
#     init_op, train_op) = training_graph
#
#cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
#avg_distance = tf.reduce_mean(scores)
#
## Initialize the variables (i.e. assign their default value)
#init_vars = tf.global_variables_initializer()
#
## Start TensorFlow session
#with tf.Session() as sess:
#    sess.run(init_vars)
#    for step in xrange(iteration_n):
#        [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])
#
## Run the initializer
#sess.run(init_vars, feed_dict={X: full_data_x})
#sess.run(init_op, feed_dict={X: full_data_x})
#
## Training
#for i in range(1, num_steps + 1):
#    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
#                         feed_dict={X: full_data_x})
#    if i % 10 == 0 or i == 1:
#        print("Step %i, Avg Distance: %f" % (i, d))
#
## Assign a label to each centroid
## Count total number of labels per centroid, using the label of each training
## sample to their closest centroid (given by 'idx')
#counts = np.zeros(shape=(k, num_classes))
#for i in range(len(idx)):
#    counts[idx[i]] += mnist.train.labels[i]
## Assign the most frequent label to the centroid
#labels_map = [np.argmax(c) for c in counts]
#labels_map = tf.convert_to_tensor(labels_map)
#
## Evaluation ops
## Lookup: centroid_id -> label
#cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
## Compute accuracy
#correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
#accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
## Test Model
#test_x, test_y = mnist.test.images, mnist.test.labels
#print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))