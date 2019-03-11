# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu Apr 12 13:13:13 2018

@author: kangw
"""

""" K-Means.
Implement K-Means algorithm with TensorFlow, and apply it to classify
handwritten digit images. This example is using the MNIST database of
handwritten digits as training samples (http://yann.lecun.com/exdb/mnist/).
Note: This example requires TensorFlow v1.1.0 or over.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
#from tensorflow.contrib.factorization import KMeans

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import MNIST data
import pandas as pd
df = pd.read_csv("RFM.csv")
df.columns
R = df.drop(['user_id'],axis=1)
full_data_x = R.as_matrix()

# Parameters
num_steps = 500 # Total steps to train
batch_size = 1024 # The number of samples per batch
k = 4 # The number of clusters
num_classes = 1 # The 10 digits
num_features = 3 # Each image is 28x28 pixels

# Input images
X = tf.placeholder(tf.float32, shape=[None, num_features])
# Labels (for assigning a label to a centroid and testing)
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-Means Parameters
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)

# Build KMeans graph
training_graph = kmeans.training_graph()

if len(training_graph) > 6: # Tensorflow 1.4+
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     cluster_centers_var, init_op, train_op) = training_graph
else:
    (all_scores, cluster_idx, scores, cluster_centers_initialized,
     init_op, train_op) = training_graph

cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start TensorFlow session
sess = tf.Session()

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(full_data_x,test_size=0.1, random_state=42)

# Run the initializer
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

# Training
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))

# Assign a label to each centroid
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')
from collections import Counter
count = Counter(idx)
print(count)
# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in count]
labels_map = tf.convert_to_tensor(labels_map)

# Evaluation ops
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# Compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
#test_x, test_y = mnist.test.images, mnist.test.labels
#print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(full_data_x,test_size=0.1, random_state=42)
X1 = full_data_x
X2 = idx.reshape(3256,1)
Xnew = np.hstack((X1, X2))
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: X1, Y: X2}))