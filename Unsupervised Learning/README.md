# Elliott-CMOR-438-Spring-2025
This directory contains implementations of several Unsupervised Learning algorithms.

1: K-means clustering:
K-means clustering is an unsupervised learning model. The goal of this algorithm is to cluster
a dataset into k different groups. The dataset is unlabeled, as is it with unsupervised learning.
The approach is to define k unique centers in the data, and with each iteration, to adjust the 
location of these centers slightly and reassign data points to centers, in order to make the most
compact clusters as possible. I used the wine dataset from scikit learn for this implementation,
which has 13 features, which describe chemical contents of the wine, and three classes, each being a
different winemaker. Unfortunately, it will not be possible to precisely replicate my results, as
I did not use a randomizing function with a seed when initializing my centers. However, the results
obtained will be very similar.
