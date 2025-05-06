# Elliott-CMOR-438-Spring-2025
This directory contains implementations of several Unsupervised Learning algorithms.

1: K-means clustering:
K-means clustering is an unsupervised learning model. The goal of this algorithm is to cluster
a dataset into k different groups. The dataset is unlabeled, as is it with unsupervised learning.
The approach is to define k unique centers in the data, and with each iteration, to adjust the 
location of these centers slightly and reassign data points to centers, in order to make the most
compact clusters as possible. I used the wine dataset from scikit learn for this implementation,
which has 13 features, which describe chemical contents of the wine, and three classes, each being a
different winemaker. Then, clustering takes place without the knowledge of the true labels, and at
the end the clusters are compared to the true label clusters. Unfortunately, it will not be possible
to precisely replicate my results, as I did not use a randomizing function with a seed when
initializing my centers. However, the results obtained will be very similar.

2: DBSCAN:
DBSCAN (Density Based Spatial Clustering of Applications with Noise), is an unsupervised learning
algorithm that works similarly to K-means clustering, but with the key difference that the number
of clusters to find is not a parameter that is passed to the model at instantiation. Instead, 
parameters relating to how strongly the algorithm should try to place points into groups instead of
marking them as outliers. The algorithm then groups clusters, but only those that are sufficiently
dense, according to the parameters. As for the K-means clustering, the win dataset was used. To 
replicate my results, just mimicking the implementation in the code

3: Principal Component Analysis:
Principal Component Analysis (PCA) is a very useful unsupervised learning algorithm that aims to lower
the dimensionality of a very large problem. To do so, it selects the most impactful features, which
can also be known as principal components, and projects the data to only retain the most necessary 
principal components. These components are computed via Singular Value Decomposition (SVD). For this 
implementation, the wine dataset was again used. The same results can be achieved by simply mimicking
tht code.

4: Singular Value Decomposition Image Compression:
In Singular Value Decomposition for Image Compression, the goal is to reduce the amount of storage
to an acceptable amount while still retaining as much of the important features of the image as
possible. Singular value decomposition for image decompression works by expressing the full image
of pixel data by only the most important singular vectors. This way, the matrix of pixel data no
longer has to be full rank, and carries less information while retaining as much of the key features
as possible. For this example, I used the fashion_mnist dataset, which was previously used in the 
k_nearest neighbors implementation. This dataset contains grayscale 28x28 pixel images of clothing.
Through singular value decomposition, the entire matrix of pixel information can be summarized in
as little as 5 singular vectors while still retaining most of what makes it identifiable. To replicate
my result, test the very first image from the set, which is a primarily light colored ankle boot.
