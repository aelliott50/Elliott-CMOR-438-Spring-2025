# Elliott-CMOR-438-Spring-2025
This directory contains implementations of several Supervised Learning algorithms.

1: Perceptron:
The perceptron algorithm is a single neuron model, used for the purpose of binary classification.
For this implementation, I define a perceptron class. When a Perceptron object is instantiated, it
is passed a learning rate and a number of epochs to iterate over. Then, the weights vector is
randomly initialized. In this model, a sign activation function is used, meaning it is equal to 1
for positive input and -1 for negative input. At each epoch, the weight assigned to each feature is
adjusted to push the weights towards values where every data point can be predicted correctly. This
works by adjusting weights each time an incorrect prediction is made. The result after enough
iterations is a linear separation of data into two classes. For this implementation, a breast cancer
data set is used, where there are 30 feature measurements for each of the 579 instances, with each
instance being either malignant or benign. Thus, the goal is to figure out if there is some way to
linearly separate the data into malignant and benign classes. In other words, we want to try to find
a linear combination of the features so that if the result is positive, the cancer is malignant, and
if the result is negative, the cancer is benign. In this case, no such separation is present.
To reproduce my results, use the breast cancer dataset from scikit learn, and focus on the mean 
radius and mean texture columns for analysis. When modeling just the two features, use a learning
rate of 0.0001, and when modeling all 30 features, use a step size of 0.0000001. While I ran these
over 100000 epochs, it should produce very similar results to run only about 10000 epochs. It is
also possible that the step sizes could be increased slightly for faster convergence.

2: Linear Regression:
The linear regression algorithm is a single neuron model, used for the purpose of predicting a
value given a feature vector. So, we want to be able to take an input feature vector and predict
the target function via a linear combination of the components of the feature vector. This works by
iteratively minimizing the mean squared error of our prediction, which is a mapping of components
of the feature vector to weights. The minimization is done via stochastic gradient descent, where
at each epoch, we calculate the gradient of the MSE function with respect to one data point at a time
, and cycling through all the data points. This way, the gradient computation is less computationally
expensive, and optimal convergence is still guaranteed. The data set I used for this implementation 
is the diabetes dataset from scikit learn. It has ten features, such as sex, age, BMI, and average
blood pressure (bp). The target value is some quantitative measure of how far the diabetes has
progressed one year after the baseline. So, the goal is to find the linear combination of features
that best predicts the diabetes progression in the next year. To replicate my results, simply
use the same diabetes dataset, and use a step size of 0.005.

3: Logistic Regression:
The logistic regression algorithm is a single neuron model, used for the purpose of binary
classification, just like the perceptron. The difference is that instead of predicting which
class a given instance falls into, it predicts a probability that the instance falls into a class.
This works by using the sigmoid function instead of the sign function as the activation function.
This choice of function is good for this application because for very negative values of z, the
prediction is asymptotically 0, for very positive values of z, the prediction is asymptotically 1,
and for z values near 0, a smooth transition from 0 to 1 occurs. The algorithm employs stochastic
gradient descent, just like in linear regression. To make things easier, the algorithm uses the
cross entropy loss function, which is the logarithm of the sum of the predicted probabilities. This
is because we want to be as close as possible to certain about the labels. Using these functions
allows the gradient to be the same as with the perceptron, just the prediction error multiplied
by the feature vector. Now, at each epoch we calculate the gradient with respect to one data point
at a time. I used the same data set as the perceptron, the breast cancer dataset from scikit learn.
This was so that these two approaches to binary classification could be directly compared. This
algorithm and dataset was very sensitive to step size, and to replicate my results use a step size
of 0.000005 over 10000 epochs. Also, I was experiencing errors with the sigmoid and loss function.
It is extremely computationally expensive to calculate sigmoid(z) for |z| large enough. So, I forced
|z| to be less than 100, which sacrifices only minimal accuracy. Also, to avoid the undefined ln(0),
I capped z out at 0.9999999, and 0.0000001.

4: Neural Network:
The neural network algorithm is the first model that is composed of a connected web of neurons. The
network is structured with a series of layers. It works as follows: There is an input layer, which 
has a node for every feature. This feeds into one or more hidden layers, which can have any number
of nodes, but should be chosen to be comparable in size to the input and output layer. The hidden
layers feed into the next hidden layer, and the last hidden layer feeds into the output layer,
which represents the target vector space. As in the logistic regression, the sigmoid activation
function is used. Also, mean squared error is used as the cost function, as we want to minimize
the deviation of prediction vectors to target vectors. With the structure of the neural network
defined, it works by having weight and bias matrices for each inner layer, and by stochastic
gradient descent, iteratively updating the weight and bias matrices to minimize mean squared error.
For this implementation, a data set of japanese hiragana was used. It consists of 70,000 handwritten
hiragana characters, 10,000 of which are randomly separated into a testing set, distinct from the
training set. The images are 28x28 pixels and grayscale. There are just 10 unique characters in the
entire dataset, to ensure that there are plenty of instances of each character to train on. So, with
this data, there are 28*28 = 784 features, so the input layer must have 784 nodes. And because there
are a total of 10 possible characters, the output layer has 10 nodes. To reproduce my results, use a
learning rate of 0.05, and 100 epochs, which takes about 15 minutes to complete training.
I tried 5 potential network architectures. They wre (60, 60), (150, 150), (200, 80), (50, 30), and
(40, 40, 40), with the last one being the only one with 3 hidden layers. I noticed erratic oscillations
for the last one, so I used a learning rate of 0.01. This makes it a little harder to compare directly
to the other architectures.

5: K Nearest Neighbors:
The K nearest neighbors algorithm is the simplest nonparametric model. It works by considering just
the Euclidean distance between feature vectors. Specifically, when applying the algorithm to some point,
we find the Euclidean distance between it and all training feature vectors, and select the k smallest
ones. This algorithm can be used in a regression, where our prediction is just the mean of the target
values of these k vectors, or it can be used in a classification, where our prediction is the class
that appears the most often among the targets of the k points. In this case, the data set is the 
fashion_mnist dataset from tensorflow. It is a collection of 70,000 grayscale images that fall into
ten different categories of clothing items. The images are stored as 28x28 pixel data. 60,000 of the
instances are artificially separated as training instances, and the other 10,000 are for testing.
For the implementation, only 7,000 of the images were used to keep computations more manageable.
To replicate the results, simply slice one tenth of the data set, which importantly is not sorted
by class, and iterate over all the feature vectors in the test set, where for each instance we find
the k nearest neighbors from the training set, and predict the mode of their targets.

6: Decision Trees/Regression Trees:
The Decision Trees algorithm is another nonparametric model. It works by iteratively making linear
partitions of the data, so that each partition accumulates, giving a kind of piecewise linear
decision border. At each iteration, the partition that separates the most points of different classes
is chosen. This branching border can be represented in a tree, where each node is a partition, and the
leaf nodes are the target predictions. In the case of a classification problem, the leaf nodes are
classifications, and it is called a decision tree, and in the case of a regression, the leaf nodes
are predictions for the target variable, where the prediction is typically an average of the target
values of the training instances that fall into that leaf. For the Decision Tree, the scikit learn
breast cancer dataset was used once more. To get the same results, you must use random state 39 with 
40 percent of the data for training when splitting the data into train and test data. This was chosen
because other options had unnatural distributions leading them to not be interesting. For the
instantiation of the decision tree, use random state 42. For the RegressionTree, the cement data
set was used, which is version 7 of the 'Concrete Compressive Strength' from scikit learn. This
data set uses 8 feature variables related to the composition of the cement, and a target variable,
which is a quantitative measure of the strength of the concrete. This makes it a good candidate for
a regression. To replicate the results, use random state 39 and 40% to test when splitting the data, and
use random state 12 when instantiating regressors. 

7: Random Forests:




