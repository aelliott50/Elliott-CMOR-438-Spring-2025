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

4:

