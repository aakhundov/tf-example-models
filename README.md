# TensorFlow Example Models
TensorFlow-based implementations of several Machine Learning models (first three - Logistic Regresion, MLP, and CNN - are heavily inspired by [TensorFlow v1.3 tutorials](https://www.tensorflow.org/versions/r1.3/tutorials/)). [models](models) folder contains simple implementations of:

* [Logistic Regression](tf_logreg.py)
* [Multi-Layer Perceptron](tf_mlp.py)
* [Convolutional Neural Network](tf_cnn.py)
* [K-Means Clustering](tf_kmeans.py)
* [Gaussian Mixture Model](tf_kmeans.py) (with EM)

[gmm](gmm) contains more elaborate versions of a Gaussian Mixture Model implementation trained by means of Expectation Maximization algorithm (with diagonal covariance, full covariance, gradient-based, etc.). [gmm/struct](gmm/struct) contains initial attempts to decompose the GMM implementation into a coherent set of classes.
