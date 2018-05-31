# TensorFlow Example Models
TensorFlow-based implementations of several Machine Learning models (first three - Logistic Regresion, MLP, and CNN - are heavily inspired by [TensorFlow v1.3 tutorials](https://www.tensorflow.org/versions/r1.3/tutorials/)). The [models](models) folder contains simple implementations of:

* [Logistic Regression](models/tf_logreg.py)
* [Multi-Layer Perceptron](models/tf_mlp.py)
* [Convolutional Neural Network](models/tf_cnn.py)
* [K-Means Clustering](models/tf_kmeans.py)
* [Gaussian Mixture Model](models/tf_kmeans.py) (with EM)

The [gmm](gmm) folder contains more elaborate versions of a Gaussian Mixture Model implementation trained by means of Expectation Maximization algorithm (with diagonal covariance, full covariance, gradient-based, etc.). The [gmm/struct](gmm/struct) folder contains initial attempts to decompose the GMM implementation into a coherent set of classes.
