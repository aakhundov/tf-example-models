import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


LOCAL_FOLDER = "MNIST_data/"

IMAGE_PIXELS = 784
NUM_CLASSES = 10

LEARNING_RATE = 0.5
TRAINING_STEPS = 1000
BATCH_SIZE = 100


# PREPARING DATA

# downloading (on first run) and extracting MNIST data
data = input_data.read_data_sets(LOCAL_FOLDER, one_hot=True, validation_size=0)


# BUILDING COMPUTATIONAL GRAPH

# model inputs: input pixels and targets
input = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
targets = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# trainable variables: weights and biases
weights = tf.Variable(tf.zeros([IMAGE_PIXELS, NUM_CLASSES]))
biases = tf.Variable(tf.zeros([NUM_CLASSES]))

# model output: logits
output = tf.matmul(input, weights) + biases

# loss function: cross-entropy with built-in
# (stable) computation of softmax from logits
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=targets, logits=output
    )
)

# training algorithm: gradient descent with configurable learning rate
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation operation: ratio of correct predictions
correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# RUNNING COMPUTATIONAL GRAPH

# creating session
sess = tf.InteractiveSession()

# initializing trainable variables
sess.run(tf.global_variables_initializer())

# training loop
for step in range(TRAINING_STEPS):
    # fetching next batch of training data
    batch_xs, batch_ys = data.train.next_batch(BATCH_SIZE)

    # running the training step with the fetched batch
    sess.run(train_step, feed_dict={input: batch_xs, targets: batch_ys})

# evaluating model prediction accuracy of the model on the test set
test_accuracy = sess.run(accuracy, feed_dict={input: data.test.images, targets: data.test.labels})


print("-------------------------------------------------")
print("Test set accuracy: {0:.4f}".format(test_accuracy))
