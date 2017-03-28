import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


LOCAL_FOLDER = "MNIST_data/"

IMAGE_PIXELS = 784
NUM_CLASSES = 10

HIDDEN1_UNITS = 500
HIDDEN2_UNITS = 300

LEARNING_RATE = 1e-4
TRAINING_STEPS = 2000
BATCH_SIZE = 100


def dense_layer(x, in_dim, out_dim, layer_name, act):
    """Creates a single densely connected layer of a NN"""
    with tf.name_scope(layer_name):
        # layer weights corresponding to the input / output dimensions
        weights = tf.Variable(
            tf.truncated_normal(
                [in_dim, out_dim], 
                stddev=1.0 / tf.sqrt(float(out_dim))
            ), name="weights"
        )

        # layer biases corresponding to output dimension
        biases = tf.Variable(tf.zeros([out_dim]), name="biases")

        # layer activations applied to Wx+b
        layer = act(tf.matmul(x, weights) + biases, name="activations")

    return layer


# PREPARING DATA

# downloading (on first run) and extracting MNIST data
data = input_data.read_data_sets(LOCAL_FOLDER, one_hot=True, validation_size=0)


# BUILDING COMPUTATIONAL GRAPH

# model inputs: input pixels and targets
input = tf.placeholder(tf.float32, [None, IMAGE_PIXELS], name="input")
targets = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="targets")

# network layers: two hidden and one output
hidden1 = dense_layer(input, IMAGE_PIXELS, HIDDEN1_UNITS, "hidden1", act=tf.nn.relu)
hidden2 = dense_layer(hidden1, HIDDEN1_UNITS, HIDDEN2_UNITS, "hidden2", act=tf.nn.relu)
output = dense_layer(hidden2, HIDDEN2_UNITS, NUM_CLASSES, "output", act=tf.identity)

# loss function: cross-entropy with built-in 
# (stable) computation of softmax from logits
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=targets, logits=output
    )
)

# training algorithm: Adam with configurable learning rate
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

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

    if step % 100 == 0:
        # reporting current accuracy of the model on every 100th batch
        batch_accuracy = sess.run(accuracy, feed_dict={input: batch_xs, targets: batch_ys})
        print("{0}:\tbatch accuracy {1:.2f}".format(step, batch_accuracy))

    # running the training step with the fetched batch
    sess.run(train_step, feed_dict={input: batch_xs, targets: batch_ys})

# evaluating model prediction accuracy of the model on the test set
test_accuracy = sess.run(accuracy, feed_dict={input: data.test.images, targets: data.test.labels})


print("-------------------------------------------------")
print("Test set accuracy: {0:.4f}".format(test_accuracy))
