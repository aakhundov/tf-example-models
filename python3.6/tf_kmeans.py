import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm


DIMENSIONS = 2
CLUSTERS = 10
DATA_POINTS = 10000

TRAINING_STEPS = 1000
TOLERANCE = 0


def generate_gmm_data(points, components, dimensions):
    np.random.seed(10)

    c_means = np.random.normal(size=[components, dimensions]) * 10
    c_variances = np.abs(np.random.normal(size=[components, dimensions]))
    c_weights = np.abs(np.random.normal(size=[components]))
    c_weights /= np.sum(c_weights)

    result = np.zeros((points, dimensions), dtype=np.float32)

    for i in range(points):
        comp = np.random.choice(np.array(range(10)), p=c_weights)
        result[i] = np.random.multivariate_normal(
            c_means[comp], np.diag(c_variances[comp])
        )

    np.random.seed()

    return result, c_means, c_variances, c_weights


def plot_raw_data(points):
    plt.plot(points[:, 0], points[:, 1], "b.")
    plt.show()


def plot_clustered_data(points, c_means, c_assignments):
    colors = cm.rainbow(np.linspace(0, 1, CLUSTERS))

    for cluster, color in zip(range(CLUSTERS), colors):
        c_points = points[c_assignments == cluster]
        plt.plot(c_points[:, 0], c_points[:, 1], ".", color=color, zorder=0)
        plt.plot(c_means[cluster, 0], c_means[cluster, 1], ".", color="black", zorder=1)

    plt.show()


# PREPARING DATA

# generating DATA_POINTS points from a GMM with CLUSTERS components
data, true_means, true_variances, true_weights = generate_gmm_data(DATA_POINTS, CLUSTERS, DIMENSIONS)


# BUILDING COMPUTATIONAL GRAPH

# model inputs: generated data points
input = tf.placeholder(tf.float32, [None, DIMENSIONS])

# trainable variables: clusters means
random_point_ids = tf.squeeze(tf.multinomial(tf.ones([1, tf.shape(input)[0]]), CLUSTERS))
means = tf.Variable(tf.gather(input, random_point_ids), dtype=tf.float32)

# E-step: recomputing cluster assignments according to the current means
inputs_ex, means_ex = tf.expand_dims(input, 0), tf.expand_dims(means, 1)
distances = tf.reduce_sum(tf.squared_difference(inputs_ex, means_ex), 2)
assignments = tf.argmin(distances, 0)

# M-step: relocating cluster means according to the computed assignments
sums = tf.unsorted_segment_sum(input, assignments, CLUSTERS)
counts = tf.reduce_sum(tf.one_hot(assignments, CLUSTERS), 0)
means_ = tf.divide(sums, tf.expand_dims(counts, 1))

# distortion measure: sum of squared distances 
# from each point to the closest cluster mean
distortion = tf.reduce_sum(tf.reduce_min(distances, 0))

# updating the means by new values
train_step = means.assign(means_)


# RUNNING COMPUTATIONAL GRAPH

# creating session
sess = tf.InteractiveSession()

# initializing trainable variables
sess.run(tf.global_variables_initializer(), feed_dict={input: data})

previous_assignments = None

# training loop
for step in range(TRAINING_STEPS):
    # executing a training step and
    # fetching evaluation information
    distortion_measure, current_means, current_assignments, _ = sess.run(
        [distortion, means_, assignments, train_step],
        feed_dict={input: data}
    )

    if step > 0:
        # computing the number of re-assignments during the step
        re_assignments = (current_assignments != previous_assignments).sum()
        print("{0}:\tdistortion {1:.2f}\tre-assignments {2}".format(
            step, distortion_measure, re_assignments))

        # stopping if no re-assignments occurred
        if re_assignments <= TOLERANCE:
            break
    else:
        print("{0}:\tdistortion {1:.2f}".format(
            step, distortion_measure))

    previous_assignments = current_assignments

plot_raw_data(data)
plot_clustered_data(data, current_means, current_assignments)
