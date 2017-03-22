import numpy as np
import tensorflow as tf

import tf_gmm_tools


DIMENSIONS = 2
COMPONENTS = 10
NUM_POINTS = 10000

BATCH_SIZE = 100
TRAINING_STEPS = 100
LEARNING_RATE = 0.001


# PREPARING DATA

# generating DATA_POINTS points from a GMM with COMPONENTS components
data, true_means, true_covariances, true_weights, responsibilities = tf_gmm_tools.generate_gmm_data(
    NUM_POINTS, COMPONENTS, DIMENSIONS, seed=10, diagonal=True)


# BUILDING COMPUTATIONAL GRAPH

# model inputs: data points and prior parameters
input = tf.placeholder(tf.float64, [None, DIMENSIONS])

# constants: ln(2 * PI) * D
ln2piD = tf.constant(np.log(2 * np.pi) * DIMENSIONS, dtype=tf.float64)

# computing input statistics
dim_means = tf.reduce_mean(input, 0)
dim_distances = tf.squared_difference(input, tf.expand_dims(dim_means, 0))
dim_variances = tf.reduce_sum(dim_distances, 0) / tf.cast(tf.shape(input)[0], tf.float64)
avg_dim_variance = tf.cast(tf.reduce_sum(dim_variances) / COMPONENTS / DIMENSIONS, tf.float64)

# default initial values of the variables
initial_means = tf.placeholder_with_default(
    tf.gather(input, tf.squeeze(tf.multinomial(tf.ones([1, tf.shape(input)[0]]), COMPONENTS))),
    shape=[COMPONENTS, DIMENSIONS]
)
initial_covariances = tf.placeholder_with_default(
    tf.cast(tf.ones([COMPONENTS, DIMENSIONS]), tf.float64) * avg_dim_variance,
    shape=[COMPONENTS, DIMENSIONS]
)
initial_weights = tf.placeholder_with_default(
    tf.cast(tf.fill([COMPONENTS], 1.0 / COMPONENTS), tf.float64),
    shape=[COMPONENTS]
)

# trainable variables: component means, covariances, and weights
means = tf.Variable(initial_means, dtype=tf.float64)
covariances = tf.Variable(initial_covariances, dtype=tf.float64)
weights = tf.Variable(initial_weights, dtype=tf.float64)

# E-step: recomputing responsibilities with respect to the current parameter values
sq_distances = tf.squared_difference(tf.expand_dims(input, 0), tf.expand_dims(means, 1))
sum_sq_dist_times_inv_var = tf.reduce_sum(sq_distances / tf.expand_dims(covariances, 1), 2)
log_coefficients = tf.expand_dims(ln2piD + tf.reduce_sum(tf.log(covariances), 1), 1)
log_components = -0.5 * (log_coefficients + sum_sq_dist_times_inv_var)
log_weighted = log_components + tf.expand_dims(tf.log(weights), 1)
log_shift = tf.expand_dims(tf.reduce_max(log_weighted, 0), 0)
exp_log_shifted = tf.exp(log_weighted - log_shift)
exp_log_shifted_sum = tf.reduce_sum(exp_log_shifted, 0)

# log-likelihood: objective function being maximized up to a TOLERANCE delta
log_likelihood = tf.reduce_sum(tf.log(exp_log_shifted_sum)) + tf.reduce_sum(log_shift)
mean_log_likelihood = log_likelihood / tf.cast(tf.shape(input)[0] * tf.shape(input)[1], tf.float64)

# training algorithm: Adam with configurable learning rate
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(-log_likelihood)

# re-normalizing weights to sum up to 1.0 again (with softmax)
re_normalize_weights = weights.assign(tf.nn.softmax(weights))


# RUNNING COMPUTATIONAL GRAPH

with tf.Session() as sess:
    # initializing trainable variables
    sess.run(
        tf.global_variables_initializer(),
        feed_dict={
            input: data,
            initial_means: data[:10],
            # initial_covariances: true_covariances,
            # initial_weights: true_weights
        }
    )

    previous_likelihood = -np.inf

    # training loop
    for step in range(TRAINING_STEPS):
        # shuffling the data before each epoch
        np.random.shuffle(data)

        # stochastic batch loop
        for b in range(int(len(data) / BATCH_SIZE)):
            # fetching subsequent batch from the data
            batch = data[b * BATCH_SIZE:(b+1) * BATCH_SIZE]

            # executing a training step on a fetched batch
            sess.run(train_step, feed_dict={input: batch})

            # re-normalizing weights
            sess.run(re_normalize_weights)

        # fetching evaluation information
        current_likelihood = sess.run(
            mean_log_likelihood,
            feed_dict={input: data}
        )

        if step > 0:
            # computing difference between consecutive log-likelihoods
            difference = current_likelihood - previous_likelihood
            print("{0}:\tmean-likelihood {1:.8f}\tdifference {2}".format(
                step, current_likelihood, difference))
        else:
            print("{0}:\tmean-likelihood {1:.8f}".format(
                step, current_likelihood))

        previous_likelihood = current_likelihood

    # fetching final parameter values
    final_means = means.eval(sess)
    final_covariances = covariances.eval(sess)

# plotting data and the obtained GMM
tf_gmm_tools.plot_fitted_data(
    data, final_means, final_covariances,
    true_means, true_covariances
)
