import numpy as np
import tensorflow as tf

import tf_gmm_tools


DIMENSIONS = 2
COMPONENTS = 10
NUM_POINTS = 10000

TRAINING_STEPS = 1000
TOLERANCE = 1e-6


# PREPARING DATA

# generating DATA_POINTS points from a GMM with COMPONENTS components
data, true_means, true_variances, true_weights, responsibilities = tf_gmm_tools.generate_gmm_data(
    NUM_POINTS, COMPONENTS, DIMENSIONS, seed=10, diagonal=True)


# BUILDING COMPUTATIONAL GRAPH

# model inputs: data points and prior parameters
input = tf.placeholder(tf.float64, [None, DIMENSIONS])
alpha = tf.placeholder_with_default(tf.cast(1.0, tf.float64), [])
beta = tf.placeholder_with_default(tf.cast(1.0, tf.float64), [])

# constants: ln(2 * PI) * D
ln2piD = tf.constant(np.log(2 * np.pi) * DIMENSIONS, dtype=tf.float64)

# computing input statistics
dim_means = tf.reduce_mean(input, 0)
dim_distances = tf.squared_difference(input, tf.expand_dims(dim_means, 0))
dim_variances = tf.reduce_sum(dim_distances, 0) / tf.cast(tf.shape(input)[0], tf.float64)
avg_variance = tf.cast(tf.reduce_sum(dim_variances) / COMPONENTS / DIMENSIONS, tf.float64)

# default initial values of the variables
initial_means = tf.placeholder_with_default(
    tf.gather(input, tf.squeeze(tf.multinomial(tf.ones([1, tf.shape(input)[0]]), COMPONENTS))),
    shape=[COMPONENTS, DIMENSIONS]
)
initial_variances = tf.placeholder_with_default(
    tf.cast(tf.ones([COMPONENTS, DIMENSIONS]), tf.float64) * avg_variance,
    shape=[COMPONENTS, DIMENSIONS]
)
initial_weights = tf.placeholder_with_default(
    tf.cast(tf.fill([COMPONENTS], 1.0 / COMPONENTS), tf.float64),
    shape=[COMPONENTS]
)

# trainable variables: component means, variances, and weights
means = tf.Variable(initial_means, dtype=tf.float64)
variances = tf.Variable(initial_variances, dtype=tf.float64)
weights = tf.Variable(initial_weights, dtype=tf.float64)

# E-step: recomputing responsibilities with respect to the current parameter values
sq_distances = tf.squared_difference(tf.expand_dims(input, 0), tf.expand_dims(means, 1))
sum_sq_dist_times_inv_var = tf.reduce_sum(sq_distances / tf.expand_dims(variances, 1), 2)
log_coefficients = tf.expand_dims(ln2piD + tf.reduce_sum(tf.log(variances), 1), 1)
log_components = -0.5 * (log_coefficients + sum_sq_dist_times_inv_var)
log_weighted = log_components + tf.expand_dims(tf.log(weights), 1)
log_shift = tf.expand_dims(tf.reduce_max(log_weighted, 0), 0)
exp_log_shifted = tf.exp(log_weighted - log_shift)
exp_log_shifted_sum = tf.reduce_sum(exp_log_shifted, 0)
gamma = exp_log_shifted / exp_log_shifted_sum

# M-step: maximizing parameter values with respect to the computed responsibilities
gamma_sum = tf.reduce_sum(gamma, 1)
gamma_weighted = gamma / tf.expand_dims(gamma_sum, 1)
means_ = tf.reduce_sum(tf.expand_dims(input, 0) * tf.expand_dims(gamma_weighted, 2), 1)
distances_ = tf.squared_difference(tf.expand_dims(input, 0), tf.expand_dims(means_, 1))
variances_ = tf.reduce_sum(distances_ * tf.expand_dims(gamma_weighted, 2), 1)
weights_ = gamma_sum / tf.cast(tf.shape(input)[0], tf.float64)

# applying prior to the computed variances
# variances_ *= tf.expand_dims(gamma_sum, 1)
# variances_ += (2.0 * beta)
# variances_ /= tf.expand_dims(gamma_sum + (2.0 * (alpha + 1.0)), 1)

# log-likelihood: objective function being maximized up to a TOLERANCE delta
log_likelihood = tf.reduce_sum(tf.log(exp_log_shifted_sum)) + tf.reduce_sum(log_shift)
mean_log_likelihood = log_likelihood / tf.cast(tf.shape(input)[0] * tf.shape(input)[1], tf.float64)

# assigning new values to the parameters
train_step = tf.group(
    means.assign(means_),
    variances.assign(variances_),
    weights.assign(weights_)
)


# RUNNING COMPUTATIONAL GRAPH

# creating session
sess = tf.InteractiveSession()

# initializing trainable variables
sess.run(
    tf.global_variables_initializer(),
    feed_dict={
        input: data,
        initial_means: true_means,
        # initial_variances: true_variances,
        # initial_weights: true_weights
    }
)

previous_likelihood = -np.inf

# training loop
for step in range(TRAINING_STEPS):
    # executing a training step and
    # fetching evaluation information
    current_likelihood, _ = sess.run(
        [mean_log_likelihood, train_step],
        feed_dict={input: data}
    )

    if step > 0:
        # computing difference between consecutive log-likelihoods
        difference = np.abs(current_likelihood - previous_likelihood)
        print("{0}:\tmean-likelihood {1:.8f}\tdifference {2}".format(
            step, current_likelihood, difference))

        # stopping if TOLERANCE was reached
        if difference <= TOLERANCE:
            break
    else:
        print("{0}:\tmean-likelihood {1:.8f}".format(
            step, current_likelihood))

    previous_likelihood = current_likelihood

# fetching final parameter values
final_means = means.eval(sess)
final_variances = variances.eval(sess)

# plotting data and the obtained GMM
tf_gmm_tools.plot_fitted_data(
    data, final_means, final_variances,
    true_means, true_variances)
