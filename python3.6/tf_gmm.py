import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as pat


DIMENSIONS = 2
COMPONENTS = 10
DATA_POINTS = 10000

TRAINING_STEPS = 1000
TOLERANCE = 1e-6


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


def plot_fitted_data(points, c_means, c_variances):
    plt.plot(points[:, 0], points[:, 1], "b.", zorder=0)
    plt.plot(c_means[:, 0], c_means[:, 1], "r.", zorder=1)

    for i in range(c_means.shape[0]):
        std = np.sqrt(c_variances[i])
        plt.axes().add_artist(pat.Ellipse(
            c_means[i], 2 * std[0], 2 * std[1],
            fill=False, color="red", linewidth=2, zorder=1
        ))

    plt.show()


# PREPARING DATA

# generating DATA_POINTS points from a GMM with COMPONENTS components
data, true_means, true_variances, true_weights = generate_gmm_data(DATA_POINTS, COMPONENTS, DIMENSIONS)


# BUILDING COMPUTATIONAL GRAPH

# model inputs: data points (images)
input = tf.placeholder(tf.float64, [None, DIMENSIONS])
alpha = tf.placeholder_with_default(tf.cast([1.0], tf.float64), [1])
beta = tf.placeholder_with_default(tf.cast([1.0], tf.float64), [1])

# constants: D*ln(2*pi), variance prior parameters
ln2piD = tf.constant(np.log(2 * np.pi) * DIMENSIONS, dtype=tf.float64)

# computing input statistics
dim_means = tf.reduce_mean(input, 0)
dim_distances = tf.squared_difference(input, tf.expand_dims(dim_means, 0))
dim_variances = tf.reduce_sum(dim_distances, 0) / tf.cast(tf.shape(input)[0], tf.float64)
avg_variance = tf.cast(tf.reduce_sum(dim_variances) / COMPONENTS / DIMENSIONS, tf.float64)
rand_point_ids = tf.squeeze(tf.multinomial(tf.ones([1, tf.shape(input)[0]]), COMPONENTS))

# trainable variables: component means, variances, and weights
means = tf.Variable(tf.gather(input, rand_point_ids), dtype=tf.float64)
variances = tf.Variable(tf.cast(tf.ones([COMPONENTS, DIMENSIONS]), tf.float64) * avg_variance)
weights = tf.Variable(tf.cast(tf.fill([COMPONENTS], 1. / COMPONENTS), tf.float64))

# E-step: recomputing responsibilities with respect to the current parameter values
distances = tf.squared_difference(tf.expand_dims(input, 0), tf.expand_dims(means, 1))
dist_times_inv_var = tf.reduce_sum(distances / tf.expand_dims(variances, 1), 2)
log_coefficients = tf.expand_dims(ln2piD + tf.reduce_sum(tf.log(variances), 1), 1)
log_components = -0.5 * (log_coefficients + dist_times_inv_var)
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
variances_ *= tf.expand_dims(gamma_sum, 1)
variances_ += (2.0 * beta)
variances_ /= tf.expand_dims(gamma_sum + (2.0 * (alpha + 1.0)), 1)

# log-likelihood: objective function being maximized up to a TOLERANCE delta
log_likelihood = tf.reduce_sum(tf.log(exp_log_shifted_sum)) + tf.reduce_sum(log_shift)
mean_log_likelihood = log_likelihood / tf.cast(tf.shape(input)[0] * tf.shape(input)[1], tf.float64)

# updating the parameters by new values
train_step = tf.group(
    means.assign(means_),
    variances.assign(variances_),
    weights.assign(weights_)
)


# RUNNING COMPUTATIONAL GRAPH

# creating session
sess = tf.InteractiveSession()

# initializing trainable variables
sess.run(tf.global_variables_initializer(), feed_dict={input: data})

previous_likelihood = -np.inf

# training loop
for step in range(TRAINING_STEPS):
    # executing a training step and
    # fetching evaluation information
    current_likelihood, current_means, current_variances, _ = sess.run(
        [mean_log_likelihood, means_, variances_, train_step],
        feed_dict={input: data}
    )

    if step > 0:
        # computing difference between consecutive likelihoods
        difference = np.abs(current_likelihood - previous_likelihood)
        print("{0}:\tmean-likelihood {1:.8f}\tdifference {2}".format(
            step, current_likelihood, difference))

        # stopping if TOLERANCE reached
        if difference <= TOLERANCE:
            break
    else:
        print("{0}:\tmean-likelihood {1:.8f}".format(
            step, current_likelihood))

    previous_likelihood = current_likelihood

plot_raw_data(data)
plot_fitted_data(data, current_means, current_variances)
