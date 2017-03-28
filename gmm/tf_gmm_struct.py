import numpy as np
import tensorflow as tf

import tf_gmm_tools


class DiagonalCovariance:

    def __init__(self, dims, variance=None, alpha=None, beta=None):
        self.dims = dims
        self.variance = variance
        self.alpha = alpha
        self.beta = beta

        self._variance = None
        self._prior = None
        self._alpha = None
        self._beta = None

    def initialize(self, dtype=tf.float64):
        if self._variance is None:
            if self.variance is not None:
                self._variance = tf.Variable(self.variance, dtype=dtype)
            else:
                self._variance = tf.Variable(tf.cast(tf.fill([self.dims], 1.0), dtype))

        if self._prior is None:
            if self.alpha is not None and self.beta is not None:
                self._prior = True
                self._alpha = tf.constant(self.alpha, dtype=dtype)
                self._beta = tf.constant(self.beta, dtype=dtype)
            else:
                self._prior = False

    def get_matrix(self):
        return tf.diag(self._variance)

    def get_quadratic_form(self, data, mean):
        sq_distances = tf.squared_difference(data, tf.expand_dims(mean, 0))

        return tf.reduce_sum(sq_distances / self._variance, 1)

    def get_log_determinant(self):
        return tf.reduce_sum(tf.log(self._variance))

    def get_prior_adjustment(self, variance, gamma_sum):
        adjusted_variance = variance
        adjusted_variance *= gamma_sum
        adjusted_variance += (2.0 * self._beta)
        adjusted_variance /= gamma_sum + (2.0 * (self._alpha + 1.0))

        return adjusted_variance

    def get_value_updater(self, data, new_mean, gamma_sum, gamma_weighted):
        new_sq_distances = tf.squared_difference(data, tf.expand_dims(new_mean, 0))
        new_variance = tf.reduce_sum(new_sq_distances * tf.expand_dims(gamma_weighted, 1), 0)

        if self._prior:
            new_variance = self.get_prior_adjustment(new_variance, gamma_sum)

        return self._variance.assign(new_variance)


class GaussianDistribution:

    def __init__(self, dims, mean=None, covariance=None):
        self.dims = dims
        self.mean = mean
        self.covariance = covariance

        self._mean = None
        self._covariance = None
        self._ln2piD = None

    def initialize(self, dtype=tf.float64):
        if self._mean is None:
            if self.mean is not None:
                self._mean = tf.Variable(self.mean, dtype=dtype)
            else:
                self._mean = tf.Variable(tf.cast(tf.fill([self.dims], 0.0), dtype))

        if self._covariance is None:
            if self.covariance is not None:
                self._covariance = self.covariance
            else:
                self._covariance = DiagonalCovariance(self.dims)

            self._covariance.initialize(dtype)

        if self._ln2piD is None:
            self._ln2piD = tf.constant(np.log(2 * np.pi) * self.dims, dtype=dtype)

    def get_parameters(self):
        return [
            self._mean,
            self._covariance.get_matrix()
        ]

    def get_log_probabilities(self, data):
        quadratic_form = self._covariance.get_quadratic_form(data, self._mean)
        log_coefficient = self._ln2piD + self._covariance.get_log_determinant()

        return -0.5 * (log_coefficient + quadratic_form)

    def get_parameter_updaters(self, data, gamma_sum, gamma_weighted):
        new_mean = tf.reduce_sum(data * tf.expand_dims(gamma_weighted, 1), 0)
        covariance_updater = self._covariance.get_value_updater(
            data, new_mean, gamma_sum, gamma_weighted)

        return [covariance_updater, self._mean.assign(new_mean)]


class MixtureModel:

    def __init__(self, data, components, dtype=tf.float64):
        self.data = data
        self.dims = data.shape[1]
        self.num_points = data.shape[0]
        self.components = components

        self._weights = tf.Variable(tf.cast(tf.fill([len(components)], 1.0 / len(components)), dtype))

        self._op_component_parameters = None

        self._initialize(dtype)

    def _initialize(self, dtype=tf.float64):
        self._component_data = []
        self._component_probabilities = []

        for comp in self.components:
            with tf.device("/cpu:0"):
                comp.initialize(dtype)
                self._component_data.append(tf.Variable(self.data, trainable=False, dtype=dtype))
                self._component_probabilities.extend([comp.get_log_probabilities(self._component_data[-1])])

        self._log_components = tf.stack(self._component_probabilities)
        self._log_weighted = self._log_components + tf.expand_dims(tf.log(self._weights), 1)
        self._log_shift = tf.expand_dims(tf.reduce_max(self._log_weighted, 0), 0)
        self._exp_log_shifted = tf.exp(self._log_weighted - self._log_shift)
        self._exp_log_shifted_sum = tf.reduce_sum(self._exp_log_shifted, 0)

        self._gamma = self._exp_log_shifted / self._exp_log_shifted_sum
        self._gamma_sum = tf.reduce_sum(self._gamma, 1)
        self._gamma_weighted = self._gamma / tf.expand_dims(self._gamma_sum, 1)

        self._log_likelihood = tf.reduce_sum(tf.log(self._exp_log_shifted_sum)) + tf.reduce_sum(self._log_shift)
        self._mean_log_likelihood = self._log_likelihood / tf.cast(self.num_points * self.dims, dtype)

        self._gamma_sum_split = tf.unstack(self._gamma_sum)
        self._gamma_weighted_split = tf.unstack(self._gamma_weighted)

        self._component_updaters = []

        for i in range(len(self.components)):
            with tf.device("/cpu:0"):
                self._component_updaters.extend(
                    self.components[i].get_parameter_updaters(
                        self._component_data[i],
                        self._gamma_sum_split[i],
                        self._gamma_weighted_split[i]
                    )
                )

        self._new_weights = self._gamma_sum / tf.cast(self.num_points, dtype)
        self._update_ops = self._component_updaters + [self._weights.assign(self._new_weights)]
        self._train_step = tf.group(*self._update_ops)

    def _get_component_parameters(self):
        if self._op_component_parameters is None:
            self._op_component_parameters = [comp.get_parameters() for comp in self.components]

        return self._op_component_parameters

    def train(self, tolerance=10e-6, max_steps=1000, feedback=None):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            previous_log_likelihood = -np.inf

            # training loop
            for step in range(max_steps):
                # executing a training step and
                # fetching evaluation information
                _, current_log_likelihood = sess.run([
                    self._train_step,
                    self._mean_log_likelihood
                ])

                if step > 0:
                    # computing difference between consecutive log-likelihoods
                    difference = current_log_likelihood - previous_log_likelihood

                    if feedback is not None:
                        # feeding back current log-likelihood and difference
                        feedback(step, current_log_likelihood, difference)

                    # stopping if TOLERANCE was reached
                    if difference <= tolerance:
                        break
                elif feedback is not None:
                    # feeding back initial log-likelihood
                    feedback(step, current_log_likelihood, None)

                previous_log_likelihood = current_log_likelihood

            output = [self._mean_log_likelihood, self._weights,
                      self._get_component_parameters()]

            return sess.run(output)


def feedback_sub(step, current_log_likelihood, difference):
    if difference is not None:
        print("{0}:\tmean-log-likelihood {1:.8f}\tdifference {2}".format(
            step, current_log_likelihood, difference))
    else:
        print("{0}:\tmean-log-likelihood {1:.8f}".format(
            step, current_log_likelihood))


DIMENSIONS = 2
COMPONENTS = 10
NUM_POINTS = 10000

TRAINING_STEPS = 1000
TOLERANCE = 10e-6


print("Generating data...")
synthetic_data, true_means, true_covariances, true_weights, responsibilities = tf_gmm_tools.generate_gmm_data(
    NUM_POINTS, COMPONENTS, DIMENSIONS, seed=10, diagonal=True)

print("Computing avg covariance...")
avg_variance = np.var(synthetic_data, axis=0).sum() / COMPONENTS / DIMENSIONS

print("Initializing components...")
mixture_components = []
for c in range(COMPONENTS):
    mixture_components.append(
        GaussianDistribution(
            DIMENSIONS, synthetic_data[c],
            DiagonalCovariance(
                DIMENSIONS,
                variance=np.full((DIMENSIONS,), avg_variance),
                alpha=1.0, beta=1.0
            )
        )
    )

print("Initializing model...")
gmm = MixtureModel(synthetic_data, mixture_components)

print("Training model...\n")
result = gmm.train(tolerance=TOLERANCE, feedback=feedback_sub)

final_means = np.stack([result[2][i][0] for i in range(COMPONENTS)])
final_covariances = np.stack([result[2][i][1] for i in range(COMPONENTS)])
final_weights = result[1]

tf_gmm_tools.plot_fitted_data(
    synthetic_data[:, :2],
    final_means[:, :2], final_covariances[:, :2, :2],
    true_means[:, :2], true_covariances[:, :2, :2]
)
