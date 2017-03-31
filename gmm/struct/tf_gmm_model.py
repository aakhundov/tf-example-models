import numpy as np
import tensorflow as tf


class MixtureModel:

    def __init__(self, data, components, cluster=None, dtype=tf.float64):
        if isinstance(data, np.ndarray):
            data = [data]

        self.data = data
        self.dims = sum(d.shape[1] for d in data)
        self.num_points = data[0].shape[0]
        self.components = components

        self._initialize_workers(cluster)
        self._initialize_graph(dtype)

    def _initialize_workers(self, cluster):
        if cluster is None:
            self.master_host = ""
            self.workers = [None]
        else:
            self.master_host = "grpc://" + cluster.job_tasks("master")[0]
            self.workers = ["/job:worker/task:" + str(i) for i in range(cluster.num_tasks("worker"))]

    def _initialize_graph(self, dtype=tf.float64):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self._dims = tf.constant(self.dims, dtype=dtype)
            self._num_points = tf.constant(self.num_points, dtype=dtype)

            self._weights = tf.Variable(tf.cast(tf.fill([len(self.components)], 1.0 / len(self.components)), dtype))

            self._inputs = []
            for data in self.data:
                self._inputs.append(tf.placeholder(
                    data.dtype, shape=[self.num_points, data.shape[1]]
                ))

            self._worker_data = []
            for w in self.workers:
                with tf.device(w):
                    self._worker_data.append(
                        [tf.Variable(input, trainable=False) for input in self._inputs]
                    )

            self._component_log_probabilities = []
            for i in range(len(self.components)):
                component = self.components[i]
                worker_id = i % len(self.workers)
                with tf.device(self.workers[worker_id]):
                    component.initialize(dtype)
                    self._component_log_probabilities.append(
                        component.get_log_probabilities(self._worker_data[worker_id])
                    )

            self._log_components = tf.parallel_stack(self._component_log_probabilities)
            self._log_weighted = self._log_components + tf.expand_dims(tf.log(self._weights), 1)
            self._log_shift = tf.expand_dims(tf.reduce_max(self._log_weighted, 0), 0)
            self._exp_log_shifted = tf.exp(self._log_weighted - self._log_shift)
            self._exp_log_shifted_sum = tf.reduce_sum(self._exp_log_shifted, 0)
            self._log_likelihood = tf.reduce_sum(tf.log(self._exp_log_shifted_sum)) + tf.reduce_sum(self._log_shift)
            self._mean_log_likelihood = self._log_likelihood / (self._num_points * self._dims)

            self._gamma = self._exp_log_shifted / self._exp_log_shifted_sum
            self._gamma_sum = tf.reduce_sum(self._gamma, 1)
            self._gamma_weighted = self._gamma / tf.expand_dims(self._gamma_sum, 1)
            self._gamma_sum_split = tf.unstack(self._gamma_sum)
            self._gamma_weighted_split = tf.unstack(self._gamma_weighted)

            self._component_updaters = []
            for i in range(len(self.components)):
                component = self.components[i]
                worker_id = i % len(self.workers)
                with tf.device(self.workers[worker_id]):
                    self._component_updaters.extend(
                        component.get_parameter_updaters(
                            self._worker_data[worker_id],
                            self._gamma_weighted_split[i],
                            self._gamma_sum_split[i]
                        )
                    )

            self._new_weights = self._gamma_sum / self._num_points
            self._weights_updater = self._weights.assign(self._new_weights)
            self._all_updaters = self._component_updaters + [self._weights_updater]
            self._train_step = tf.group(*self._all_updaters)

            self._component_parameters = [comp.get_parameters() for comp in self.components]

    def train(self, tolerance=10e-6, max_steps=1000, feedback=None):
        with tf.Session(target=self.master_host, graph=self.graph) as sess:
            sess.run(
                tf.global_variables_initializer(),
                feed_dict={self._inputs[i]: self.data[i] for i in range(len(self.data))}
            )

            previous_log_likelihood = -np.inf

            for step in range(max_steps):
                _, current_log_likelihood = sess.run([
                    self._train_step,
                    self._mean_log_likelihood
                ])

                if step > 0:
                    difference = current_log_likelihood - previous_log_likelihood

                    if feedback is not None:
                        feedback(step, current_log_likelihood, difference)

                    if tolerance is not None and difference <= tolerance:
                        break
                else:
                    if feedback is not None:
                        feedback(step, current_log_likelihood, None)

                previous_log_likelihood = current_log_likelihood

            return sess.run([
                self._mean_log_likelihood,
                self._weights, self._component_parameters
            ])
