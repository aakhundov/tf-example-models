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

        self.tf_graph = tf.Graph()

        self._initialize_workers(cluster)
        self._initialize_component_mapping()
        self._initialize_data_sources()
        self._initialize_variables(dtype)
        self._initialize_graph(dtype)

    def _initialize_workers(self, cluster):
        if cluster is None:
            self.master_host = ""
            self.workers = [None]
        else:
            self.master_host = "grpc://" + cluster.job_tasks("master")[0]
            self.workers = ["/job:worker/task:" + str(i) for i in range(cluster.num_tasks("worker"))]

    def _initialize_component_mapping(self):
        self.mapping = {
            w: [] for w in range(len(self.workers))
        }
        for component_id in range(len(self.components)):
            worker_id = component_id % len(self.workers)
            self.mapping[worker_id].append(component_id)

    def _initialize_data_sources(self):
        with self.tf_graph.as_default():
            self.tf_input_sources = []
            for data in self.data:
                self.tf_input_sources.append(tf.placeholder(
                    data.dtype, shape=[self.num_points, data.shape[1]]
                ))

            self.tf_worker_data = []
            for w in self.workers:
                with tf.device(w):
                    self.tf_worker_data.append(
                        [tf.Variable(input, trainable=False)
                         for input in self.tf_input_sources]
                    )

    def _initialize_variables(self, dtype):
        with self.tf_graph.as_default():
            self.tf_dims = tf.constant(self.dims, dtype=dtype)
            self.tf_num_points = tf.constant(self.num_points, dtype=dtype)
            self.tf_weights = tf.Variable(
                tf.cast(tf.fill(
                    [len(self.components)], 1.0 / len(self.components)
                ), dtype)
            )

    def _initialize_graph(self, dtype=tf.float64):
        with self.tf_graph.as_default():
            tf_component_log_probabilities = []
            for worker_id in self.mapping.keys():
                for component_id in self.mapping[worker_id]:
                    with tf.device(self.workers[worker_id]):
                        self.components[component_id].initialize(dtype)
                        tf_component_log_probabilities.append(
                            self.components[component_id].get_log_probabilities(
                                self.tf_worker_data[worker_id]
                            )
                        )

            tf_log_components = tf.parallel_stack(tf_component_log_probabilities)
            tf_log_weighted = tf_log_components + tf.expand_dims(tf.log(self.tf_weights), 1)
            tf_log_shift = tf.expand_dims(tf.reduce_max(tf_log_weighted, 0), 0)
            tf_exp_log_shifted = tf.exp(tf_log_weighted - tf_log_shift)
            tf_exp_log_shifted_sum = tf.reduce_sum(tf_exp_log_shifted, 0)
            tf_log_likelihood = tf.reduce_sum(tf.log(tf_exp_log_shifted_sum)) + tf.reduce_sum(tf_log_shift)

            self.tf_mean_log_likelihood = tf_log_likelihood / (self.tf_num_points * self.tf_dims)

            tf_gamma = tf_exp_log_shifted / tf_exp_log_shifted_sum
            tf_gamma_sum = tf.reduce_sum(tf_gamma, 1)
            tf_gamma_weighted = tf_gamma / tf.expand_dims(tf_gamma_sum, 1)
            tf_gamma_sum_split = tf.unstack(tf_gamma_sum)
            tf_gamma_weighted_split = tf.unstack(tf_gamma_weighted)

            tf_component_updaters = []
            for worker_id in self.mapping.keys():
                for component_id in self.mapping[worker_id]:
                    with tf.device(self.workers[worker_id]):
                        tf_component_updaters.extend(
                            self.components[component_id].get_parameter_updaters(
                                self.tf_worker_data[worker_id],
                                tf_gamma_weighted_split[component_id],
                                tf_gamma_sum_split[component_id]
                            )
                        )

            tf_new_weights = tf_gamma_sum / self.tf_num_points
            tf_weights_updater = self.tf_weights.assign(tf_new_weights)
            tf_all_updaters = tf_component_updaters + [tf_weights_updater]

            self.tf_train_step = tf.group(*tf_all_updaters)
            self.tf_component_parameters = [
                comp.get_parameters() for comp in self.components
            ]

    def train(self, tolerance=10e-6, max_steps=1000, feedback=None):
        with tf.Session(target=self.master_host, graph=self.tf_graph) as sess:
            sess.run(
                tf.global_variables_initializer(),
                feed_dict={self.tf_input_sources[i]: self.data[i] for i in range(len(self.data))}
            )

            previous_log_likelihood = -np.inf
            for step in range(max_steps):
                _, current_log_likelihood = sess.run([
                    self.tf_train_step,
                    self.tf_mean_log_likelihood
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
                self.tf_mean_log_likelihood,
                self.tf_weights, self.tf_component_parameters
            ])
