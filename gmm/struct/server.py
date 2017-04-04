import sys

import tensorflow as tf


job_name = sys.argv[1]
task_id = int(sys.argv[2])

cluster_spec = tf.train.ClusterSpec({
    "master": ["localhost:2222"],
    "worker": ["localhost:2223", "localhost:2224"]
})

server = tf.train.Server(cluster_spec, job_name=job_name, task_index=task_id)
server.join()
