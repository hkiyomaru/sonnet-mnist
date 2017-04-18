# Copyright 2017 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Example script to train a MLP model on MNIST datset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# dependency imports
import sonnet as snt
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("num_training_iterations", 1000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 50,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
tf.flags.DEFINE_integer("num_hidden", 128, "Size of MLP hidden layer.")
tf.flags.DEFINE_integer("output_size", 10, "Size of MLP output layer.")


tf.logging.set_verbosity(tf.logging.INFO)


class MNIST(snt.AbstractModule):
    """MNIST dataset model."""

    def __init__(self, mnist, batch_size, name='mnist'):
        """Construct a `MNIST`.

        Args:
            mnist: Dataset class object which has MNIST data.
            batch_size: Size of the output layer on top of the MLP.
            nonlinearity: Activation function.
            name: Name of the module.
        """

        super(MNIST, self).__init__(name=name)

        self._num_examples = mnist.num_examples
        self._images = tf.constant(mnist.images, dtype=tf.float32)
        self._labels = tf.constant(mnist.labels, dtype=tf.float32)
        self._batch_size = batch_size

    def _build(self):
        """Returns MNIST images and corresponding labels."""
        indices = tf.random_uniform([self._batch_size],
                                    0, self._num_examples, tf.int64)
        x = tf.gather(self._images, indices)
        y_ = tf.gather(self._labels, indices)
        return x, y_

    def cost(self, logits, target):
        """Returns cost.

        Args:
            logits: Model output.
            target: Correct labels.

        Returns:
            Cross-entropy loss for given outputs.
        """

        return -tf.reduce_sum(target * tf.log(logits))


class MLP(snt.AbstractModule):
    """MLP model, for use on MNIST dataset."""

    def __init__(self, num_hidden, output_size,
                 nonlinearity=tf.sigmoid, name='mlp'):
        """Construct a `MLP`.

        Args:
            num_hidden: Number of hidden units in first FC layer.
            output_size: Size of the output layer on top of the MLP.
            nonlinearity: Activation function.
            name: Name of the module.
        """

        super(MLP, self).__init__(name=name)

        self._num_hidden = num_hidden
        self._output_size = output_size
        self._nonlinearity = nonlinearity

        with self._enter_variable_scope():
            self._l1 = snt.Linear(output_size=self._num_hidden, name='l1')
            self._l2 = snt.Linear(output_size=self._output_size, name='l2')

    def _build(self, inputs):
        """Builds the MLP model sub-graph.

        Args
            inputs: A Tensor with the input MNIST data encoded as a
            784-dimensional representation. Its dimensions should be
            `[batch_size, 784]`.

        Returns:
            A Tensor with the prediction of given MNIST data encoded as a
            10-dimensional representation. Its dimensions should be
            `[batch_size, 10]`.
        """

        l1 = self._l1
        h = self._nonlinearity(l1(inputs))

        l2 = self._l2
        outputs = tf.nn.softmax(l2(h))

        return outputs


def train(num_training_iterations, report_interval):
    """Run the training of the MLP model on MNIST dataset."""

    mnist = read_data_sets('./MNIST_data', one_hot=True)
    dataset_train = MNIST(mnist.train, batch_size=FLAGS.batch_size)
    dataset_validation = MNIST(mnist.validation, batch_size=FLAGS.batch_size)
    dataset_test = MNIST(mnist.test, batch_size=FLAGS.batch_size)

    model = MLP(num_hidden=FLAGS.num_hidden, output_size=FLAGS.output_size)

    # Build the training model and get the training loss.
    train_x, train_y_ = dataset_train()
    train_y = model(train_x)
    train_loss = dataset_train.cost(train_y, train_y_)

    # Build the validation model and get the validation loss.
    validation_x, validation_y_ = dataset_validation()
    validation_y = model(validation_x)
    validation_loss = dataset_validation.cost(validation_y, validation_y_)

    # Build the test model and get the test loss.
    test_x, test_y_ = dataset_test()
    test_y = model(test_x)
    test_loss = dataset_test.cost(test_y, test_y_)

    # Set up optimizer
    train_step = tf.train.AdamOptimizer().minimize(train_loss)

    # Train.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for training_iteration in range(FLAGS.num_training_iterations):
            if (training_iteration + 1) % report_interval == 0:
                train_loss_v, validation_loss_v, _ = sess.run(
                    (train_loss, validation_loss, train_step))

                tf.logging.info("%d: Training loss %f. Validation loss %f.",
                                training_iteration,
                                train_loss_v,
                                validation_loss_v)
            else:
                train_loss_v, _ = sess.run((train_loss, train_step))
                tf.logging.info("%d: Training loss %f.",
                                training_iteration,
                                train_loss_v)

        test_loss = sess.run(test_loss)
        tf.logging.info("Test loss %f", test_loss)


def main(unused_argv):

    train(
        num_training_iterations=FLAGS.num_training_iterations,
        report_interval=FLAGS.report_interval)


if __name__ == '__main__':
    tf.app.run()
