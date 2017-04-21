# Sonnet implementation of MNIST classification

## Development environment

* macOS: Sierra 10.12.4
* Python: 2.7.9
* Sonnet: 1.0
* TensorFlow: 1.0.1

## Getting started

First of all, you have to install [bazel](https://bazel.build/versions/master/docs/install.html), a build software.

Sonnet is a TensorFlow-based neural network library.
Ensure the version of TensorFlow installed is at least 1.0.1.

```
$ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.1-py2-none-any.whl
```

Clone the Sonnet source code.

```
$ git clone --recursive https://github.com/deepmind/sonnet
```

Call `configure` to set up configuration of introducing GPU support and so on.

```
$ cd sonnet/tensorflow
$ ./configure
$ cd ../
```

Build and run the installer.

```
$ mkdir /tmp/sonnet
$ bazel build --config=opt :install
$ ./bazel-bin/install /tmp/sonnet
```

Finally, pip install the generated whell file.

```
$ pip install /tmp/sonnet/*.whl
```

That's all you need to do.

## Try Sonnet anyway

OK, everything is ready.

`cd source/` and run the following command.

```
$ python sonnet_mnist.py
```

## How it works

First of all, you need to import dependent libraries.

```python
import sonnet as snt
import tensorflow as tf
```

Then, let's define the model to classify MNIST images.
In Sonnet, you just create a new class which inherits from `snt.AbstractModule` to define a your own module.

```python
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
```

It is a good idea to also define a dataset class as a `snt.AbstractModule`.

```python
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
```

It is really simple to describe the training flow.

```python
mnist = read_data_sets('./MNIST_data', one_hot=True)
dataset_train = MNIST(mnist.train, batch_size=FLAGS.batch_size)
dataset_validation = MNIST(mnist.validation, batch_size=FLAGS.batch_size)
dataset_test = MNIST(mnist.test, batch_size=FLAGS.batch_size)

model = MLP(num_hidden=FLAGS.num_hidden, output_size=FLAGS.output_size)

# Build the training model and get the training loss.
train_x, train_y_ = dataset_train()
train_y = model(train_x)
train_loss = dataset_train.cost(train_y, train_y_)

# Get the validation loss.
validation_x, validation_y_ = dataset_validation()
validation_y = model(validation_x)
validation_loss = dataset_validation.cost(validation_y, validation_y_)

# Get the test loss.
test_x, test_y_ = dataset_test()
test_y = model(test_x)
test_loss = dataset_test.cost(test_y, test_y_)

# Set up optimizer.
train_step = tf.train.AdamOptimizer().minimize(train_loss)
```

Then, launch a session and run the training iterations on it.

```python
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
```
