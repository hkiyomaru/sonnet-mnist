# Sonnet implementation of word2vec

## Development Environment

* OS: macOS Sierra 10.12.4
* Lang: Python 2.7.9

## Getting Started

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

Then, call `configure`.

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

## Sonnet
