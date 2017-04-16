"""Tensorflow implementation of word2vec."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from six.moves import urllib
import zipfile

import numpy as np
import tensorflow as tf

# Configuration
batch_size = 256
embedding_size = 128
num_sampled = 15  # Number of negative examples to sample.
vocab_size = 30000


def download(filename='text8.zip', url='http://mattmahoney.net/dc/'):
    """Download a file if it does not exist."""
    if os.path.exists(filename):
        print("{} already exists.".format(filename))
    else:
        print('Start downloading {}.'.format(filename))
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    return filename


def read_data(filename):
    """Extract the corpus, which has a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, vocabulary_size=vocab_size):
    count = [('UNK', -1)]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = {word: i for i, (word, _) in enumerate(count)}
    data = [dictionary[word] if word in dictionary else 0 for word in words]
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# sentences to words and count
filename = download()  # download text8.zip
words = read_data(filename)

# Build dictionaries
data, count, dictionary, reverse_dictionary = build_dataset(words)

# Let's make a training data for window size 1 for simplicity
# ([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...
cbow_pairs = [[[data[i-1], data[i+1]], data[i]] for i in range(len(data)-1)]

# Let's make skip-gram pairs
# (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
skip_gram_pairs = [[c[1], c[0][0]] for c in cbow_pairs] + \
                  [[c[1], c[0][1]] for c in cbow_pairs]


def generate_batch(size):
    r = np.random.choice(range(len(skip_gram_pairs)), size, replace=False)
    x_batch = [skip_gram_pairs[i][0] for i in r]
    y_batch = [[skip_gram_pairs[i][1]] for i in r]
    return x_batch, y_batch


train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
                 tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)  # lookup table

nce_weights = tf.Variable(
    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([vocab_size]))

loss = tf.reduce_mean(
       tf.nn.nce_loss(nce_weights, nce_biases, train_labels,
                      embed, num_sampled, vocab_size))

train_op = tf.train.AdamOptimizer(1e-1).minimize(loss)

# Launch the graph in a session
with tf.Session() as sess:
    # Initializing all variables
    tf.global_variables_initializer().run()

    for step in range(100):
        batch_inputs, batch_labels = generate_batch(batch_size)
        _, loss_val = sess.run([train_op, loss],
                               feed_dict={train_inputs: batch_inputs,
                                          train_labels: batch_labels})
        if step % 10 == 0:
            print("Step: {}, Loss: {}".format(step, loss_val))

    trained_embeddings = embeddings.eval()
