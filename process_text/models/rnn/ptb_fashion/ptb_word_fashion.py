# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

From
$ python ptb_word_lm.py --data_path=../../../../simple-examples/data/


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from process_text.models.rnn.ptb_fashion import reader
import os

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string(
    "zappos", "no", "indicate whether to use zappos vocab. Options: no, with_ngrams, only")

FLAGS = flags.FLAGS


class ToSave(object):
    """

    """
    pass



class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and config.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):  # 20 steps unrolled
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b

        self.probabilities = tf.nn.softmax(logits=logits)
        self.output = output

        loss = seq2seq.sequence_loss_by_example([logits],
                                                [tf.reshape(self._targets, [-1])],
                                                [tf.ones([batch_size * num_steps])],
                                                vocab_size)
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SusyConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 5
    hidden_size = 13
    max_epoch = 3
    max_max_epoch = 4
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 11
    vocab_size = 10000

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, m, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                      m.num_steps)):
        cost, state, probabilities, _ = session.run([m.cost, m.final_state, m.probabilities, eval_op],
                                                    {m.input_data: x,
                                                     m.targets: y,
                                                     m.initial_state: state})
        costs += cost
        iters += m.num_steps

        # if verbose and step % (epoch_size // 10) == 10:
        if verbose:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def generate_words(session, m, data, eval_op=tf.no_op()):
    """
    m is an object of PTBModel class
    data is a list of word indices
    """

    states_seq = []
    probabilities_seq = []
    predicted_word_seq = []
    x_seq = []
    y_seq = []
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                      m.num_steps)):
        state = m.initial_state.eval()
        cost, state, probabilities, _ = session.run([m.cost, m.final_state,
                                                     m.probabilities,
                                                     eval_op],
                                                    {m.input_data: x,
                                                     m.targets: y,
                                                     m.initial_state: state})
        predicted_word = np.argmax(probabilities, axis=1)
        predicted_word_seq.append(predicted_word)
        states_seq.append(state)
        probabilities_seq.append(probabilities)
        x_seq.append(x)
        y_seq.append(y)

    return predicted_word_seq, states_seq, probabilities_seq, x_seq, y_seq


def get_data_path():
    if FLAGS.zappos == 'no_zappos':
        return os.path.join(FLAGS.data_path, 'no_zappos')
    elif FLAGS.zappos == 'with_ngrams':
        return os.path.join(FLAGS.data_path, 'with_ngrams')
    elif FLAGS.zappos == 'only_zappos':
        return os.path.join(FLAGS.data_path, 'only_zappos')


def get_config():
    if FLAGS.model == "susy":
        return SusyConfig()
    elif FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)


def main(unused_args):
    # FLAGS.data_path = '../../.././../data/fashion53k/text/text_data/no_zappos/'

    if not FLAGS.data_path:
        FLAGS.data_path = '/Users/susanaparis/Documents/Belgium/DeepFashion/data/fashion53k/text/text_data/'
        # raise ValueError("Must set --data_path to PTB data directory")

    raw_data = reader.ptb_raw_data(get_data_path())
    train_data, valid_data, test_data, _ = raw_data

    print (get_data_path())

    train_vocab = reader.Vocabulary(get_data_path(), split='train')
    val_vocab = reader.Vocabulary(get_data_path(), split='val')
    test_vocab = reader.Vocabulary(get_data_path(), split='test')

    # train_data = train_data[0:5000]
    # valid_data = valid_data[0:1000]
    # test_data = test_data[0:1000]

    config = get_config()

    config.vocab_size = train_vocab.size  # the vocabulary size always from the train set
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    eval_config.vocab_size = train_vocab.size  # the vocabulary size always from the train set

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)

        saver = tf.train.Saver()
        # check if there is a previously saved model
        model_checkpoint = 'not_found.ckpt'
        if os.path.isfile(model_checkpoint):
            saver.restore(session, save_path=model_checkpoint)
            print ("model has been restored from file {}".format(model_checkpoint))
        else:
            tf.initialize_all_variables().run()
            print ("variables has been initialized")

        train_perplexity = -1
        valid_perplexity = -1

        if TRAIN_NOW:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)

        # predicted_word_seq, states_seq, probabilities_seq, x_seq, y_seq = generate_words(session,
        # mtest, test_data, tf.no_op())
        predicted_word_seq, vocab_embeddings, probabilities_seq, x_seq, y_seq = generate_words(session, mtest,
                                                                                               range(train_vocab.size),
                                                                                               tf.no_op())

        # Save the variables to disk.
        save_fname = "{}_model_{}.ckpt".format(FLAGS.model, FLAGS.zappos)
        save_path = saver.save(session, save_fname)
        print("Model saved in file: ", save_path)

        susy_to_save = ToSave()

        susy_to_save.train_vocab = train_vocab
        susy_to_save.valie_vocab = val_vocab
        susy_to_save.test_vocab = test_vocab

        susy_to_save.train_perplexity = train_perplexity
        susy_to_save.valid_perplexity = valid_perplexity
        susy_to_save.test_perplexity = test_perplexity

        susy_to_save.predicted_word_seq = predicted_word_seq
        susy_to_save.vocab_embeddings = vocab_embeddings
        susy_to_save.x_seq = x_seq
        susy_to_save.y_seq = y_seq

        var_save_fname = "all_variables_{}_model_{}.pkl".format(FLAGS.model, FLAGS.zappos)
        with open(var_save_fname, 'wb') as ff:
            pickle.dump(susy_to_save, ff)
        print ("all variables saved")


if __name__ == "__main__":

    TRAIN_NOW = True

    tf.app.run()


    # """
    # Epoch: 1 Learning rate: 1.000
    # 0.004 perplexity: 6464.393 speed: 526 wps
    # 0.104 perplexity: 843.819 speed: 655 wps
    # 0.204 perplexity: 626.071 speed: 662 wps
    # 0.304 perplexity: 507.349 speed: 671 wps
    # 0.404 perplexity: 437.786 speed: 662 wps
    # 0.504 perplexity: 392.019 speed: 673 wps
    # 0.604 perplexity: 352.973 speed: 686 wps
    # 0.703 perplexity: 325.982 speed: 682 wps
    # 0.803 perplexity: 304.863 speed: 691 wps
    # 0.903 perplexity: 285.472 speed: 697 wps
    # Epoch: 1 Train Perplexity: 270.953
    # Epoch: 1 Valid Perplexity: 182.201
    # Epoch: 2 Learning rate: 1.000
    # 0.004 perplexity: 212.932 speed: 736 wps
    # 0.104 perplexity: 152.170 speed: 751 wps
    # 0.204 perplexity: 159.342 speed: 736 wps
    # 0.304 perplexity: 154.188 speed: 721 wps
    # 0.404 perplexity: 151.077 speed: 722 wps
    # 0.504 perplexity: 148.570 speed: 718 wps
    # 0.604 perplexity: 143.737 speed: 716 wps
    # 0.703 perplexity: 141.491 speed: 713 wps
    # 0.803 perplexity: 139.459 speed: 712 wps
    # 0.903 perplexity: 135.764 speed: 707 wps
    # Epoch: 2 Train Perplexity: 133.665
    # Epoch: 2 Valid Perplexity: 142.842
    # Epoch: 3 Learning rate: 1.000
    # 0.004 perplexity: 143.436 speed: 745 wps
    # 0.104 perplexity: 105.349 speed: 695 wps
    # 0.204 perplexity: 114.197 speed: 697 wps
    # 0.304 perplexity: 111.273 speed: 691 wps
    # 0.404 perplexity: 110.212 speed: 698 wps
    # 0.504 perplexity: 109.323 speed: 700 wps
    # 0.604 perplexity: 106.671 speed: 689 wps
    # 0.703 perplexity: 105.937 speed: 691 wps
    # 0.803 perplexity: 105.354 speed: 694 wps
    # 0.903 perplexity: 103.112 speed: 695 wps
    # Epoch: 3 Train Perplexity: 102.118
    # Epoch: 3 Valid Perplexity: 131.147
    # Epoch: 4 Learning rate: 1.000
    # 0.004 perplexity: 113.805 speed: 756 wps
    # 0.104 perplexity: 85.052 speed: 757 wps
    # 0.204 perplexity: 93.458 speed: 757 wps
    # 0.304 perplexity: 91.243 speed: 732 wps
    # 0.404 perplexity: 90.779 speed: 703 wps
    # 0.504 perplexity: 90.380 speed: 686 wps
    # 0.604 perplexity: 88.527 speed: 676 wps
    # 0.703 perplexity: 88.284 speed: 658 wps
    # 0.803 perplexity: 88.046 speed: 651 wps
    # 0.903 perplexity: 86.410 speed: 636 wps
    # Epoch: 4 Train Perplexity: 85.868
    # Epoch: 4 Valid Perplexity: 128.424
    # Epoch: 5 Learning rate: 1.000
    # 0.004 perplexity: 100.637 speed: 565 wps
    # 0.104 perplexity: 73.565 speed: 598 wps
    # 0.204 perplexity: 81.233 speed: 592 wps
    # 0.304 perplexity: 79.434 speed: 590 wps
    # 0.404 perplexity: 79.199 speed: 602 wps
    # 0.504 perplexity: 79.007 speed: 613 wps
    # 0.604 perplexity: 77.608 speed: 621 wps
    # 0.703 perplexity: 77.602 speed: 627 wps
    # 0.803 perplexity: 77.550 speed: 631 wps
    # 0.903 perplexity: 76.252 speed: 631 wps
    # Epoch: 5 Train Perplexity: 75.912
    # Epoch: 5 Valid Perplexity: 128.226
    # Epoch: 6 Learning rate: 0.500
    # 0.004 perplexity: 89.473 speed: 666 wps
    # 0.104 perplexity: 64.076 speed: 661 wps
    # 0.204 perplexity: 69.514 speed: 662 wps
    # 0.304 perplexity: 66.940 speed: 663 wps
    # 0.404 perplexity: 65.972 speed: 664 wps
    # 0.504 perplexity: 65.090 speed: 664 wps
    # 0.604 perplexity: 63.193 speed: 645 wps
    # 0.703 perplexity: 62.517 speed: 614 wps
    # 0.803 perplexity: 61.816 speed: 595 wps
    # 0.903 perplexity: 60.101 speed: 587 wps
    # Epoch: 6 Train Perplexity: 59.204
    # Epoch: 6 Valid Perplexity: 120.332
    # Epoch: 7 Learning rate: 0.250
    # 0.004 perplexity: 74.262 speed: 703 wps
    # 0.104 perplexity: 53.167 speed: 690 wps
    # 0.204 perplexity: 57.783 speed: 638 wps
    # 0.304 perplexity: 55.553 speed: 671 wps
    # 0.404 perplexity: 54.668 speed: 688 wps
    # 0.504 perplexity: 53.818 speed: 699 wps
    # 0.604 perplexity: 52.147 speed: 702 wps
    # 0.703 perplexity: 51.463 speed: 701 wps
    # 0.803 perplexity: 50.738 speed: 707 wps
    # 0.903 perplexity: 49.162 speed: 711 wps
    # Epoch: 7 Train Perplexity: 48.265
    # Epoch: 7 Valid Perplexity: 121.626
    # Epoch: 8 Learning rate: 0.125
    # 0.004 perplexity: 65.646 speed: 739 wps
    # 0.104 perplexity: 47.208 speed: 744 wps
    # 0.204 perplexity: 51.407 speed: 739 wps
    # 0.304 perplexity: 49.397 speed: 722 wps
    # 0.404 perplexity: 48.576 speed: 728 wps
    # 0.504 perplexity: 47.784 speed: 733 wps
    # 0.604 perplexity: 46.255 speed: 736 wps
    # 0.703 perplexity: 45.596 speed: 739 wps
    # 0.803 perplexity: 44.891 speed: 730 wps
    # 0.903 perplexity: 43.431 speed: 728 wps
    # Epoch: 8 Train Perplexity: 42.576
    # Epoch: 8 Valid Perplexity: 123.049
    # Epoch: 9 Learning rate: 0.062
    # 0.004 perplexity: 61.116 speed: 615 wps
    # 0.104 perplexity: 44.196 speed: 705 wps
    # 0.204 perplexity: 48.198 speed: 704 wps
    # 0.304 perplexity: 46.292 speed: 702 wps
    # 0.404 perplexity: 45.517 speed: 708 wps
    # 0.504 perplexity: 44.770 speed: 716 wps
    # 0.604 perplexity: 43.316 speed: 722 wps
    # 0.703 perplexity: 42.671 speed: 726 wps
    # 0.803 perplexity: 41.966 speed: 730 wps
    # 0.903 perplexity: 40.565 speed: 732 wps
    # Epoch: 9 Train Perplexity: 39.728
    # Epoch: 9 Valid Perplexity: 123.891
    # Epoch: 10 Learning rate: 0.031
    # 0.004 perplexity: 58.841 speed: 747 wps
    # 0.104 perplexity: 42.630 speed: 756 wps
    # 0.204 perplexity: 46.578 speed: 759 wps
    # 0.304 perplexity: 44.700 speed: 761 wps
    # 0.404 perplexity: 43.936 speed: 761 wps
    # 0.504 perplexity: 43.213 speed: 762 wps
    # 0.604 perplexity: 41.799 speed: 763 wps
    # 0.703 perplexity: 41.163 speed: 763 wps
    # 0.803 perplexity: 40.452 speed: 758 wps
    # 0.903 perplexity: 39.080 speed: 748 wps
    # Epoch: 10 Train Perplexity: 38.253
    # Epoch: 10 Valid Perplexity: 124.008
    # Epoch: 11 Learning rate: 0.016
    # 0.004 perplexity: 57.527 speed: 662 wps
    # 0.104 perplexity: 41.726 speed: 674 wps
    # 0.204 perplexity: 45.663 speed: 672 wps
    # 0.304 perplexity: 43.821 speed: 674 wps
    # 0.404 perplexity: 43.067 speed: 674 wps
    # 0.504 perplexity: 42.357 speed: 674 wps
    # 0.604 perplexity: 40.959 speed: 674 wps
    # 0.703 perplexity: 40.323 speed: 673 wps
    # 0.803 perplexity: 39.614 speed: 668 wps
    # 0.903 perplexity: 38.260 speed: 668 wps
    # Epoch: 11 Train Perplexity: 37.442
    # Epoch: 11 Valid Perplexity: 123.624
    # Epoch: 12 Learning rate: 0.008
    # 0.004 perplexity: 56.688 speed: 665 wps
    # 0.104 perplexity: 41.151 speed: 666 wps
    # 0.204 perplexity: 45.075 speed: 660 wps
    # 0.304 perplexity: 43.279 speed: 658 wps
    # 0.404 perplexity: 42.545 speed: 647 wps
    # 0.504 perplexity: 41.849 speed: 646 wps
    # 0.604 perplexity: 40.467 speed: 641 wps
    # 0.703 perplexity: 39.833 speed: 645 wps
    # 0.803 perplexity: 39.129 speed: 642 wps
    # 0.903 perplexity: 37.788 speed: 643 wps
    # Epoch: 12 Train Perplexity: 36.979
    # Epoch: 12 Valid Perplexity: 123.155
    # Epoch: 13 Learning rate: 0.004
    # 0.004 perplexity: 56.178 speed: 661 wps
    # 0.104 perplexity: 40.806 speed: 666 wps
    # 0.204 perplexity: 44.713 speed: 654 wps
    # 0.304 perplexity: 42.950 speed: 655 wps
    # 0.404 perplexity: 42.232 speed: 655 wps
    # 0.504 perplexity: 41.547 speed: 671 wps
    # 0.604 perplexity: 40.180 speed: 683 wps
    # 0.703 perplexity: 39.552 speed: 692 wps
    # 0.803 perplexity: 38.853 speed: 699 wps
    # 0.903 perplexity: 37.521 speed: 704 wps
    # Epoch: 13 Train Perplexity: 36.718
    # Epoch: 13 Valid Perplexity: 122.831
    # Test Perplexity: 116.862




    # """

    """
    0.556 perplexity: 369.408 speed: 894 wps
0.667 perplexity: 372.481 speed: 876 wps
0.778 perplexity: 361.678 speed: 867 wps
0.889 perplexity: 365.498 speed: 873 wps
Epoch: 4 Train Perplexity: 365.498
Epoch: 4 Valid Perplexity: 2915.074
Test Perplexity: 12961.036

0.980 perplexity: 528.674 speed: 943 wps
0.990 perplexity: 529.198 speed: 943 wps
Epoch: 4 Train Perplexity: 529.198
Epoch: 4 Valid Perplexity: 1515.820
Test Perplexity: 1760.325


0.980 perplexity: 502.067 speed: 918 wps
0.990 perplexity: 502.568 speed: 919 wps
Epoch: 1 Train Perplexity: 502.568
Epoch: 1 Valid Perplexity: 1586.947
Epoch: 2 Learning rate: 1.000

Epoch: 1 Learning rate: 1.000
0.000 perplexity: 426.237 speed: 355 wps
0.010 perplexity: 492.804 speed: 508 wps
0.020 perplexity: 529.891 speed: 610 wps


Epoch: 1 Learning rate: 1.000
0.000 perplexity: 369.820 speed: 349 wps
0.010 perplexity: 440.493 speed: 490 wps
0.020 perplexity: 488.267 speed: 581 wps
0.030 perplexity: 486.468 speed: 653 wps

    """



