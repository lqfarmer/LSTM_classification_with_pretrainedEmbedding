#!/usr/bin/env python
#coding=utf8
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

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import re
import sys

import tensorflow.python.platform

import numpy as np
import tensorflow as tf
import struct
import codecs


from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
# from tensorflow.models.nn import rnn_cell
# from tensorflow.models.rnn import seq2seq
#from tf.nn.rnn_cell import rnn_cell
#from tf.nn.seq2seq import seq2seq
import reader_zhx1

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")

flags.DEFINE_string("data_path", "/search/odin/doc2vec_jar/data/", "data_path")

flags.DEFINE_string("load_word2vec", "true",
    "Whether to load existing word2vec dictionary file.")
flags.DEFINE_string('model_path','/search/odin/data/liuqi/doc2vec_jar/lstm/model/lstm_sentenceset_segment.tfmodel','path of the word2vec file')

FLAGS = flags.FLAGS


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, word_embeddings = None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size])
    self._lengths = tf.placeholder(tf.int32, [batch_size])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    offsets = (self._lengths - 1) * self.batch_size + tf.range(0, self.batch_size)

    with tf.device("/cpu:0"):
      if word_embeddings is None:
          #self.embedding = tf.get_variable("embedding", [vocab_size, size])
          self.embedding = tf.get_variable("embedding", [4200000, 100])
      else:
          rows = word_embeddings.shape[0]
          cols = word_embeddings.shape[1]
          self.embedding = tf.get_variable("embedding", [rows, cols], trainable=False) #embedding = tf.constant(word_embeddings)
          '''
          print('before embedding !!!!')
          self.embedding = tf.constant(word_embeddings)
          print ('after embedding !!!!')
          '''
        #self.embedding = tf.constant(np.random.random_sample((vocab_size * size,)) * config.init_scale - config.init_scale, tf.float32, [vocab_size, size], "embedding")
      inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)

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
      # outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    states = []
    state = self._initial_state
    self._initial_state = tf.convert_to_tensor(self._initial_state)
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
        states.append(state)

    output = tf.reshape(tf.concat(0, outputs), [-1, size])
    output = tf.gather(output, offsets)
    logits = tf.nn.xw_plus_b(output,
                             tf.get_variable("softmax_w", [size, config.num_label]),
                             tf.get_variable("softmax_b", [config.num_label]))
    self.logits_ = logits
    self.results = tf.argmax(logits, 1)

    batch_size = tf.size(self._targets)
    labels = tf.expand_dims(self._targets, 1)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, config.num_label]), 1.0, 0.0)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                          onehot_labels,
                                                          name='xentropy')

    corrects = tf.nn.in_top_k(logits, self._targets, 1)
    self._corrects_num = tf.reduce_sum(tf.cast(corrects, tf.int32))
    self._cost = cost = tf.reduce_mean(loss, name='xentropy_mean')
    self._final_state = states[-1]

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
  def lengths(self):
    return self._lengths

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def corrects_num(self):
    return self._corrects_num

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  #learning_rate = 0.1
  learning_rate = 1
  max_grad_norm = 5
  num_layers = 1
  num_steps = 30
  #hidden_size = 256
  hidden_size = 300
  #max_epoch = 9999
  max_epoch = 4
  max_max_epoch = 1000
  keep_prob = 1.0
  #lr_decay = 0.5
  lr_decay = 0.9
  batch_size = 10000
  vocab_size = 60000
  num_label = 2


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 9999
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 5000
  vocab_size = 60000
  num_label = 23


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 9999
  max_max_epoch = 55
  #keep_prob = 0.35
  keep_prob = 1
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 60000
  num_label = 23
  

def run_epoch(session, m, sList, lList, eval_op, fout, fout1, run_type="train"):
  """Runs the model on the given data."""
  epoch_size = (len(lList) // m.batch_size)
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  total_correct_num = 0
  currentCorrectForDebug = 0
  currentItersForDebug = 0
  local_accuracy = 0
  #print('state_len=' + str(len(state)))
  #with vs.variable_scope("BasicLSTMCell"):
    #c, h = array_ops.split(1, 2, state)
    #print(c.eval())
    #print(h.eval())
  '''
  for v in tf.all_variables():
    print(v.name)
    if re.match('^.+Matrix.+$', v.name):
      mat = v.value().eval()
      for r in range(0,len(mat)):
        mat_line = ''
        for c in range(0, len(mat[r])):
          mat_line += str(mat[r][c])
          if c < len(mat[r])-1:
            mat_line += ' '
        print(mat_line)
      print('\n')
    #print(v.value().eval())

    #theatas = tf.get_variable(v.name).eval()
  '''

  for step, (x, length, y) in enumerate(reader_zhx1.ptb_iterator(sList, lList, m.batch_size,
                                                    m.num_steps)):
    #np.savetxt("embedd"+str(step), session.run(m.embedding))    
    for i in range(len(length)):
        if length[i] > m.num_steps: length[i] = m.num_steps
    logits,results, cost, correct_num, state, _ = session.run([m.logits_,m.results, m.cost, m.corrects_num, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.lengths: length,
                                  m.initial_state: state})
    #print("______results______ %.3f" % (local_accuracy))

    # mark middle result
    if run_type=="test":
        for l in logits:
            fout1.write(str(l))
            fout1.write("\n")
        for r in results:
            fout.write(str(r))
            fout.write("\n")
#       print(logits)
    #  print("m.results:")
    #  print(results)
    #  print("m.cost:")
    #  print(cost)
    #  print("m.corrects_num:")
    #  print(correct_num)

    #print("______targets______")
    #print(y)
    total_correct_num += correct_num
    currentCorrectForDebug += correct_num
    currentItersForDebug += 1
    iters += 1



    #if run_type=="train" and step % (epoch_size // 200) == 10:
    if run_type=="train":
        local_accuracy = (currentCorrectForDebug / (currentItersForDebug * m.batch_size))
        print("[%s] cur_step:%d/%.3f accuracy:%.3f speed:%.0fwps local-accuracy:%.3f" %(time.ctime(), step + 1, (step + 1 ) * 1.0 / epoch_size, (total_correct_num / (step * m.batch_size)),iters * m.batch_size / (time.time() - start_time), local_accuracy))
        currentCorrectForDebug = 0
        currentItersForDebug = 0
  if run_type == 'train':
    return total_correct_num / (epoch_size * m.batch_size),local_accuracy
  if run_type == 'test':
    return total_correct_num / (epoch_size * m.batch_size),total_correct_num,(epoch_size * m.batch_size)

  return total_correct_num / (epoch_size * m.batch_size)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(unused_args):
    if not FLAGS.data_path:
      raise ValueError("Must set --data_path to PTB data directory")
    start = time.clock()
    fout = codecs.open("//search//odin//doc2vec_jar//data//result1.txt","w","utf-8")
    fout1 = codecs.open("//search//odin//doc2vec_jar//data//logits1.txt","w","utf-8")
    
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    #load word2vec 并把句子转成id表达,这两个函数写成服务调用形式,加快实验效率
    #TODO:替换load_word2vec函数
    if FLAGS.load_word2vec == "true":
        word2id, word_embeddings = reader_zhx1.load_word2vec("/search/odin/data/liuqi/doc2vec_jar/lstm/vector.skip.win2.100.float.for_python")
#       word2id = reader_zhx.load_word2id("ptest")#word2id.txt
        (sListTrain, lListTrain), (sListValid, lListValid), (sListTest, lListTest), vocabulary,train_count,valid_count,test_count = reader_zhx1.get_data_by_word2vec(word2id, FLAGS.data_path)  

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)) as session:
      print('--- start ---')
      with tf.device('/cpu:0'):
        #initializer = tf.random_uniform_initializer(-config.init_scale,
        initializer = None
        with tf.variable_scope("model", reuse=None, initializer=initializer):
          m = PTBModel(is_training=True, config=config, word_embeddings = None)
        tf.initialize_all_variables()


      with tf.device('/gpu:7'):
        #lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        #cur_rate = max(config.learning_rate * lr_decay,0.1)
        #m.assign_lr(session, cur_rate)
        saver = tf.train.Saver()
        saver.restore(session,FLAGS.model_path)

        print(len(sListTrain))
        print(len(sListValid))
        print(len(sListTest))

        #valid_perplexity = run_epoch(session, m, sListValid, lListValid, tf.no_op(), "valid")
        #print("Valid Accuracy: %.3f" % (valid_perplexity))

        test_perplexity,total_correct_num,total_num = run_epoch(session, m, sListTest, lListTest, tf.no_op(), fout, fout1, "test")
        print("Test Accuracy: %.3f Count:%d/%d" % (test_perplexity,total_correct_num,total_num))
	end = time.clock()
	print("The function run time is : %.03f seconds" %(end-start))

if __name__ == "__main__":
  tf.app.run()
