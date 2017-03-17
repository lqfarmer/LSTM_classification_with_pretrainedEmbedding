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

# pylint: disable=unused-import,g-bad-import-order

"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time
import struct

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile

TRAIN_FILE='train.txt'
TEST_FILE = 'test.txt'
# VALID_FILE='valid.txt'
#TEST_FILE='test.txt'


def _read_words(filename):
  with gfile.GFile(filename, "r") as f:
    lines = f.read().split('\n')
    sentencesList = []
    labelsList = []
    for line in lines:
        if len(line) == 0:
            break
        parts = line.split('\t')
        sentencesList.append(parts[0])
        if parts[1] == "0 1":
            labelsList.append([0, 1])
        else:
            labelsList.append([1, 0])  
    sList = []
    allWordsList = []
    for s in sentencesList:
        words = s.split(' ')
        sList.append(words)
        allWordsList += words
    lList = []
    count = 0
#     for l in labelsList:
#         tag01 = l.split(' ')        
#         label = tag01.index('1') #[gaoteng]: No Multilabel now.
#         if label == 0:
#             count += 1
#         lList.append(label)
    return allWordsList, sList, np.array(labelsList),count
    # replace("\n", "<eos>").split() [gaoteng]: Add <eos> or not? For better accuracy?


def _build_vocab(allWordsList, vocab_size):  
  counter = collections.Counter(allWordsList)
  count_pairs = sorted(counter.items(), key=lambda x: -x[1])

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words[0:vocab_size], range(vocab_size)))

  return word_to_id


def _get_vocabulary_size(filename):
  allWordsList, _, _,_ = _read_words(filename)
  counter = collections.Counter(allWordsList)
  return len(counter)


def _list_to_word_ids(sList, word_to_id,max_sentence_length):
  unkown_word_id = len(word_to_id)
  word_ids = []
  count  = 0
  if max_sentence_length == len(sList):
    for word in sList:
        if word in word_to_id: 
            word_ids.append(word_to_id[word])
        else:
            word_ids.append(unkown_word_id)
  elif len(sList) < max_sentence_length:
    for word in sList:
        count += 1
        if word in word_to_id: 
            word_ids.append(word_to_id[word])
        else:
            word_ids.append(unkown_word_id)
    while count < max_sentence_length:
        count += 1
        word_ids.append(unkown_word_id)         
  return word_ids


def file_to_ids(data_full_path, word_to_id,max_sentence_length):
  _, sList, lList,_ = _read_words(data_full_path)  
  idsList = []

  for s in sList:
      ids = _list_to_word_ids(s, word_to_id, max_sentence_length)
      idsList.append(ids)
  return (idsList, lList)

def load_word2vec(path):
  fileData = open(path, 'rb')
  vocab_size, = struct.unpack("i", fileData.read(4))
  #vocab_size = int(vocab_size / 3) #[gaoteng]: Max graph size can't exceed 2GB!!!
  #vocab_size = int(50) #[gaoteng]: Max graph size can't exceed 2GB!!!
  dim, = struct.unpack("i", fileData.read(4))
  word2id = {}
  word_embeddings = np.zeros((vocab_size, dim), dtype = np.float32)
  print(word_embeddings.shape)

  for i in range(vocab_size - 1):
    word_len, = struct.unpack("i", fileData.read(4))
    word_str, = struct.unpack(str(word_len) + "s", fileData.read(word_len))
    word2id[word_str] = i
    for j in range(dim):
        elem_value, = struct.unpack("f", fileData.read(4))
        word_embeddings[i, j] = elem_value    
        #print("(%f, %f)" % (word_embeddings[i, j], elem_value))
  for i in range(dim):
    word_embeddings[vocab_size - 1, i] = 0.0
  return word2id, word_embeddings

def get_data_by_word2vec(word_to_id, data_path=None):
  train_path = os.path.join(data_path, TRAIN_FILE)
  test_path = os.path.join(data_path, TEST_FILE)
  vocabulary = len(word_to_id) + 1 # +1: <unkown_word> (low-frequency)
  _, tsList, tlList,_ = _read_words(train_path)
  _, tesList, telList,_ = _read_words(test_path)
  max_sentence_length1 = max([len(x) for x in tsList])
  max_sentence_length2 = max([len(x) for x in tesList])
  max_sentence_length = 0
  if max_sentence_length1 >= max_sentence_length2:
      max_sentence_length = max_sentence_length1
  else:
      max_sentence_length = max_sentence_length2

  print(" Original Max_sentence_length is: {:d}".format(max_sentence_length))
  #if(max_sentence_length < 22):
  max_sentence_length = 35
  train_idsList, train_lList = file_to_ids(train_path, word_to_id,max_sentence_length)
  test_idsList,  test_lList  = file_to_ids(test_path, word_to_id,max_sentence_length)
  _,_,_,train_count = _read_words(train_path)
  _,_,_,test_count = _read_words(test_path)
  print ("The value of train count:")
  print (train_count)
  print ("The value of test count:")
  print (test_count)
  return (train_idsList, train_lList),  (test_idsList, test_lList), vocabulary,train_count, test_count,max_sentence_length


def ptb_iterator(sList, lList, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  # print(num_steps) #[gaoteng]: max-sentence length = 30
  sentenceCount = len(sList)
  epoch_size = sentenceCount // batch_size
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
  validSentenceCount = epoch_size * batch_size
  data = np.zeros([validSentenceCount, num_steps], dtype=np.int32)
  length = np.zeros([validSentenceCount], dtype=np.int32)
  for i in range(validSentenceCount):
    arr = np.asarray(sList[i])
    length[i] = min(num_steps, len(sList[i])) #[gaoteng]: Over-length sentence is cut here.
    arr.resize((num_steps))
    data[i] = arr      
  
  for i in range(epoch_size):
  #for i in range(10):
    x = data[i * batch_size : (i + 1) * batch_size, :]
    y = lList[i * batch_size : (i + 1) * batch_size]
    yield (x, length[i * batch_size : (i + 1) * batch_size], y)

def persist_store(ifile_path,ofile_path):
  fileData = open(ifile_path, 'rb')
  vocab_size, = struct.unpack("i", fileData.read(4))
  dim, = struct.unpack("i", fileData.read(4))
  word2id = {}
  word_embeddings = np.zeros((vocab_size, dim), dtype = np.float32)
  print(word_embeddings.shape)

  for i in range(vocab_size - 1):
    word_len, = struct.unpack("i", fileData.read(4))
    word_str, = struct.unpack(str(word_len) + "s", fileData.read(word_len))
    word2id[word_str] = i
    for j in range(dim):
        elem_value, = struct.unpack("f", fileData.read(4))
        word_embeddings[i, j] = elem_value    
        #print("(%f, %f)" % (word_embeddings[i, j], elem_value))
  for i in range(dim):
    word_embeddings[vocab_size - 1, i] = 0.0

  with open(ofile_path,'w') as ofile:
    for k,v in word2id.items():
        print ('%s\t%d'%(k,v),file=ofile)
  return word2id, word_embeddings

def load_word2id(file_path):
  w2id = dict()
  with open(file_path,'r') as ifile:
    for line in ifile:
      lst = line.strip().split('\t')
      if len(lst) != 2:
        continue
      word,id_ = lst[0],int(lst[1])
      w2id[word] = id_
  return w2id


if __name__ == '__main__':
  #words_count = _get_vocabulary_size(os.path.join(data_path,TRAIN_FILE))
  #print(words_count)
  #(sListTrain, lListTrain), (sListValid, lListValid), (sListTest, lListTest), vocabulary = ptb_raw_data("")
  #print(vocabulary)
  #trainLog = open("./zhangxiao/trainLog.txt", "w")
  #trainLog.writelines(["%s\n" % item  for item in sListTrain])
  #trainLog.close()
  #validLog = open("./zhangxiao/validLog.txt", "w")
  #validLog.writelines(["%s\n" % item  for item in sListValid])
  #validLog.close()
  #testLog = open("./zhangxiao/testLog.txt", "w")
  #testLog.writelines(["%s\n" % item  for item in lListTest])
  #testLog.close()    
  '''
  for step, (data, length, label) in enumerate(ptb_iterator(sListTrain, lListTrain, 20, 30)):
      print(step)
      print(data)
      print(length)
      print(label)
      print("________________________________")
  '''
  persist_store('vector.skip.win2.100.float.for_python','word2id.txt')

