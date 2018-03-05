#coding=utf-8
import os
import json
import pickle
import nltk
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
import pandas as pd

# 使用nltk分词分句器
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')   # sentence segmentation
word_tokenizer = WordPunctTokenizer()   # word segmentation
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def build_vocab(vocab_path, input_file):
    # 记录每个单词及其出现的频率
    word_freq = defaultdict(int)
    # 读取数据集，并进行分词，统计每个单词出现次数，保存在word freq中
    df = pd.read_csv(input_file, header=0, index_col=0)
    docnum = df.shape[0]
    for line in df["comment_text"].fillna("davidie").values:
        words = word_tokenizer.tokenize(line.decode('utf-8'))
        for word in words:
            word_freq[word] += 1
    print "load finished"
    # 构建vocablary，并将出现次数小于5的单词全部去除，视为UNKNOWN
    vocab = {}
    i = 1
    vocab['UNKNOW_TOKEN'] = 0
    for word, freq in word_freq.items():
        if freq > 15:
            vocab[word] = i
            i += 1
    # 将词汇表保存下来
    with open(vocab_path, 'wb') as g:
        pickle.dump(vocab, g)
        print "In total %d words" %len(vocab)
        print "vocab save finished"
    return docnum, vocab

def load_dataset(input_file, max_sent_in_doc, max_word_in_sent):
    input_data_path = input_file[0:-4] + "_data.pickle"
    vocab_path = input_file[0:-4] + "_vocab.pickle"
    doc_num, vocab = build_vocab(vocab_path, input_file)
    num_classes = 6
    UNKNOWN = 0

    data_x = np.zeros([doc_num,max_sent_in_doc,max_word_in_sent])
    data_y = []

    #将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
    # 不够的补零，多余的删除，并保存到最终的数据集文件之中
    df = pd.read_csv(input_file, header=0, index_col=0)
    for line_index, line in enumerate(df["comment_text"].fillna("davidie").values):
        sents = sent_tokenizer.tokenize(line.decode('utf-8'))   # here use .decode('utf-8') to avoid 'ascii can't decode 0xc2' error
        doc = np.zeros([max_sent_in_doc, max_word_in_sent])
        for i, sent in enumerate(sents):
            if i < max_sent_in_doc:
                word_to_index = np.zeros([max_word_in_sent],dtype=int)
                for j, word in enumerate(word_tokenizer.tokenize(sent)):
                    if j < max_word_in_sent:
                            word_to_index[j] = vocab.get(word, UNKNOWN)
                doc[i] = word_to_index
        data_x[line_index] = doc
    data_y= df[list_classes].values
    print 'Total number of samples: %d' %len(data_x) #229907

    length = len(data_x)
    train_x, dev_x = data_x[:int(length*0.9)], data_x[int(length*0.9)+1 :]
    train_y, dev_y = data_y[:int(length*0.9)], data_y[int(length*0.9)+1 :]
    return train_x, train_y, dev_x, dev_y, len(vocab)

def load_testset(input_file, max_sent_in_doc, max_word_in_sent):
    vocab_file = open('data/train_vocab.pickle', 'rb')
    vocab = pickle.load(vocab_file)
    df = pd.read_csv(input_file, header=0, index_col=0)
    doc_num = df.shape[0]
    UNKNOWN = 0

    data_x = np.zeros([doc_num,max_sent_in_doc,max_word_in_sent])
    for line_index, line in enumerate(df["comment_text"].fillna("davidie").values):
        sents = sent_tokenizer.tokenize(line.decode('utf-8'))   # here use .decode('utf-8') to avoid 'ascii can't decode 0xc2' error
        doc = np.zeros([max_sent_in_doc, max_word_in_sent])
        for i, sent in enumerate(sents):
            if i < max_sent_in_doc:
                word_to_index = np.zeros([max_word_in_sent],dtype=int)
                for j, word in enumerate(word_tokenizer.tokenize(sent)):
                    if j < max_word_in_sent:
                            word_to_index[j] = vocab.get(word, UNKNOWN)
                doc[i] = word_to_index
        data_x[line_index] = doc
    print 'Total number of test samples: %d' %len(data_x) #229907
    return data_x, len(vocab)

if __name__ == '__main__':
    load_dataset("data/train.csv", 10, 20)

