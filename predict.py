#coding=utf-8
import os
import sys
import datetime
import tensorflow as tf
import pandas as pd
from data_helper import load_testset
from HAN_model import HAN


# Data loading params
tf.flags.DEFINE_string("input_path", 'data/test.csv', "data directory")
tf.flags.DEFINE_integer("num_classes", 6, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 50, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_sent_in_doc", 10, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_word_in_sent", 20, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("lr", 0.01, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
tf.flags.DEFINE_float("lr_decay", 0.5, "learning rate decay (default: 0.5)")
tf.flags.DEFINE_float("nepoch_no_imprv", 5, "early stopping (default: 5)")
tf.flags.DEFINE_float("nepoch_lr_decay", 2, "decay of lr if no improvement (default: 3)")
tf.flags.DEFINE_string("dir_model", "models", "path to save model files (default: word_char_models)")


FLAGS = tf.flags.FLAGS

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
test_x, _vocab_size = load_testset(FLAGS.input_path, FLAGS.max_sent_in_doc, FLAGS.max_word_in_sent)
print "data load finished"

with tf.Session() as sess:
    han = HAN(vocab_size=_vocab_size,
                    num_classes=FLAGS.num_classes,
                    embedding_size=FLAGS.embedding_size,
                    hidden_size=FLAGS.hidden_size)
#    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, 'models/' + sys.argv[1])
    with tf.name_scope('prediction'):
        pred = tf.sigmoid(han.out, name='pred')

    def pred_step(x_batch):
        feed_dict = {
            han.input_x: x_batch,
            han.max_sentence_num: FLAGS.max_sent_in_doc,
            han.max_sentence_length: FLAGS.max_word_in_sent,
            han.batch_size: FLAGS.batch_size
        }
        _pred = sess.run(pred, feed_dict)
        return _pred

    preds = []
    for i in range(0, test_x.shape[0], FLAGS.batch_size):
        x = test_x[i:i + FLAGS.batch_size]
        p = pred_step(x)
        preds.extend(p)
    sample_submission = pd.read_csv("template.csv")
    sample_submission[list_classes] = preds
    sample_submission.to_csv("submission.csv", index=False)
