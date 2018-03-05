#coding=utf-8
import datetime
import tensorflow as tf
import time
import os
from data_helper import load_dataset
from HAN_model import HAN


# Data loading params
tf.flags.DEFINE_string("input_path", 'data/train.csv', "data directory")
tf.flags.DEFINE_integer("num_classes", 6, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 50, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_sent_in_doc", 10, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_word_in_sent", 20, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("lr", 0.01, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
tf.flags.DEFINE_float("lr_decay", 0.5, "learning rate decay (default: 0.5)")
tf.flags.DEFINE_float("nepoch_no_imprv", 3, "early stopping (default: 5)")
tf.flags.DEFINE_float("nepoch_lr_decay", 2, "decay of lr if no improvement (default: 3)")
tf.flags.DEFINE_string("dir_model", "models", "path to save model files (default: word_char_models)")


FLAGS = tf.flags.FLAGS

train_x, train_y, dev_x, dev_y , _vocab_size= load_dataset(FLAGS.input_path, FLAGS.max_sent_in_doc, FLAGS.max_word_in_sent)
print "training samples: %d" %train_x.shape[0]
print "dev samples: %d" %dev_x.shape[0]
print "data load finished"

with tf.Session() as sess:
    han = HAN(vocab_size=_vocab_size,
                    num_classes=FLAGS.num_classes,
                    embedding_size=FLAGS.embedding_size,
                    hidden_size=FLAGS.hidden_size)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=han.input_y,
                                                                      logits=han.out,
                                                                      name='loss'))

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

#    print 'Global Variables:'
#    for elem in tf.global_variables():
#        print elem
    print 'Trainable Variables:'
    for elem in tf.trainable_variables():
        print elem

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)

    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: FLAGS.max_sent_in_doc,
            han.max_sentence_length: FLAGS.max_word_in_sent,
            han.batch_size: FLAGS.batch_size
        }
        _, step, summaries, cost = sess.run([train_op, global_step, train_summary_op, loss], feed_dict)
        time_str = str(int(time.time()))
        train_summary_writer.add_summary(summaries, step)
        return cost

    def dev_step(x_batch, y_batch, writer=None):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: FLAGS.max_sent_in_doc,
            han.max_sentence_length: FLAGS.max_word_in_sent,
            han.batch_size: FLAGS.batch_size
        }
        step, summaries, cost = sess.run([global_step, dev_summary_op, loss], feed_dict)
        time_str = str(int(time.time()))
        if writer:
            writer.add_summary(summaries, step)
        return cost

    best_loss = 10.0
    nepoch_no_imprv = 0 # for early stopping
    nepoch_lr_decay = 0 # for lr decay
    for epoch in range(FLAGS.num_epochs):
        starttime = datetime.datetime.now()
        train_loss = 0.0
        _train_step = 0
        dev_loss = 0.0
        _dev_step = 0
        for i in range(0, train_x.shape[0], FLAGS.batch_size):
            x = train_x[i:i + FLAGS.batch_size]
            y = train_y[i:i + FLAGS.batch_size]
            _loss = train_step(x, y)
            train_loss += _loss
            _train_step += 1
        train_loss = train_loss * 1.0 / _train_step
        for i in range(0, dev_x.shape[0], FLAGS.batch_size):
            x = dev_x[i:i + FLAGS.batch_size]
            y = dev_y[i:i + FLAGS.batch_size]
            _loss = dev_step(x, y, dev_summary_writer)
            dev_loss += _loss
            _dev_step += 1
        dev_loss = dev_loss * 1.0 / _dev_step
        endtime = datetime.datetime.now()
        if dev_loss < best_loss:
            nepoch_no_imprv = 0
            nepoch_lr_decay = 0
            saver.save(sess, FLAGS.dir_model+"/sentiment", 
                    global_step = epoch+1, write_meta_graph = False)
            best_loss = dev_loss
            print("Epoch %d/%d, train_loss: %f, dev_loss: %f, time_cost: %d - new best model" %(epoch+1, FLAGS.num_epochs, train_loss, dev_loss, (endtime-starttime).seconds))
        else:
            nepoch_no_imprv += 1
            nepoch_lr_decay += 1
            if nepoch_lr_decay >= FLAGS.nepoch_lr_decay:
                FLAGS.lr *= FLAGS.lr_decay # decay learning rate
                nepoch_lr_decay = 0
            if nepoch_no_imprv >= FLAGS.nepoch_no_imprv:
                print("Epoch %d/%d, train_loss: %f, dev_loss: %f, lr: %f, time_cost: %d - early stropping" %(epoch+1, FLAGS.num_epochs, train_loss, dev_loss, FLAGS.lr, (endtime-starttime).seconds))
                break
            print("Epoch %d/%d, train_loss: %f, dev_loss: %f, lr: %f, time_cost: %d" %(epoch+1, FLAGS.num_epochs, train_loss, dev_loss, FLAGS.lr, (endtime-starttime).seconds))
