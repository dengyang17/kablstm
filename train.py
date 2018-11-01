#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from input_helpers import InputHelper
from kablstm import KABLSTM
from tensorflow.contrib import learn
import gzip
from random import random
import argparse

# python train.py -d dataset

parser = argparse.ArgumentParser()
parser.add_argument("train", help="train dataset") #Wiki or Trec
parser.add_argument("dev", help="dev dataset")
parser.add_argument("model", help="model path")
parser.add_argument("-l","--length", help="sequence length", default=30)
parser.add_argument("-e","--entity_embed", help="entity embed", default='deepwalk')
parser.add_argument("-m","--mode", help="mode")
args = parser.parse_args()
train = args.train
dev = args.dev
model = args.model
max_document_length=int(args.length)
entity_embed=args.entity_embed
mode = args.mode
n_entity=5


# Parameters
# ==================================================
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_files", "../../data/%s-train.txt" % train, "training file (default: None)")
tf.flags.DEFINE_string("training_labeled_files", "../../data/%s-train.txt.labeled" % train, "labeled training file (default: None)")
tf.flags.DEFINE_string("dev_files", "../../data/%sQA-test.txt" % dev, "dev file (default: None)")
tf.flags.DEFINE_string("dev_labeled_files", "../../data/%sQA-test.txt.labeled" % dev, "labeled dev file (default: None)")
tf.flags.DEFINE_string("embedding_file", "../../data/glove.6B.300d.txt", "embedding file (default: None)")
tf.flags.DEFINE_string("entity_embedding_file", "../../embed/fb5m-%s.%s" % (dev,entity_embed), "entity embedding file (default: None)")
tf.flags.DEFINE_integer("entity_embedding_dim", 64, "Dimensionality of entity embedding (default: 300)")
tf.flags.DEFINE_integer("hidden_units", 200, "Number of hidden units in softmax regression layer (default:50)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_files==None:
    print "Input Files List is empty. use --training_files argument."
    exit()
 


inpH = InputHelper()
train_set, dev_set, vocab_processor,sum_no_of_batches = inpH.getDataSets(FLAGS.training_files, FLAGS.training_labeled_files, 
                                                        FLAGS.dev_files, FLAGS.dev_labeled_files, max_document_length, 10, FLAGS.batch_size)
embedding_matrix = inpH.getEmbeddings(FLAGS.embedding_file,FLAGS.embedding_dim)
entity_embedding_matrix = inpH.getEntityEmbeddings(FLAGS.entity_embedding_file,FLAGS.entity_embedding_dim)
entity_vocab_size = len(entity_embedding_matrix)
if mode == 'random':
    entity_embedding_matrix = np.asarray(None)

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        siameseModel = KABLSTM(
            sequence_length=max_document_length,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            hidden_units=FLAGS.hidden_units,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            batch_size=FLAGS.batch_size,
            embedding_matrix=embedding_matrix,
            entity_embedding_matrix=entity_embedding_matrix,
            entity_embedding_dim=FLAGS.entity_embedding_dim,
            entity_vocab_size=entity_vocab_size,
            n_entity=n_entity,
            mode=mode)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(5e-4)
        print("initialized siameseModel object")
    
    grads_and_vars=optimizer.compute_gradients(siameseModel.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", model))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", siameseModel.loss)
    acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    # Write vocabulary
    vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    def train_step(x1_batch, x2_batch, ent_x1_batch, ent_x2_batch, y_batch, add_fea_batch):
        """
        A single training step
        """
        
        feed_dict = {
            siameseModel.input_x1: x1_batch,
            siameseModel.input_x2: x2_batch,
            siameseModel.ent_x1: ent_x1_batch,
            siameseModel.ent_x2: ent_x2_batch,
            siameseModel.input_y: y_batch,
            siameseModel.add_fea: add_fea_batch,
            siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
        }

        _, input1, embed, step, loss, accuracy, dist, sim, summaries = sess.run(
            [tr_op_set, siameseModel.input_x1, siameseModel.embedded_chars1, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.soft_prob, siameseModel.predictions, train_summary_op],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)
        #print y_batch, dist, sim
        #print input1[0], embed[0]

    def dev_step(x1_batch, x2_batch, ent_x1_batch, ent_x2_batch, y_batch, add_fea_batch,x1_dev_text_batch,x2_dev_text_batch):
        """
        A single training step
        """ 
        feed_dict = {
            siameseModel.input_x1: x1_batch,
            siameseModel.input_x2: x2_batch,
            siameseModel.ent_x1: ent_x1_batch,
            siameseModel.ent_x2: ent_x2_batch,
            siameseModel.input_y: y_batch,
            siameseModel.add_fea:add_fea_batch,
            siameseModel.dropout_keep_prob: 1.0,
        }
        
        step, loss, accuracy, dist, sim, summaries = sess.run([global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.soft_prob, siameseModel.predictions, dev_summary_op],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        dev_summary_writer.add_summary(summaries, step)
        #print y_batch, dist, sim
        return accuracy, dist

    # Generate batches
    batches=inpH.batch_iter(
                list(zip(train_set[0], train_set[1], train_set[2], train_set[3], train_set[4], train_set[5])), FLAGS.batch_size, FLAGS.num_epochs)

    ptr=0
    max_validation_acc=0.0
    max_map = 0.0
    max_mrr = 0.0
    for nn in xrange(sum_no_of_batches*FLAGS.num_epochs):
        batch = batches.next()
        if len(batch)<1:
            continue
        x1_batch,x2_batch, ent_x1_batch, ent_x2_batch, y_batch, add_fea_batch  = zip(*batch)
        if len(y_batch)<1:
            continue

        train_step(x1_batch,x2_batch, ent_x1_batch, ent_x2_batch, y_batch, add_fea_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc=0.0
        
        if current_step % FLAGS.evaluate_every == 0:
            all_predictions = []
            all_y = []
            all_q = []
            print("\nEvaluation:")
            dev_batches = inpH.batch_iter(list(zip(dev_set[0],dev_set[1],dev_set[2],dev_set[3],dev_set[4],dev_set[5],dev_set[6],dev_set[7])), FLAGS.batch_size, 1)
            for db in dev_batches:
                if len(db)<1:
                    continue
                x1_dev_b,x2_dev_b,ent_x1_dev_b,ent_x2_dev_b,y_dev_b,add_fea_dev_b,x1_dev_text_b,x2_dev_text_b = zip(*db)
                if len(y_dev_b)<1:
                    continue
                acc,batch_predictions = dev_step(x1_dev_b,x2_dev_b,ent_x1_dev_b,ent_x2_dev_b,y_dev_b,add_fea_dev_b,x1_dev_text_b,x2_dev_text_b)
                sum_acc = sum_acc + acc
                all_predictions = np.concatenate([all_predictions, [x[1] for x in batch_predictions]])
                all_y.extend(y_dev_b)
                all_q.extend(x1_dev_text_b)
            print("")
            result = {}
            for i in range(len(all_y)):
                if not result.has_key(all_q[i]):
                    result[all_q[i]] = []
                result[all_q[i]].append((all_predictions[i], all_y[i]))
            rank_all = 0
            count = 0

            for key in result.keys():
                answers = sorted(result[key], key=lambda x:x[0], reverse=True)
                print key
                print answers[:10]
                rank = 0
                for i in range(len(answers)):
                    if answers[i][1] == 1:
                        rank = 1.0/(i+1.0)
                        break
                if rank != 0:
                    rank_all += rank
                count +=1


            print 'MRR:' + str(rank_all/count)
            MRR = rank_all/count

            MAP = 0
            count = 0
            for key in result.keys():
                answers = sorted(result[key], key=lambda x:x[0], reverse=True)
                rank = 0
                rank_all = 0
                for i in range(len(answers)):
                    if answers[i][1] == 1:
                        rank += 1.0
                        rank_all += rank/(i+1.0)
                if rank != 0:
                    MAP += rank_all/rank
                count +=1
            print 'MAP:' + str(MAP/count)
        if current_step % FLAGS.checkpoint_every == 0:
            print sum_acc
            #if sum_acc >= max_validation_acc:
            
            if MRR>=max_mrr:
                #max_validation_acc = sum_acc
                max_map = MAP
                max_mrr = MRR

                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc, checkpoint_prefix))
            print max_map/count,max_mrr