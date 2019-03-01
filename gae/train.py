from __future__ import division
from __future__ import print_function

import time
import os
import matplotlib.pyplot as plt
# Train on CPU (hide GPU) due to memory constraints
from gae.optimizer import OptimizerVRAE

os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE, GCNModelVAE_test,GCNModelVAEimproved
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
tf.app.flags.DEFINE_string('f', '', 'kernel')
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.015, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 48, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 24, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'test', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Load data
adj, features = load_data(dataset_str)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
elif model_str == 'test':
    model = GCNModelVAEimproved(placeholders,num_features,num_nodes,features_nonzero)
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)
    elif model_str == 'test':
        opt = OptimizerVRAE(preds=model.reconstructions,
                            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                        validate_indices=False), [-1]),
                            x_preds_params = model.x_reconstructions,
                            x_data = tf.sparse_tensor_to_dense(placeholders['features'],
                                                               validate_indices=False),
                            model=model, num_nodes=num_nodes,
                            pos_weight=pos_weight,
                            norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []
ELBO_list = []


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def get_roc_score_test(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb,side = sess.run([model.z_mean,model.w_embed], feed_dict=feed_dict)


    def sigmoid(x):
        import warnings
        warnings.filterwarnings('error')
        try:
            return 1 / (1 + np.exp(-x))
        except RuntimeWarning:
            return 0.0

    decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gcnmodelvaeimproved/noisyresidualdecoder_1_vars/')

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    w_rec = np.dot(side,side.T)
    print(decoder_vars)
    #print((decoder_vars[-3]).eval(session=sess))
    #print((decoder_vars[-2]).eval(session=sess))
    #print((decoder_vars[-1]).eval(session=sess))


    #adj_rec = (decoder_vars[-1]).eval(session=sess)*w_rec
    #print(len(decoder_vars))
    adj_rec = (decoder_vars[-2]).eval(session=sess)*adj_rec+(decoder_vars[-1]).eval(session=sess)*w_rec
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


cost_val = []
acc_val = []
val_roc_score = []


adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]
    ELBO_list.append(avg_cost)
    roc_curr, ap_curr = get_roc_score_test(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))


interested_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'gcnmodelvaeimproved/noisyresidualdecoder_1_vars/')



print(interested_vars[0].eval(sess))
print(interested_vars[1].eval(sess))
print(interested_vars[2].eval(sess))
word_embed_mat = interested_vars[2].eval(sess)
print("Optimization Finished!")
plt.title('recall_score')
plt.plot(val_roc_score)
plt.show()
plt.title('ELBO')
plt.plot(ELBO_list)
plt.show()
roc_score, ap_score = get_roc_score_test(test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))

def read_labels(dataset_str):
    obj = np.genfromtxt("data/{}.content".format(dataset_str),
                                        dtype=np.dtype(str))
    id = []
    feat = []
    lab = []
    for i in range(len(obj)):
        id.append(obj[i][0])
        feat.append(obj[i][1:-1])
        lab.append(obj[i][-1])
    return(id, feat, lab)

_,_,labels = read_labels(dataset_str)
unique_lab = set(labels)
lab_dict = {}
for i, lab in enumerate(labels):
    try:
        lab_dict[lab].append(i)
    except KeyError:
        lab_dict[lab] = [i]


feed_dict.update({placeholders['dropout']: 0})
z_emb = sess.run(model.z_mean,feed_dict=feed_dict)
z_tsne = TSNE(n_components=2).fit_transform(z_emb)

plot_margin = 0.25
x0, x1, y0, y1 = plt.axis()
plt.axis((x0 - plot_margin,
          x1 + plot_margin,
          y0 - plot_margin,
          y1 + plot_margin))

plt.title('node embedded - t-SNE')
for key in lab_dict:
    plt.plot(z_tsne[lab_dict[key], 0], z_tsne[lab_dict[key], 1], '.', label=str(key))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


