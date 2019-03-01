from initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class ImprovedGCN(Layer):
    """GCN with improved featured preserving structure"""
    def __init__(self, input_dim, output_dim, adj,  dropout=0., act=tf.nn.relu, **kwargs):
        super(ImprovedGCN,self).__init__(**kwargs)
        with tf.variable_scope(self.name+'_vars'):
            self.vars['weights_orig'] = weight_variable_glorot(input_dim, output_dim, name="weights_orig")
            self.vars['weights_new'] = weight_variable_glorot(input_dim, output_dim, name="weights_new")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True


    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x,1-self.dropout)
        x_orig = tf.matmul(x,self.vars['weights_orig'])
        x_orig = tf.sparse_tensor_dense_matmul(self.adj,x_orig)
        x_new = tf.matmul(x, self.vars['weights_new'])
        outputs = self.act(x_orig+x_new)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class ImprovedGCNsp(Layer):
    """GCN with improved featured preserving structure"""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(ImprovedGCNsp, self).__init__(**kwargs)
        with tf.variable_scope(self.name+'_vars'):
            self.vars['weights_orig'] = weight_variable_glorot(input_dim, output_dim, name="weights_orig")
            self.vars['weights_new'] = weight_variable_glorot(input_dim, output_dim, name="weights_new")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x,1-self.dropout,self.features_nonzero)
        x_orig = tf.sparse_tensor_dense_matmul(x,self.vars['weights_orig'])
        x_orig = tf.sparse_tensor_dense_matmul(self.adj,x_orig)
        x_new = tf.sparse_tensor_dense_matmul(x, self.vars['weights_new'])
        outputs = self.act(x_orig+x_new)
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


class NoisyResidualDecoder(Layer):
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(NoisyResidualDecoder, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_mu'] = weight_variable_glorot(input_dim, output_dim, name='weights_mu')
            self.vars['weights_sigma'] = weight_variable_glorot(input_dim,output_dim,name='weights_sigma')
            self.vars['weights_embed'] = weight_variable_glorot(output_dim,32,name='weights_embed')
            self.vars['beta_zzT'] = tf.get_variable('beta_zzT',shape=[],initializer=tf.initializers.random_uniform(0,0.5))
            self.vars['beta_wwT'] = tf.get_variable('beta_wwT',shape=[],initializer=tf.initializers.random_uniform(0,1))
            #self.vars['beta_0'] = tf.get_variable('beta_0',shape=[],initializer=tf.contrib.layers.xavier_initializer())
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        x_mu = tf.matmul(inputs, self.vars['weights_mu'])
        x_log_sigma = tf.matmul(inputs, self.vars['weights_sigma'])
        x_sample = tf.distributions.Normal(loc=x_mu,scale=tf.exp(x_log_sigma)).sample()
        x = x_sample
        w_embed = (tf.matmul(x, self.vars['weights_embed'])) # because the features are words, we should
                                                                             # do a layer of word embedding
        #w_embed = tf.layers.batch_normalization(inputs = w_embed,training=True)
        wT = tf.transpose(w_embed)
        wwT = tf.matmul(w_embed,wT)
        zT = tf.transpose(inputs)
        zzT = tf.matmul(inputs, zT)
        #outputs = self.act(tf.reshape(tf.scalar_mul(self.vars['beta_wwT'], wwT), [-1]))
        outputs = self.act(tf.reshape(tf.scalar_mul(self.vars['beta_zzT'],zzT) + tf.scalar_mul(self.vars['beta_wwT'],wwT),[-1]))
        return outputs,(x_mu,x_log_sigma),w_embed

