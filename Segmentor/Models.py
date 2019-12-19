# coding: utf-8
# -------------------- Package import -------------------- #
# System operation packages
import os
import pickle
pickle_options = { 'encoding': 'latin-1' }
    
# Theano and numpy
import numpy as np
import theano
import theano.tensor as T

# -------------------- Tool functions -------------------- #
def _get_shape(i, o, keepdims):
    """
    Whether to keep the axis if the dimension equals 1. 
    """
    if (i == 1 or o == 1) and not keepdims:
        return (max(i,o),)
    else:
        return (i, o)
    
def _slice(tensor, size, i):
    """
    Gets slice of columns of the tensor.
    """
    if tensor.ndim == 2:
        return tensor[:, i*size:(i+1)*size]
    elif tensor.ndim == 1:
        return tensor[i*size:(i+1)*size]
    else:
        raise NotImplementedError("Tensor should be 1 or 2 dimensional")
        
def load(file_path, minibatch_size, x, p=None):
    """
    Load the model.
    """
    try:
        from . import Models
    except ImportError:
        import Models
        
    with open(file_path, 'rb') as f:
        state = pickle.load(f, **pickle_options)
    Model = getattr(Models, state["type"])
    rng = np.random
    rng.set_state(state["random_state"])
    # Load the model
    net = Model(rng=rng,
                x=x,
                minibatch_size=minibatch_size,
                n_hidden=state["n_hidden"],
                x_vocabulary=state["x_vocabulary"],
                y_vocabulary=state["y_vocabulary"]) 
    # Load the parameter of optimizor
    for net_param, state_param in zip(net.params, state["params"]):
        net_param.set_value(state_param, borrow=True)
    gsums = [theano.shared(gsum) for gsum in state["gsums"]] if state["gsums"] else None

    return net, (gsums, state["learning_rate"], state["validation_ppl_history"], state["epoch"], rng)

# -------------------- Model Initialization -------------------- #
def weights_const(i, o, name, const, keepdims=False):
    """
    Initalize weights to a constant.
    """
    W_values = np.ones(_get_shape(i, o, keepdims)).astype(theano.config.floatX) * const
    return theano.shared(value=W_values, name=name, borrow=True)

def weights_Glorot(i, o, name, rng, keepdims=False):
    """
    Glorot uniform weights initialization.
    """
    d = np.sqrt(6. / (i + o))
    W_values = rng.uniform(low=-d, high=d, size=_get_shape(i, o, keepdims)).astype(theano.config.floatX)
    return theano.shared(value=W_values, name=name, borrow=True)

# -------------------- Gated Recurrent Unit -------------------- #
class GRULayer(object):
    """
    Gated Recurrent Unit which keeps memory of long term.
    """
    def __init__(self, rng, n_in, n_out, minibatch_size):
        """
        Input:
            rng:                Numpy random seed
            n_in:               Dimension of input
            n_out:              Dimension of output
            minibatch_size:     Mini-batch size
        """
        super(GRULayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        
        # Initial hidden state to 0: N * n_output
        self.h0 = theano.shared(value=np.zeros((minibatch_size, n_out)).astype(theano.config.floatX), name='h0', borrow=True)

        # Gate parameters: W_x = [W_xr, W_xz]
        self.W_x = weights_Glorot(n_in, n_out*2, 'W_x', rng)
        self.W_h = weights_Glorot(n_out, n_out*2, 'W_h', rng)
        self.b = weights_const(1, n_out*2, 'b', 0)
        
        # Input parameters
        self.W_x_h = weights_Glorot(n_in, n_out, 'W_x_h', rng)
        self.W_h_h = weights_Glorot(n_out, n_out, 'W_h_h', rng)
        self.b_h = weights_const(1, n_out, 'b_h', 0)
        
        self.params = [self.W_x, self.W_h, self.b, self.W_x_h, self.W_h_h, self.b_h]

    def step(self, x_t, h_tm1):
        """
        Forward: h_tm = GRU(x_t, h_tm1)
        
        Input:
            x_t:        size: (mini-batchSize * n_in), only one sample was input because of 'theano.scan'
            h_tm1:      size: (mini-batchsize * n_out)
        """
        # Compute reset gate and update gate togather and silce to two
        # Dot product between each element of x_t and W_x
        rz = T.nnet.sigmoid(T.dot(x_t, self.W_x) + T.dot(h_tm1, self.W_h) + self.b)
        r = _slice(rz, self.n_out, 0)
        z = _slice(rz, self.n_out, 1)

        h = T.tanh(T.dot(x_t, self.W_x_h) + T.dot(h_tm1 * r, self.W_h_h) + self.b_h)
        h_t = z * h_tm1 + (1. - z) * h
        return h_t

# -------------------- Model -------------------- #
class GRU(object):
    """
    Bidirectional gated recurrent neural network model with attention mechanism. 
    """
    def __init__(self, rng, x, minibatch_size, n_hidden, x_vocabulary, y_vocabulary):
        """
        Input:
            rng:                Numpy random seed
            x:                  Training data with size (sequenceLen * mini-batchSize)
            minibatch_size:     Mini-batch size
            n_hidden:           Hidden layer size
            x_vocabulary:       Vocabulary dictionary of words
            y_vocabulary:       Vocabulary dictionary of punctuations
        """
        # -------------------- Model component defination -------------------- #
        x_vocabulary_size = len(x_vocabulary)
        y_vocabulary_size = len(y_vocabulary)
        self.n_hidden = n_hidden
        self.x_vocabulary = x_vocabulary
        self.y_vocabulary = y_vocabulary

        # ---------- Input component ---------- #
        # Word embeding: share embeddings between forward and backward model
        n_emb = n_hidden
        self.We = weights_Glorot(x_vocabulary_size, n_emb, 'We', rng) # Each row represents a vector
        self.GRU_f = GRULayer(rng=rng, n_in=n_emb, n_out=n_hidden, minibatch_size=minibatch_size)
        self.GRU_b = GRULayer(rng=rng, n_in=n_emb, n_out=n_hidden, minibatch_size=minibatch_size)
        # Bi-directional recurrence
        def input_recurrence(x_f_t, x_b_t, h_f_tm1, h_b_tm1):
            h_f_t = self.GRU_f.step(x_t=x_f_t, h_tm1=h_f_tm1)
            h_b_t = self.GRU_b.step(x_t=x_b_t, h_tm1=h_b_tm1)
            return [h_f_t, h_b_t]
        
        # ---------- Attention component ---------- #
        n_attention = n_hidden * 2 # to match concatenated forward and reverse model states
        self.Wa_h = weights_Glorot(n_hidden, n_attention, 'Wa_h', rng)    # Output model previous hidden state to attention model weights
        self.Wa_c = weights_Glorot(n_attention, n_attention, 'Wa_c', rng) # Contexts to attention model weights
        self.ba = weights_const(1, n_attention, 'ba', 0)
        self.Wa_y = weights_Glorot(n_attention, 1, 'Wa_y', rng)           # Gives weights to contexts

        # ---------- Output component ---------- #
        # The GRU used in attention machanism (decoder)
        self.GRU = GRULayer(rng=rng, n_in=n_hidden*2, n_out=n_hidden, minibatch_size=minibatch_size)
        self.Wy = weights_const(n_hidden, y_vocabulary_size, 'Wy', 0)
        self.by = weights_const(1, y_vocabulary_size, 'by', 0)

        # ---------- Parameters fusion component ---------- #
        self.Wf_h = weights_const(n_hidden, n_hidden, 'Wf_h', 0)
        self.Wf_c = weights_const(n_attention, n_hidden, 'Wf_c', 0)
        self.Wf_f = weights_const(n_hidden, n_hidden, 'Wf_f', 0)
        self.bf = weights_const(1, n_hidden, 'by', 0)
        
        def output_recurrence(x_t, h_tm1, Wa_c, ba, Wa_h, Wa_y, Wf_h, Wf_c, Wf_f, bf, Wy, by, context):
            # Attention model; Context has 3 dims;
            # Compute the similarity of each input and the current output (so h_tm1 can be broadcast and same each row)
            h_a = T.tanh((T.dot(context, Wa_c) + ba) + T.dot(h_tm1, Wa_h))
            alphas = T.exp(T.dot(h_a, Wa_y))
            alphas = alphas.reshape((alphas.shape[0], alphas.shape[1]))    # Drop 2-axis (sized 1)
            alphas = alphas / alphas.sum(axis=0, keepdims=True)            # 0-axis is time steps
            weighted_context = (context * alphas[:,:,None]).sum(axis=0)
            # Late fusion: y_t = g(hf_t) = g(f(weighted_context))
            h_t = self.GRU.step(x_t=x_t, h_tm1=h_tm1)
            lfc = T.dot(weighted_context, Wf_c)                           # late fused context
            fw = T.nnet.sigmoid(T.dot(lfc, Wf_f) + T.dot(h_t, Wf_h) + bf) # fusion weights
            hf_t = lfc * fw + h_t                                         # weighted fused context + hidden state
            z = T.dot(hf_t, Wy) + by
            y_t = T.nnet.softmax(z)
            return [h_t, hf_t, y_t, alphas]
        # --------------------------------------------------------------------- #
        # -------------------- Pipeline -------------------- #
        # Each row of x represents one time t; Use index of word embeding matrix(We) to embed each word
        # self.We[x.flatten()] = [X_batch0_t0_emb X_batch1_t0_emb X_batch2_t0_emb ...]
        x_emb = self.We[x.flatten()].reshape((x.shape[0], minibatch_size, n_emb)) # (sequenceLen * mini-batchSize) * wordEmbed
        # 'Theano.scan' interatively input single data of 'x_emb' 
        # 'Theano.scan' gather all the results in a list, h_f_t = [h_f_1, h_f_2, ...]
        [h_f_t, h_b_t], _ = theano.scan(fn=input_recurrence,
                                        sequences=[x_emb, x_emb[::-1]], 
                                        outputs_info=[self.GRU_f.h0, self.GRU_b.h0])
        
        # The bidirectional state h t is constructed by concatenating the states of the forward and backward layers
        # 0-axis is time steps, 1-axis is batch size and 2-axis is hidden layer size
        context = T.concatenate([h_f_t, h_b_t[::-1]], axis=2)
        
#         projected_context =  T.dot(context, self.Wa_c) + self.ba # Not change the size
        [_, self.last_hidden_states, self.y, self.alphas], _ = theano.scan(fn=output_recurrence,
            # Ignore the 1st word in context, because there's no punctuation before that                                                               
            sequences=[context[1:]], 
            non_sequences=[self.Wa_c, self.ba, self.Wa_h, self.Wa_y, self.Wf_h, self.Wf_c, self.Wf_f, self.bf, self.Wy, self.by, context],
            outputs_info=[self.GRU.h0, None, None, None])
        # -------------------------------------------------- #
        # All the parameters
        self.params = [self.We,
                       self.Wy, self.by,
                       self.Wa_h, self.Wa_c, self.ba, self.Wa_y,
                       self.Wf_h, self.Wf_c, self.Wf_f, self.bf]
        self.params += self.GRU.params + self.GRU_f.params + self.GRU_b.params
        # print("Number of parameters is %d" % sum(np.prod(p.shape.eval()) for p in self.params))
        
        # Compute model complexity
        self.L1 = sum(abs(p).sum() for p in self.params)
        self.L2_sqr = sum((p**2).sum() for p in self.params)

    def cost(self, y):
        """
        Compute the maximium-likelihood, namely minimum log-likelihood of the coocurance of punctuations.
        """
        num_outputs = self.y.shape[0]*self.y.shape[1] # time steps * number of parallel sequences in batch
        output = self.y.reshape((num_outputs, self.y.shape[2]))
        return -T.sum(T.log(output[T.arange(num_outputs), y.flatten()]))

    def save(self, file_path, gsums=None, learning_rate=None, validation_ppl_history=None, best_validation_ppl=None, epoch=None, random_state=None):
        """
        Save the model.
        """
        state = {"type":                     self.__class__.__name__,
                 "n_hidden":                 self.n_hidden,
                 "x_vocabulary":             self.x_vocabulary,
                 "y_vocabulary":             self.y_vocabulary,
                 "params":                   [p.get_value(borrow=True) for p in self.params],
                 "gsums":                    [s.get_value(borrow=True) for s in gsums] if gsums else None,
                 "learning_rate":            learning_rate,
                 "validation_ppl_history":   validation_ppl_history,
                 "epoch":                    epoch,
                 "random_state":             random_state}

        with open(file_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
