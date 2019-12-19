# coding: utf-8
# -------------------- Package import -------------------- #
# Other package
import warnings
warnings.filterwarnings("ignore")
from collections import OrderedDict

# System operation packages
import os
import sys
from time import time

# Theano and numpy
import theano
import theano.tensor as T
import numpy as np

# Self-defined model
from Segmentor import Models
from Segmentor import VectorizeData

# -------------------- Parameters setting -------------------- #
# Model hyperparameter
L2_REG = 0.0
CLIPPING_THRESHOLD = 2.0

# Training setting
MAX_EPOCHS = 50
PATIENCE_EPOCHS = 1
MINIBATCH_SIZE = 128

# -------------------- Function defination -------------------- # 
def get_minibatch(file_name, batch_size, shuffle=False):
    """
    Build a generator to produce data in mini-batch.
    
    Input:
        file_name:          File name
        batch_size:         Batch size
        shuffle:            Whether to suffle the data or not
    """
    # Load vectorized data
    dataset = []
    with open(file_name, 'r') as file:
        for line in file:
            dataset.append(eval(line))
    # Whether shuffle
    if shuffle:
        np.random.shuffle(dataset)
        
    # Generator mini-batch data
    X_batch = []
    Y_batch = []
    for subsequence in dataset:
        X_batch.append(subsequence[0])
        Y_batch.append(subsequence[1])
        if len(X_batch) == batch_size:
            # Transpose, because the model assumes the first axis is time
            X = np.array(X_batch, dtype=np.int32).T
            Y = np.array(Y_batch, dtype=np.int32).T
            yield X, Y
            X_batch = []
            Y_batch = []

# -------------------- Main function -------------------- # 
if __name__ == "__main__":
    if len(sys.argv) > 4:
        data_path, model_name, num_hidden, learning_rate = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    else:
        sys.exit("Arguments missing!")

    model_file_name = "Model_%s_h%d_lr%s.pcl" % (model_name, num_hidden, learning_rate)
    # --------------- Load vocabulary dict --------------- #
    # Load vocabulary and create vocabulary dictionary
    WORD_VOCAB_FILE = os.path.join(data_path, "vocabulary")
    with open(WORD_VOCAB_FILE, 'r', encoding='utf-8') as f:
        word_vocabulary = dict((x.strip(), i) for (i, x) in enumerate(f.readlines()))
    # Load punctuation dicitonary
    punctuation_vocabulary = dict((x.strip(), i) for (i, x) in enumerate(VectorizeData.PUNCTUATION_VOCABULARY))
 
    # --------------- Set the model --------------- #
    # ---------- Load model ---------- #
    # Creates matrix variables with name; Tensor has no initial value, just a symbolic
    x, y, lr = T.imatrix('x'), T.imatrix('y'), T.scalar('lr')
    # If the model exist, continue training from it
    if os.path.isfile(model_file_name):
        print("Loading previous model state")
        net, state = Models.load(model_file_name, MINIBATCH_SIZE, x)
        gsums, learning_rate, validation_ppl_history, starting_epoch, rng = state
        best_ppl = min(validation_ppl_history)

    else:
    # ---------- Initalize model ---------- #
        # Initalize training setting
        starting_epoch = 0
        # Perplexity: a measurement of how well a probability model predicts a sample
        best_ppl = np.inf 
        validation_ppl_history = []
        
        # Initalize a random seed
        rng = np.random
        rng.seed(1)
        # Build model
        print("Building model...")
        net = Models.GRU(rng=rng,
                         x=x,
                         minibatch_size=MINIBATCH_SIZE,
                         n_hidden=num_hidden,
                         x_vocabulary=word_vocabulary,
                         y_vocabulary=punctuation_vocabulary)
        # Shared variable can be stored in GPU; Shared variable has its initial value
        # gsums: the state of 'Adagrad' optimizer; list of gradient sum: [theano.shared0, theano.shared1, ...] 
        gsums = [theano.shared(np.zeros_like(param.get_value(borrow=True))) for param in net.params]
    # Add model cost with L2 penality if it's needed
    cost = net.cost(y) + L2_REG * net.L2_sqr
    
    # --------------- Adagrad optimizor --------------- #
    # Used to store net and optimizor paramaters updated in order: {old value: new value} 
    updates = OrderedDict()
    # Auto grad
    gparams = T.grad(cost, net.params)
    # Compute norm of gradients
    norm = T.sqrt(T.sum([T.sum(gparam ** 2) for gparam in gparams]))
    
    for gparam, param, gsum in zip(gparams, net.params, gsums):
        gparam = T.switch(T.ge(norm, CLIPPING_THRESHOLD),
                          gparam / norm * CLIPPING_THRESHOLD,
                          gparam) # Clipping of gradients
        updates[gsum] = gsum + (gparam ** 2)
        # Adagrad optimizor take one step
        updates[param] = param - lr * (gparam / (T.sqrt(updates[gsum] + 1e-6)))
    
    # --------------- Model training --------------- #
    # Define training and validation function
    train_model = theano.function(inputs=[x, y, lr], outputs=cost, updates=updates)
    validate_model = theano.function(inputs=[x, y], outputs=net.cost(y))
    
    # Load vectorized data
    train_file = os.path.join(data_path, "train")
    dev_file   = os.path.join(data_path, "dev")
    
    print("Training...")
    for epoch in range(starting_epoch, MAX_EPOCHS):
        # Count the time
        t0 = time()
        # Count the total loss and 
        total_neg_log_likelihood = 0
        total_num_output_samples = 0
        # Count the number of mini-batch
        iteration = 0 

        for X, Y in get_minibatch(train_file, MINIBATCH_SIZE, shuffle=True):
            total_neg_log_likelihood += train_model(X, Y, learning_rate)
            # Count the total number of data (punctuation)
            total_num_output_samples += np.prod(Y.shape) 
            iteration += 1
            if iteration % 100 == 0:
                sys.stdout.write("PPL: %.4f; Speed: %.2f sps\n" % (np.exp(total_neg_log_likelihood / total_num_output_samples), total_num_output_samples / max(time() - t0, 1e-100)))
                sys.stdout.flush()
        print("Total number of training labels: %d" % total_num_output_samples)
        
        # --------------- Validation every epoch --------------- #
        total_neg_log_likelihood = 0
        total_num_output_samples = 0
        for X, Y in get_minibatch(dev_file, MINIBATCH_SIZE, shuffle=False):
            total_neg_log_likelihood += validate_model(X, Y)
            total_num_output_samples += np.prod(Y.shape)
        print("Total number of validation labels: %d" % total_num_output_samples)
        
        ppl = np.exp(total_neg_log_likelihood / total_num_output_samples)
        validation_ppl_history.append(ppl)
        print("Validation perplexity is %s" % np.round(ppl, 4))
        
        # --------------- Save the model --------------- #
        if ppl <= best_ppl:
            best_ppl = ppl
            net.save(model_file_name, gsums=gsums, learning_rate=learning_rate, validation_ppl_history=validation_ppl_history, best_validation_ppl=best_ppl, epoch=epoch, random_state=rng.get_state())
        elif best_ppl not in validation_ppl_history[-PATIENCE_EPOCHS:]:
            # Prevent over-fitting: If validation ppl start to grow, stop training
            print("Finished!")
            print("Best validation perplexity was %s" % best_ppl)
            break