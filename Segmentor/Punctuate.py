# coding: utf-8
# -------------------- Package import -------------------- #
# Other package
import warnings
warnings.filterwarnings("ignore")

# System operation packages
import sys

# Theano and numpy
import theano
import theano.tensor as T
import numpy as np

# Self-defined model
from Segmentor import Models
from Segmentor import VectorizeData

# NLP package
import re
from nltk.tokenize import word_tokenize

def punctuateInner(net, predict, word_vocabulary, punctuation_vocabulary, text, f_out=False):
    """
    Punctuate text according to the prediction result and output a punctuated text.
    
    Input:
        net:                        Theano network model
        predict:                    Theano-defined function
        word_vocabulary:            Word vocabulary dictionary
        punctuation_vocabulary:     Punctuation vocabulary dictionary
        text:                       Text needed to be punctuated
        f_out:                      Output file name
    """
    if len(text) == 0:
        sys.exit("Input text from stdin missing.")
        
    # Parameter to control the punctuation process
    i = 0
    output_text = ''

    # Create reverse dictionary: {0: '_space', ...}
    reverse_punctuation_vocabulary = {v:k for k,v in punctuation_vocabulary.items()}
    reverse_word_vocabulary = {v:k for k,v in word_vocabulary.items()}
    human_readable_punctuation_vocabulary = [p[0] for p in punctuation_vocabulary if p != VectorizeData.SPACE]
    
    # Use to judge whether it's a number
    numbers = re.compile(r'\d')
    is_number = lambda x: len(numbers.sub('', x)) / len(x) < 0.6
    
    # Used to remove all the original punctuation, but remain the " ' "
    tokenizer = word_tokenize
    untokenizer = lambda text: text.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
    text = [w for w in untokenizer(' '.join(tokenizer(text))).split() 
               if w not in punctuation_vocabulary and w not in human_readable_punctuation_vocabulary]
    # End the text with END notation
    if text[-1] != VectorizeData.END:
        text += [VectorizeData.END]

    # Start prediction sequence by sequence
    while True:
        subsequence = text[i:i+VectorizeData.MAX_SEQUENCE_LEN]
        if len(subsequence) == 0:
            break
        # Embed subsequence to a vector
        converted_subsequence = [word_vocabulary.get("<NUM>" if is_number(w) \
                                                             else w.lower(), word_vocabulary[VectorizeData.UNK]) for w in subsequence]
        y = predict(np.array([converted_subsequence], dtype=np.int32).T) # Make sequence as column
        # Write the first letter of the sequence
        output_text += subsequence[0].capitalize()
        last_eos_idx = 0
        punctuations = []
        # Embed prediction result back to punctuation
        for y_t in y:
            p_i = np.argmax(y_t.flatten())
            # Get the punctuation with the highest probability
            punctuation = reverse_punctuation_vocabulary[p_i]
            punctuations.append(punctuation)
            if punctuation in VectorizeData.EOS_TOKENS:
                last_eos_idx = len(punctuations) 
        if last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1

        # Write predicted punctuation before the word 
        for j in range(step):
            if punctuations[j] != VectorizeData.SPACE:
                if punctuations[j] =="-DASH" :
                    output_text += punctuations[j][:1]
                else:
                    output_text += punctuations[j][:1] + " "
            else:
                output_text += " "
            if j < step - 1:
                if punctuations[j] in VectorizeData.EOS_TOKENS:
                    output_text += subsequence[1+j].capitalize()
                else:
                    output_text += subsequence[1+j]
                    
        # Break at the end of text
        if subsequence[-1] == VectorizeData.END:
            break
        i += step

    if f_out == False:
        pass
    else:
        f_out.write(output_text)
    return output_text

def buildModel(model_file):
    """
    Load GRU model.

    Input:
        model_file:                 Name of the pre-trained model model_file
    Output:
        Loaded Pre-trained model and prediction function
    """
    x = T.imatrix('x')

    print("Loading model parameters...")
    net, _ = Models.load(model_file, 1, x)

    print("Building model...")
    predict = theano.function(inputs=[x], outputs=net.y)
    return net, predict

def punctuate(text, net, predict, output_file=False):
    """
    Punctuate text.

    Input:
        text:                       Text needed to be punctuated
        net:                        Loaded Pre-trained model 
        predict:                    Theano prediction function
    """
    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary
    # Whether to save the output to the local
    if output_file == False:
        output_text = punctuateInner(net, predict, word_vocabulary, punctuation_vocabulary, text)
    else:
        output_text = punctuateInner(net, predict, word_vocabulary, punctuation_vocabulary, text, output_file)
    return output_text

# -------------------- Main function -------------------- #
if __name__ == '__main__':
    if len(sys.argv) > 3:
        input_file, output_file, model_file  = sys.argv[1], sys.argv[2], sys.argv[3] 
    else:
        sys.exit("Arguments are missing!")
    # Load the input file
    with open(input_file,  'r', encoding='utf-8') as f_in:
        text = f_in.read()
    # Load the model
    net, predict = buildModel(model_file)
    # Whether to save the output to the local
    if output_file == False:
        output_text  = punctuate(text, net, predict, output_file=False)
    else:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            punctuate(text, net, predict, model_file, output_file=f_out)
            f_out.flush()
    