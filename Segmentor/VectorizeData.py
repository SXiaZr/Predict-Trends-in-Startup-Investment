# coding: utf-8
# -------------------- Package import -------------------- #
# System operation packages
import os
import sys
import shutil
import fnmatch
import operator

# -------------------- Parameters setting -------------------- #
# Set special letter notation
END = "</S>"
UNK = "<UNK>"

# Set punctuation list
SPACE = "_SPACE"
PUNCTUATION_VOCABULARY = ["_SPACE", ",COMMA", ".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK", ":COLON", ";SEMICOLON", "-DASH"]
EOS_TOKENS = [".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK"]

# Vocabulary setting
MIN_WORD_COUNT_IN_VOCAB  = 2
MAX_WORD_VOCABULARY_SIZE = 100000

# Length of the sequence
MAX_SEQUENCE_LEN = 50

# -------------------- Function defination -------------------- # 
def vectorizeData(input_path, output_path, save_vocabulary=True):
    """
    Read the textual data and embed it.
    
    Input:
        input_path:         Directory of input data
        output_path:        Directory of output data
        save_vocabulary:    Whether to save vocabulary dictionary
    """
    # Set output files' names
    train_output = os.path.join(output_path, "train")
    dev_output   = os.path.join(output_path, "dev")
    test_output  = os.path.join(output_path, "test")
    
    # {word: count} - Used for word dictionay creation
    word_counts = dict()
    
    # Get file names lists
    train_txt_files, dev_txt_files, test_txt_files = [], [], []
    for root, dirnames, filenames in os.walk(input_path):
        for filename in fnmatch.filter(filenames, '*.txt'):
            path = os.path.join(root, filename)
            if filename.endswith(".test.txt"):
                test_txt_files.append(path)
            elif filename.endswith(".dev.txt"):
                dev_txt_files.append(path)
            else:
            
    # Use training data to build vocabulary dict
                train_txt_files.append(path)
                with open(path, 'r', encoding='utf-8') as text:
                    for line in text:
                        for w in line.split(): # Split any whitespace by default including '\n'
                            if w in PUNCTUATION_VOCABULARY:
                                continue
                            word_counts[w] = word_counts.get(w, 0) + 1
    vocabulary = [wc[0] for wc in reversed(sorted(word_counts.items(), key=operator.itemgetter(1))) 
                            if wc[1] >= MIN_WORD_COUNT_IN_VOCAB and wc[0] != UNK][:MAX_WORD_VOCABULARY_SIZE]
    if END not in vocabulary:
        vocabulary.append(END)
    if UNK not in vocabulary:
        vocabulary.append(UNK)
        
    # Save the vocabulary
    if save_vocabulary:
        print("Saving vocabulary... (size: %d)" % len(vocabulary))
        WORD_VOCAB_FILE = os.path.join(output_path, "vocabulary")
        with open(WORD_VOCAB_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(vocabulary))
            
    # Build numerically-represented sequence
    innerVectorize(train_txt_files, train_output, vocabulary)
    innerVectorize(dev_txt_files, dev_output, vocabulary)
    innerVectorize(test_txt_files, test_output, vocabulary)


def innerVectorize(input_files, output_file, vocabulary):
    """
    Textual data will be converted to two sets of aligned subsequences (words and punctuations) of MAX_SEQUENCE_LEN.
    
    words        : posInVocab0, posInVocab1, posInVocab2, posInVocab3 ...   
    punctuations :              PuntInDict0, PuntInDict1, PuntInDict2 ...
    (actually punctuation sequence will be 1 element shorter)

    If a sentence is cut, it will be added to next subsequence (words before the cut belong to both sequences)
    So a subsequence contains information both before and after the end-of-sentence punction.
    
    Input:
        input_files:        List of input data files
        output_file:        Output data file
        vocabulary:    Word vocabulary dictionary
    """
    data, current_words, current_punctuations = [], [], []
    num_total, num_unks, last_eos_idx = 0, 0, 0

    # if a sentence does not fit into subsequence, then we need to skip tokens until we find a new sentence
    skip_until_eos = False 
    last_token_was_punctuation = True # Skip first token if it's punctuation
    
    # Get punctuation vocabulary: {punctuation: punctNum} 
    word_vocabulary = dict((x.strip(), i) for (i, x) in enumerate(vocabulary))
    punctuation_vocabulary = dict((x.strip(), i) for (i, x) in enumerate(PUNCTUATION_VOCABULARY))
    
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as text:
            for line in text:
                for token in line.split(): # Split by white space
                    # Used for skipping until next sentence
                    if skip_until_eos:
                        if token in EOS_TOKENS:
                            skip_until_eos = False
                        continue
                    
                    # If the token is a punctuation
                    elif token in punctuation_vocabulary:
                        # If encounter consecutive punctuations, only keep the first punctuation
                        if last_token_was_punctuation:
                            continue
                        # Mark the end of sentence (the start index of next sentence)
                        if token in EOS_TOKENS:
                            last_eos_idx = len(current_punctuations) 
                        # Add numerical presentation of punctuation
                        punctuation = punctuation_vocabulary[token]
                        current_punctuations.append(punctuation)
                        last_token_was_punctuation = True
                        
                    # If the token is not a punctuation (word)
                    else:
                        # Add the punctuation of last word
                        if not last_token_was_punctuation:
                            # Since last_token_was_punctuation is initialized to 'False', no first punctuation 
                            current_punctuations.append(punctuation_vocabulary[SPACE])
                        # Add numerical presentation of word
                        word = word_vocabulary.get(token, word_vocabulary[UNK])
                        current_words.append(word)
                        last_token_was_punctuation = False
                        # Count number
                        num_total += 1
                        num_unks += int(word == word_vocabulary[UNK])
                        
                    # If the sequence fetch its maximum length
                    if len(current_words) == MAX_SEQUENCE_LEN:
                        # If sentence can not fit into subsequence - skip it
                        if last_eos_idx == 0: 
                            skip_until_eos = True
                            current_words = []
                            current_punctuations = []
                            # next sequence starts with a new sentence, so is preceded by eos which is punctuation
                            last_token_was_punctuation = True 
                        else:
                            # Remove last word and append 'END' notation
                            subsequence = [current_words[:-1] + [word_vocabulary[END]],
                                           current_punctuations]
                            data.append(subsequence)
                            # Carry unfinished sentence to next subsequence
                            current_words = current_words[last_eos_idx+1:]
                            current_punctuations = current_punctuations[last_eos_idx+1:]
                        last_eos_idx = 0
                        
    # Save the numerically-represented data
    print("%.2f%% UNK-s in %s" % (num_unks / num_total * 100, output_file))
    with open(output_file, 'w') as f:
        for seq in data:
            f.write("%s\n" % repr(seq))

# -------------------- Main function -------------------- #
if __name__ == "__main__":
    if len(sys.argv) > 2:
        input_path, output_path = sys.argv[1], sys.argv[2] 
    else:
        sys.exit("The path to directory is missing!")

    # Create 'data' folder, replace the old if it exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    # Vectorize the data and label
    vectorizeData(input_path, output_path)
    print('Task has been done!')