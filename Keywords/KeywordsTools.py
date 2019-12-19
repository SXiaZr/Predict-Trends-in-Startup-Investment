# coding: utf-8
# -------------------- Package import -------------------- #
import re
import Keywords.w2n
import string

# -------------------- Tool functions -------------------- #
def toUpper(text, threshold=0.6):
    """
        Turn the first word of sentences text to uppercase.
        Input:
            text:                       Text to be judged
        Output:
            text:                       Text has been processed.
    """ 
    count, return_text = 0, ''
    eos_punctuation    = ['.', '!', '?'] 
    eos_flag           = True
    regx               = re.compile('[a-zA-Z]')

    # Remove punctuatation and white spaces
    text_upper = ''.join([i for i in text if i not in string.punctuation]).replace(' ', '')
    for i in text_upper:
        if i.isupper():
            count += 1
            
    # If the subtitle are all in upper case, turn first letter to upper case
    for letter in text:
        # If it's a punctuation
        if len(regx.findall(letter)) == 0: 
            # Only keep one punctuation if there are two eos
            if eos_flag and letter!=' ':
                letter = ''
            if letter in eos_punctuation:
                eos_flag = True
        # If it's a letter
        else:  
            if eos_flag:
                letter = letter.upper()
                eos_flag = False
            else:
                if count > len(text_upper) * threshold:
                    letter = letter.lower()
        return_text += letter    
    return return_text

def decontract(text):
    """
        Decontract abbreviation. Adapted from: https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python

        Input:
            text:                       Text to be decontracted
        Output:
            text:                       Text has been decontracted
    """
    text = re.sub(r"won't" , "will not", text)
    text = re.sub(r"can\'t", "can not" , text)
    text = re.sub(r"n\'t"  , " not"    , text)
    text = re.sub(r"\'re"  , " are"    , text)
    text = re.sub(r"\'s"   , " is"     , text)
    text = re.sub(r"\'d"   , " would"  , text)
    text = re.sub(r"\'ll"  , " will"   , text)
    text = re.sub(r"\'t"   , " not"    , text)
    text = re.sub(r"\'ve"  , " have"   , text)
    text = re.sub(r"\'m"   , " am"     , text)
    return text

def delete_num(word_list):
    """
        Delete keywords containing number.
        
        Input:
            word_list:                 A list of words needed to clean number in it
        Output:
            cleand_list:               A list of words after cleaning
    """
    middle_list, cleand_list = [], []
    hasNumbers  = lambda inputString: any(char.isdigit() for char in inputString)

    # Delete keywords containing digital number
    for word in word_list:
        numFlag = False
        word_split = word.split(' ')
        for i in word_split:
            if hasNumbers(i):
                numFlag = True
                break
        if not numFlag:
            middle_list.append(word)
                
    # Delete keywords containing number word
    for word in middle_list:
        numFlag = False
        word_split = word.split(' ')
        for i in word_split:
            if Keywords.w2n.word_to_num(i):
                numFlag = True
                break
        if not numFlag:
            cleand_list.append(word)
    return cleand_list