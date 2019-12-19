# coding: utf-8
# -------------------- Package import -------------------- #
# Basic package
import numpy as np
# NLP Packages
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from pattern.en import comparative
# Tool packages
from Keywords.KeywordsTools import toUpper, decontract, delete_num
# Other packages
import string
from enum import Enum
from collections import Counter, defaultdict
from itertools import chain, groupby, product
import warnings; warnings.filterwarnings('ignore')

# -------------------- RakeKeywords class -------------------- #
class RakeKeywords():
    """
        Class for Rake("Rapid Automatic Keyword Extraction")
    """
    def __init__(self, text):
        # Original text
        super(RakeKeywords, self).__init__()
        self.org_text        = text
        # Text after processed
        self.proc_text       = ''
        self.token_lemma     = []
        # Result of RAKE
        self._pos_dict       = {}
        self.rake_bigram     = []
        self.rake_singleword = []
        # Extract and select keywords
        self._preprocess()
        self._keywords_extr()
        self._keywords_filter()
    
    def _preprocess(self):
        """
            Pre-process keywords (Tokenize and limmatize), and remain the (lower of upper) case of text.
        """
        text = self.org_text
        punctuation = string.punctuation
        stop = stopwords.words('english')
        
        # Lower case and decontract: I'm -> I am
        text = toUpper(text)
        text = decontract(text)

        # Tokenize and define stopwords
        tokens = word_tokenize(text)
        # Lemmatize words which are not stopwords
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = [wordnet_lemmatizer.lemmatize(i, pos='v') if i not in stop else i for i in tokens]
        tokens = [wordnet_lemmatizer.lemmatize(i, pos='n') if i not in stop else i for i in tokens]
        # Combine them back
        text_lemma = ' '.join(tokens)
        self.token_lemma = tokens
        self.proc_text   = text_lemma
    
    def _keywords_extr(self, max_length=2):
        """
            Extra keywords according to Rake algorithm. Rake uses stopwords for english from NLTK, and all puntuation characters
            Remain keywords of [noun], [ajdective, noun], [noun, noun] and delete others.
            
            Input:
                max_length:                Max length of keywords extracted by Rake.
        """
        text_lemma = self.proc_text
        # -------------------- Rake Extraction -------------------- #
        # Control the max words in a phrase, we assume the max length of keywords is 2
        rake_model = Rake(max_length=max_length)
        rake_model.extract_keywords_from_text(text_lemma)
        # To get keyword phrases ranked highest to lowest
        rake_keywords = rake_model.ranked_phrases

        # ------------------ Choose useful words ------------------ #
        # Generate a dictionary of POS tag of words; Usually POS use the surrounding context to predict
        # Use the most common POS to tag a word
        temp_dict = dict(Counter(pos_tag(word_tokenize(text_lemma))))
        for word_pos, values in temp_dict.items():
            if word_pos[0] not in self._pos_dict:
                self._pos_dict[word_pos[0]] = word_pos[1]
            else:
                aimed_key = [i for i in temp_dict.keys() if i[0]==word_pos[0]]
                max_index = np.argmax([temp_dict[key] for key in aimed_key])
                self._pos_dict[word_pos[0]] = aimed_key[max_index][1]
                
        # Remain keywords of [noun], [ajdective, noun], [noun, noun] and delete others
        rake_singleword, rake_bigram = [], []
        for rake_keyword in rake_keywords:
            keyword_list = word_tokenize(rake_keyword)
            # If it's a single keyword
            if len(keyword_list) == 1:
                keyword = keyword_list[0]
                try:
                    # Remain keywords of [noun]
                    if self._pos_dict[keyword] == 'NN':
                        rake_singleword.append(keyword.lower())
                except:
                    # print('%s\tcannot be matched.'%keyword)
                    pass
            # If it's a bigram keyword
            if len(keyword_list) == 2:
                keyword_1 = keyword_list[0]
                keyword_2 = keyword_list[1]
                try:
                    # Remain keywords of [noun], [ajdective, noun], [noun, noun]
                    if self._pos_dict[keyword_2] == 'NN' and \
                      (self._pos_dict[keyword_1] == 'NN' or self._pos_dict[keyword_1][0] == 'J'):
                        rake_bigram.append(' '.join([keyword_1, keyword_2]))
                except:
                    # print('(%s, %s)\tcannot be matched.'%(keyword_1, keyword_2))
                    pass
        self.rake_singleword, self.rake_bigram = rake_singleword, rake_bigram
        
    def _delete_name(self):
        """
            Delete name and place in keywords.
        """
        rake_singleword_temp = []
        # Delete name and place in lower case, which cannot be regarded.
        
        # Youtube subtitles will captialize some names, which can be recognized. We captialize keywords and match them with those capitalized names.
        for word in self.rake_singleword:
            cap_word = string.capwords(word)
            if cap_word in self._pos_dict:
                if self._pos_dict[cap_word] == 'NNP' or self._pos_dict[cap_word] == 'NNS':
                    continue
            rake_singleword_temp.append(word)
        self.rake_singleword = rake_singleword_temp
        
        # Similar for bigram keywords
        rake_bigram_temp = []
        for word in self.rake_bigram:
            cap_word = string.capwords(word).split(' ')
            if cap_word[0] in self._pos_dict:
                if self._pos_dict[cap_word[0]] == 'NNP' or self._pos_dict[cap_word[0]] == 'NNS':
                    continue
            if cap_word[1] in self._pos_dict:
                if self._pos_dict[cap_word[1]] == 'NNP' or self._pos_dict[cap_word[1]] == 'NNS':
                    continue
            rake_bigram_temp.append(word)
        self.rake_bigram = rake_bigram_temp
    
    def _delete_useless(self):
        """
            Delete bigram keywords start with meaningless word like 'big','new','large'. It uses a method judging whether the adjective has comparative form.
        """
        wnl = WordNetLemmatizer()
        rake_bigram_temp = []
        for i, bi_keyword in enumerate(self.rake_bigram):
            word1, word2 = bi_keyword.split(' ')
            word1_pos, word2_pos = self._pos_dict[word1], self._pos_dict[word2]

            # delete key words having only 1 letter
            if len(word1) > 1 and len(word2) > 1:
                if (word1_pos[0]) == 'J': 
                    comp_word1 = word_tokenize(comparative(word1))
                    # e.g. important -> more important
                    if len(comp_word1) > 1:
                        rake_bigram_temp.append(bi_keyword)
                    # e.g. sim -> simer, will be removed
                    if len(comp_word1) == 1 and \
                        wnl.lemmatize(comp_word1[0].lower(), 'a') != word1.lower():
                        rake_bigram_temp.append(bi_keyword)
        self.rake_bigram = rake_bigram_temp
        
        # Keep all single keyword but remove one letter word
        self.rake_singleword = [sing_keyword for sing_keyword in self.rake_singleword if len(sing_keyword)>1]
        
    
    def _keywords_filter(self):
        """
            Pipeline of filtering useless keywords.
        """
        self.rake_singleword = delete_num(self.rake_singleword)
        self.rake_bigram     = delete_num(self.rake_bigram)
        self._delete_name()
        self._delete_useless()


class Metric(Enum):
    """
    Different metrics that can be used for ranking.
    """
    DEGREE_TO_FREQUENCY_RATIO = 0  # Uses d(w)/f(w) as the metric
    WORD_DEGREE = 1  # Uses d(w) alone as the metric
    WORD_FREQUENCY = 2  # Uses f(w) alone as the metric

    
class Rake(object):
    """
    Rapid Automatic Keyword Extraction Algorithm.
    """
    def __init__(self,
                 stopwords=None,
                 punctuations=None,
                 language="english",
                 ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,
                 max_length=100000,
                 min_length=1):
        """
        Constructor.
        
        Input:
            stopwords:             List of Words to be ignored for keyword extraction
            punctuations:          Punctuations to be ignored for keyword extraction
            language:              Language to be used for stopwords
            max_length:            Maximum limit on the number of words in a phrase
            min_length:            Minimum limit on the number of words in a phrase
        """
        # By default use degree to frequency ratio as the metric.
        if isinstance(ranking_metric, Metric):
            self.metric = ranking_metric
        else:
            self.metric = Metric.DEGREE_TO_FREQUENCY_RATIO
        # If stopwords not provided we use language stopwords by default.
        self.stopwords = stopwords
        if self.stopwords is None:
            self.stopwords = nltk.corpus.stopwords.words(language)
        # If punctuations are not provided we ignore all punctuation symbols.
        self.punctuations = punctuations
        if self.punctuations is None:
            self.punctuations = string.punctuation
        # All things which act as sentence breaks during keyword extraction.
        self.to_ignore = set(chain(self.stopwords, self.punctuations))
        # Assign min or max length to the attributes
        self.min_length = min_length
        self.max_length = max_length
        # Stuff to be extracted from the provided text.
        self.frequency_dist = None
        self.degree = None
        self.rank_list = None
        self.ranked_phrases = None

    def extract_keywords_from_text(self, text):
        """
        Method to extract keywords from the text provided.

        Input:
            text:               Text to extract keywords from, provided as a string
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        phrase_list = self._generate_phrases(sentences)
        self._build_frequency_dist(phrase_list)
        self._build_word_co_occurance_graph(phrase_list)
        self._build_ranklist(phrase_list)

    def _build_frequency_dist(self, phrase_list):
        """
        Builds frequency distribution of the words in the given body of text.

        Input:
            phrase_list:        List of List of strings where each sublist is a collection of words which form a contender phrase
        """
        self.frequency_dist = Counter(chain.from_iterable(phrase_list))

    def _build_word_co_occurance_graph(self, phrase_list):
        """
        Builds the co-occurance graph of words in the given body of text to compute degree of each word.

        Input:
            phrase_list:        List of List of strings
        """
        co_occurance_graph = defaultdict(lambda: defaultdict(lambda: 0))
        for phrase in phrase_list:
            # For each phrase in the phrase list, count co-occurances of the word with other words in the phrase.
            # Note: Keep the co-occurances graph as is, to help facilitate its use in other creative ways if required later.
            for (word, coword) in product(phrase, phrase):
                co_occurance_graph[word][coword] += 1
        self.degree = defaultdict(lambda: 0)
        for key in co_occurance_graph:
            self.degree[key] = sum(co_occurance_graph[key].values())

    def _build_ranklist(self, phrase_list):
        """
        Method to rank each contender phrase using the formula:
            phrase_score = sum of scores of words in the phrase
            word_score = d(w)/f(w) where d is degree and f is frequency

        Input:
            phrase_list:        List of List of strings
        """
        self.rank_list = []
        for phrase in phrase_list:
            rank = 0.0
            for word in phrase:
                if self.metric == Metric.DEGREE_TO_FREQUENCY_RATIO:
                    rank += 1.0 * self.degree[word] / self.frequency_dist[word]
                elif self.metric == Metric.WORD_DEGREE:
                    rank += 1.0 * self.degree[word]
                else:
                    rank += 1.0 * self.frequency_dist[word]
            self.rank_list.append((rank, " ".join(phrase)))
        self.rank_list.sort(reverse=True)
        self.ranked_phrases = [ph[1] for ph in self.rank_list]

    def _generate_phrases(self, sentences):
        """
        Method to generate contender phrases given the sentences of the text document.

        Input:
            sentences:          List of strings where each string represents a sentence which forms the text
        Output: 
            Set of string tuples where each tuple is a collection of words forming a contender phrase
        """
        phrase_list = set()
        # Create contender phrases from sentences.
        for sentence in sentences:
            word_list = [word for word in word_tokenize(sentence)]
            phrase_list.update(self._get_phrase_list_from_words(word_list))
        return phrase_list

    def _get_phrase_list_from_words(self, word_list):
        """
        Method to create contender phrases from the list of words that form
        a sentence by dropping stopwords and punctuations and grouping the left
        words into phrases. Only phrases in the given length range (both limits
        inclusive) would be considered to build co-occurrence matrix. Ex:

        Sentence: Red apples, are good in flavour.
        List of words: ['red', 'apples', ",", 'are', 'good', 'in', 'flavour']
        List after dropping punctuations and stopwords.
        List of words: ['red', 'apples', *, *, good, *, 'flavour']
        List of phrases: [('red', 'apples'), ('good',), ('flavour',)]

        List of phrases with a correct length:
        For the range [1, 2]: [('red', 'apples'), ('good',), ('flavour',)]
        For the range [1, 1]: [('good',), ('flavour',)]
        For the range [2, 2]: [('red', 'apples')]

        Input:
            word_list: List of words which form a sentence when joined in the same order
        Output: 
            List of contender phrases that are formed after dropping stopwords and punctuations
        """
        groups = groupby(word_list, lambda x: x not in self.to_ignore)
        phrases = [tuple(group[1]) for group in groups if group[0]]
        return list(filter(lambda x: self.min_length <= len(x) <= self.max_length, phrases))