# coding: utf-8
# -------------------- Package import -------------------- #
# Numpy and pandas
import numpy as np
import pandas as pd 
# NLP package
import spacy
from nltk.corpus import stopwords
# Use word2vec to embed word
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Other packages
import re
import os
import pickle
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from bs4 import BeautifulSoup
# Plot pacakges
import matplotlib.pyplot as plt
# Self-defined package
from Segmentor import buildModel, punctuate
from Keywords import RakeKeywords
from InvestorDataFrame import InvestorDataFrame

# -------------------- Load model and set path -------------------- #
DATA_PATH = './data/feature/'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
# Load GRU model
net, predict = buildModel('./model/Europarl-EN.pcl')
# Load word-embeding model
print('Loading word2vector model...')
modelpath = './model/GoogleNews-vectors-negative300.bin' # Load google model
google_model = KeyedVectors.load_word2vec_format(modelpath, binary=True)
print('Done!')

# -------------------- Feature extraction -------------------- #
def get_investors_list(ori=False):
    """
    Get a list of https://seedproof.com/investors/
    """
    file = './setting/investorsList.pickle'
    if os.path.exists(file) and not ori:
        with open(file, 'rb') as f:
            investors = pickle.load(f)

    else:
        with open("./setting/Seedproof - List of investors.html", "r", encoding='utf-8') as f:
            investor_html= f.read()
        soup = BeautifulSoup(investor_html, 'html.parser')
        lis = soup.findAll('div', {'class': 'colPrimary grid-row-cell cell-left'})
        investors = [re.findall("[a-zA-z]*\s[a-zA-z]*", i.text)[0] for i in lis]
        # Save to local
        if not ori:
            with open(file, 'wb') as f:
                pickle.dump(investors, f, protocol=pickle.HIGHEST_PROTOCOL)
    return investors

def punctuated_rake(IDFrame):
    """
    Extract rake keywords.
    """
    file = DATA_PATH + IDFrame._investor.replace(' ', '_') + '_youtube_feature.csv'
    youtube_fea_df    = pd.DataFrame(columns=['title', 'punctuatedText', 'tokenLemma', 'rakeKeywords'])
    youtube_data_list = []

    if os.path.exists(file):
        youtube_fea_df = pd.read_csv(file, index_col=0)
    else:
        for _, irow in IDFrame._youtube_df.iterrows():
            punctuateText = punctuate(irow['caption_text'], net, predict)
            res      = RakeKeywords(punctuateText)
            keywords = res.rake_bigram + res.rake_singleword
            youtube_data_list.append({'title'         : irow['title'],
                                      'punctuatedText': punctuateText,
                                      'tokenLemma'    : str(res.token_lemma),
                                      "rakeKeywords"  : str(keywords)})
        # Save to local
        temp = pd.DataFrame.from_dict(youtube_data_list)
        youtube_fea_df = pd.concat([youtube_fea_df, temp], join='inner')
        youtube_fea_df.to_csv(file)
        print('{}: Punctuation and rake has been done.'.format(IDFrame._investor))
    return youtube_fea_df

def get_sentiment(youtube_fea_df, threshold=0.05):
    """
    Calculate normalized sentiment scores of sentences.
        1. Count the number of positive sentences and negative sentences
        2. Accumulate positive/negative sentences sentiment scores 

    Input:
        threshold:                     Threshold to decide whether a sentence is positive or negative
    """
    nlp = spacy.load('en')
    analyzer = SentimentIntensityAnalyzer()
    posSentNum, negSentNum, posSentScore, negSentScore = [], [], [], []
    for _, irow in youtube_fea_df.iterrows():
        doc   = nlp(irow.punctuatedText)
        
        sents = np.array([analyzer.polarity_scores(sent.text)['compound'] for sent in doc.sents])
        posSentNum.append(sum(sents>= 0.05) / len(sents))
        negSentNum.append(sum(sents<=-0.05) / len(sents))
        posSentScore.append(sum(sents[sents>= 0.05]) / sum(sents>= 0.05))
        negSentScore.append(sum(sents[sents<=-0.05]) / sum(sents<=-0.05))
        
    youtube_fea_df['posSentNum'] = posSentNum
    youtube_fea_df['negSentNum'] = negSentNum
    youtube_fea_df['posSentScore'] = posSentScore
    youtube_fea_df['negSentScore'] = negSentScore
    return youtube_fea_df

def merge_dataframe(IDFrame, youtube_fea_df):
    """
    Merge youtube data, crunchbase data, and published date data on title.
        1. First, merge automatically-extracted published date data and youtube data on title.
        2. Then cross product dataframes of youtube data and crunchbase data.     
    """
    investor_df = youtube_fea_df.merge(IDFrame._real_time_auto, on=['title'])
    investor_df = IDFrame._crunchbase_df.assign(key=1).merge(investor_df.assign(key=1), on='key').drop('key', 1)
    return investor_df

def get_time_interval(investor_df):
    """
    Filter items according to time:
        1. 'Announced_Date' >= 'InterviewTime'
        2. 'Announced_Date' < 2099: Since I sign 2199 to the video, if no data can be found
    Compute the time interval of 'Announced_Date' and 'InterviewTime'.
    """
    investor_df = investor_df[investor_df['Announced_Date']   >= investor_df['InterviewTimeAuto']]
    investor_df = investor_df[investor_df['InterviewTimeAuto']<  pd.to_datetime('2099-1-1')]
    # Compute time interval in month
    investor_df['Time_Interval_Month'] = investor_df.apply(lambda x: (x['Announced_Date'].year  - x['InterviewTimeAuto'].year) * 12\
                                                                   + (x['Announced_Date'].month - x['InterviewTimeAuto'].month), axis=1).values.tolist()
    return investor_df

def _compare_similarity_keyword(caption_text, keywords, tokenLemma, domain, thres_ratio = 0.5):
    """
    Function to compute the similiarity between the caption_text and domain
        
    Input:
        caption_text:                Punctuated caption text
        keywords:                    List of rake keywords
        tokenLemma:                  List of Lemmatized tokens
        domain:                      The domain of the company
        thres_ratio:                 Threshold use to count keyword number
    Output:
        Normalized similarity and keyword numbers.
    """
    stop = stopwords.words('english')
    keywords_list = []
    # Replace '-' in keyword and domain and split 
    for keyword in keywords:
        keywords_list.extend(keyword.lower().replace('-', ' ').split(' '))
    domain_list = domain.lower().replace('-', ' ').split(' ')
    
    # Accumulate similarity for normalization
    accumulated_sim = 0
    sim_dict = {}
    for keyword in keywords_list:
        # Calculate similarity of each combination of keyword and domain
        if keyword not in stop: 
            sim_sum = 0
            for i in domain_list:
                try:
                    # Some of the similarity(keyword, i) are minus but I still keep it to show the uncorrelated
                    sim = google_model.similarity(i, keyword)
                    # google_model.similarity is related to upper or lower case 
                    accumulated_sim += sim
                    sim_sum += sim
                except:
                    continue
            if keyword not in sim_dict:
                sim_dict[keyword] = sim_sum
                
    # Compute frequency of keywords at the same time
    if len(sim_dict)==0:
        return None, None
    max_sim = max(sim_dict.items(), key=lambda x:x[1])[1]
    # If one word whose similarity with domain larger than a half of the maximum similarity, count it
    keywords_thres = [i for i in sim_dict.keys() if sim_dict[i] > max_sim * thres_ratio]
    keywords_freq = 0
    for i in tokenLemma:
        if i in keywords_thres:
            keywords_freq += 1
    # Normalize the accumulated similarity and keyword number by dividing total number of context
    return accumulated_sim / len(keywords), keywords_freq / len(tokenLemma)

def get_similarity_keyword(investor_df):
    """
    Compare keywords of the subtitle with the domain in domian list, compute and sum the similiarities.
        1. Divide the summed similiarity by the number of keywords to get a normalized similiarity
        2. Regard it as the similiairy between the interview and that domain
    """
    investor_df['rakeKeywords'] = investor_df['rakeKeywords'].map(lambda x: eval(x))
    investor_df['tokenLemma']   = investor_df['tokenLemma'].map(lambda x: eval(x))

    temp = investor_df.apply(lambda x: _compare_similarity_keyword(x['punctuatedText'], x['rakeKeywords'], x['tokenLemma'], x['Domain']), axis=1)
    investor_df['Similarity'] = [i[0] for i in temp.values]
    investor_df['rakeKeywordsFreq'] = [i[1] for i in temp.values]
    return investor_df

def get_network_centrality(domain_df, title, adj_method='VectorDistance', visualize='off'):
    """
    Construct network and compute centrality of network.

    Input:
        domain_df:            Each row is an one-hot embeding of the investment domains of a investor
        title:                Name of the network         
        adj_methed:
            'VectorDistance': Use cosine distance and Gaussian kernel to normalize the distance to weight.
            'WeightCount'   : Every same component of vector increase weights by 1.
    """
    if adj_method == 'VectorDistance':
        # Compute cosine similiarity of nodes and distance
        distances = pdist(domain_df.values, metric='cosine')
        # Distance to weights
        kernel_width = distances.mean()
        weights = np.exp(-distances**2 / kernel_width**2)
        # Turn the list of weights into a matrix.
        adjacency = squareform(weights)
        # Filter to sparsify the network
        adjacency[adjacency < 0.3] = 0
    if adj_method == 'WeightCount':
        weights = []
        n_nodes = domain_df.shape[0]
        adjacency = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes-1):
            for j in range(i+1, n_nodes):
                val = np.sum(np.multiply(domain_df.iloc[i].values, domain_df.iloc[j].values))
                weights.append(val)
                adjacency[i][j] = val
                adjacency[j][i] = val
        weights = np.array(weights)
        adjacency = adjacency.astype('int')
        
    # Construct the network
    graph = nx.from_numpy_array(adjacency)
    
    # Visualize
    if visualize == 'on':
        # Only visualize the largest Find largest component
        comp = list(nx.connected_components(graph))
        largest_comp = list(max(comp, key=len))
        largest_comp = graph.subgraph(largest_comp)
        # Plot the network
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].hist(weights, color='b')
        axes[0].set_xlabel('Bins')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of weights')
        # nx.draw_spectral(graph)
        nx.draw_spring(largest_comp, 
                       with_labels=True, 
                       ax = axes[1],
                       node_color=np.array(list(dict(largest_comp.degree(weight='weight')).values())), 
                       cmap='Reds',
                       node_shape='.',
                       width=0.1)
        axes[1].set_title(title + ' Network: Largest component')
        plt.show()
    
    # Calculate centrality features
    # Degree centrality
    degree_centrality = dict(graph.degree(weight='weight'))
    max_degree = max(degree_centrality.values())
    for i in degree_centrality.keys(): # Normalize
        degree_centrality[i] = degree_centrality[i] / max_degree
    # Betweenness centrality   
    betweenness_centrality = nx.betweenness_centrality(graph, weight='weight')
    # Closeness centrality
    g_distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in graph.edges(data='weight')}
    nx.set_edge_attributes(graph, g_distance_dict, 'distance')
    closeness_centrality = nx.closeness_centrality(graph, distance='distance')
    
    # Form the dataframe
    inv_nx_df = pd.DataFrame([degree_centrality, dict(betweenness_centrality), dict(closeness_centrality)]).T
    inv_nx_df.index = domain_df.index
    inv_nx_df.columns = [title+'_degree_centrality', title+'_betweenness_centrality', title+'_closeness_centrality']
    return inv_nx_df