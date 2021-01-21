from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import json, re

def topic_name(model, count_vectorizer, topics, topic_number, number_words):
    if topic_number in topics:
        return topics[topic_number]
    words = count_vectorizer.get_feature_names()
    topic = model.components_[topic_number]
    topic = ' '.join([words[i] for i in topic.argsort()[:-number_words-1:-1]])
    topics[topic_number] = topic
    return topic

config = json.load(open('config.json', 'r'))

comments = pd.read_csv(config['input_doc'])

review_groups = [(1, comments)]
if config['groupby']:
    review_groups = comments.groupby(config['groupby'])

all_comments = pd.DataFrame()

for group, frame in review_groups:
    reviews = frame[config['input_column']].map(lambda x: re.sub('[,\.!?\*\$\'\"\(\)]', '', str(x)))
    reviews = frame[config['input_column']].map(lambda x: str(x).lower())
    
    count_vectorizer = TfidfVectorizer(min_df = config['min_df'], max_df = config['max_df'], stop_words = 'english')
    count_data = count_vectorizer.fit_transform(reviews)

    number_topics = config['topics_number']
    number_words = config['topics_n_words']

    lda = LDA(n_components = number_topics, doc_topic_prior = config['alpha'], topic_word_prior = config['beta'], n_jobs = -1, verbose = 1)
    transformed = lda.fit_transform(count_data)
    topics = {}

    frame[config['output_column']] = list(map(lambda x: topic_name(lda, count_vectorizer, topics, x.argmax(), number_words), tqdm(transformed)))

    df = pd.DataFrame(transformed, columns = [topics[i] for i in range(len(topics))])
    df.reset_index(drop = True, inplace = True)
    frame.reset_index(drop = True, inplace = True)
    frame = pd.concat([frame, df], axis = 1)
    all_comments = pd.concat([all_comments, frame])
    
all_comments.to_csv(config['output_doc'], index = False)