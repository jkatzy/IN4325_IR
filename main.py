import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
from collections import defaultdict
import os
import json
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import treebank
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
from nltk.stem import SnowballStemmer

from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
#from skmultilearn.adapt import MLkNN
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report


class conversation:
    def __init__(self, i, title, category, dialog_time, frequency, utterances):
        self.i = i;
        self.title = title;
        self.category = category;
        self.dialog_time = dialog_time;
        self.frequency = frequency;
        self.utterances = utterances;
        self.utterances_count = len(utterances)

class utterance:
    def __init__(self, i, utterance_pos, actor_type, user_id, utterance, vote, utterance_time, affiliation, is_answer, tags):
        self.i = i;
        self.utterance_pos = utterance_pos;
        self.actor_type = actor_type;
        self.user_id = user_id;
        self.utterance = utterance.lower();
        self.vote = vote;
        self.utterance_time = utterance_time;
        self.affiliation = affiliation;
        self.is_answer = is_answer;
        self.tags = [t for t in tags.strip().split(" ")];


def read_data(location = "./MSDialog/MSDialog-Intent.json"):
    conversations = [];
    with open(location) as json_file:
        data = json.load(json_file)
        for k in data.fromkeys(data):
            current = data[str(k)]
            utternaces = []
            utterances_data = current['utterances']
            for u in utterances_data:
                utternaces.append(utterance(i = u['id'], utterance_pos=u['utterance_pos'], actor_type=u['actor_type'], user_id=u['user_id'], utterance=u['utterance'], vote=u['vote'], utterance_time=u['utterance_time'], affiliation=u['affiliation'], is_answer=u['is_answer'], tags=u['tags']))
            conversations.append(conversation(i = k, title=current['title'], category=current['category'], dialog_time=current['dialog_time'], frequency=current['frequency'], utterances=utternaces))
    return conversations;

# Content embedding methods
def q_mark(utterance):
    u = utterance.utterance
    return '?' in u

def duplicate(utterance):
    u = utterance.utterance.lower()
    return 'same' in u or 'similar' in u

def w5h1(utterance):
    u = utterance.utterance.lower()
    w5 = ['what', 'where', 'when', 'why', 'who', 'how']
    return [w in u for w in w5]


# Structural embedding methods
def norm_abs_pos(conversation, utterance):
    return utterance.utterance_pos/conversation.utterances_count


def length(utterance):
    return len([w for w in utterance.utterance.split() if w not in stopwords.words('english')])


def len_uni(utterance):
    return len(set(w for w in utterance.utterance.split() if w not in stopwords.words('english')))


def len_stem(stemmer, utterance):
    return len(set(stemmer.stem(w) for w in utterance.utterance.split() if w not in stopwords.words('english')))


def is_starter(utterance):
    return utterance.actor_type == 'User'


# Sentiment embedding methods
def thank(utterance):
    return 'thank' in utterance.utterance


def e_mark(utterance):
    return '!' in utterance.utterance


def feedback(utterance):
    return 'did not' in utterance.utterance or 'does not' in utterance.utterance


def opinion_lex(tokenizer, utterance):
    pos = 0
    neg = 0
    for word in tokenizer.tokenize(utterance.utterance):
        pos += word in opinion_lexicon.positive()
        neg += word in opinion_lexicon.negative()

    return pos, neg


def sentiment_score(analyzer, utterance):
    scores = analyzer.polarity_scores(text=utterance.utterance)
    return scores['neg'], scores['neu'], scores['pos']


def remove_junk_labels(labels):
    if len(labels) > 1 and 'GG' in labels:
        labels.remove('GG')
    if len(labels) > 1 and 'O' in labels:
        labels.remove('O')
    if len(labels) > 1 and 'JK' in labels:
        labels.remove('JK')
    #return labels


def preprocess_labels(y):
    u, count = np.unique(y, return_counts=True)
    count_sort_ind = np.argsort(-count)[:32]
    label_dict = defaultdict(int, dict.fromkeys(map(str, list(u[count_sort_ind])), 1))

    for i in range(len(y)):
        if not label_dict[str(y[i])]:
            y[i] = [random.choice(y[i])]
    return y


def combine_structural(conversations):
    x = []
    y = []
    conv_count = len(conversations)
    stemmer = SnowballStemmer('english')

    for i, c in enumerate(conversations):
        for u in c.utterances:
            remove_junk_labels(u.tags)
            y.append(u.tags)
            x.append([u.utterance_pos, norm_abs_pos(c, u), length(u), len_uni(u), len_stem(stemmer, u), is_starter(u)])
        print('\r>>>> {}/{} done...'.format((i + 1), conv_count), end='')
    return np.asarray(x), np.asarray(y)


def combine_sentimental(conversations):
    x = []
    y = []
    conv_count = len(conversations)
    analyzer = SentimentIntensityAnalyzer()
    tokenizer = treebank.TreebankWordTokenizer()

    for i, c in enumerate(conversations):
        for u in c.utterances:
            remove_junk_labels(u.tags)
            y.append(u.tags)
            x.append([thank(u), e_mark(u), feedback(u), *sentiment_score(analyzer, u), *opinion_lex(tokenizer, u)])
        print('\r>>>> {}/{} done...'.format((i + 1), conv_count), end='')
    return np.asarray(x), np.asarray(y)


def combine_str_sent():
    X_str = np.load('embeddings/structural.npy', allow_pickle=True)
    X_sen = np.load('embeddings/sentimental.npy', allow_pickle=True)
    y = np.load('embeddings/labels.npy', allow_pickle=True)
    return np.hstack((X_str, X_sen)), y


def load_embeddings(group):
    if os.path.exists('embeddings/' + group + '.npy'):
        X = np.load('embeddings/' + group + '.npy', allow_pickle=True)
        y = np.load('embeddings/labels.npy', allow_pickle=True)
    else:
        data = read_data('/media/nommoinn/New Volume/veci/MSDialog/MSDialog-Intent.json')
        if group == 'structural':
            X, y = combine_structural(data)
        elif group == 'sentimental':
            X, y = combine_sentimental(data)
        #y = label_binarize(y, classes=['OQ', 'RQ', 'CQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'JK', 'OO'])
        y = preprocess_labels(y)
        y = MultiLabelBinarizer().fit_transform(y)
        np.save('embeddings/' + group, X)
        np.save('embeddings/labels.npy', y)
    return X, y


#X, y = load_embeddings('sentimental')
X, y = combine_str_sent()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

#model = ClassifierChain(LinearSVC(C=1, max_iter=1000, fit_intercept=True))
#model = ClassifierChain(AdaBoostClassifier())
#model = MLkNN(k=3, s=0.1)


model = ClassifierChain(RandomForestClassifier(
    n_estimators=1500,
    min_samples_split=7,
    min_samples_leaf=7,
    max_features='sqrt'
))

model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))
acc = jaccard_score(y_test, pred, average='samples')
print(f"Accuracy: {acc}")
