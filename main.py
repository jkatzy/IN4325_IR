import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
import os
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import demo_liu_hu_lexicon
from nltk.corpus import stopwords
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
from skmultilearn.adapt import MLkNN
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import jaccard_score


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
        self.utterance = utterance;
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


def len_stem(utterance):
    stemmer = SnowballStemmer('english')
    return len(set(stemmer.stem(w) for w in utterance.utterance.split() if w not in stopwords.words('english')))


def is_starter(utterance):
    return utterance.actor_type == 'User'

# Sentiment embedding methods

def thank(utterance):
    u = utterance.utterance.lower()
    return 'thank' in u

def e_mark(utterance):
    u = utterance.utterance.lower()
    return '!' in u

def feedback(utterance):
    u = utterance.utterance.lower()
    return 'did not' in u or 'does not' in u

def opinion_lexicon(utterance):
    u = utterance.utterance.lower()
    return 1 if 'Positive' in str(demo_liu_hu_lexicon(u)) else 0

def sentiment_score(utterance):
    u = utterance.utterance.lower()
    Analyzer = SentimentIntensityAnalyzer()
    return Analyzer.polarity_scores(text=u)['compound']


def combine_structural(conversations):
    x = []
    y = []
    conv_count = len(conversations)

    for i, c in enumerate(conversations):
        for u in c.utterances:
            x.append([u.utterance_pos, norm_abs_pos(c, u), length(u), len_uni(u), len_stem(u)])
            #x.append([thank(u), e_mark(u), feedback(u), opinion_lexicon(u), sentiment_score(u)])
            y.append(u.tags)
        print('\r>>>> {}/{} done...'.format((i + 1), conv_count), end='')
    return np.asarray(x), np.asarray(y)


def load_embeddings(group):
    if os.path.exists('embeddings/' + group + '.npz'):
        data = np.load('embeddings/' + group + '.npz', allow_pickle=True)
        X = data['X']
        y = data['y']
    else:
        data = read_data('/media/nommoinn/New Volume/veci/MSDialog/MSDialog-Intent.json')
        X, y = combine_structural(data)
        #y = label_binarize(y, classes=['OQ', 'RQ', 'CQ', 'FD', 'FQ', 'IR', 'PA', 'PF', 'NF', 'GG', 'JK', 'OO'])
        y = MultiLabelBinarizer().fit_transform(y)
        np.savez('embeddings/' + group, X=X, y=y)
    return X, y


X, y = load_embeddings('structural')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

#model = ClassifierChain(LinearSVC(C=1, max_iter=1000, fit_intercept=True))
#model = ClassifierChain(AdaBoostClassifier())
model = RandomForestClassifier()
#model = MLkNN(k=3, s=0.1)

model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = jaccard_score(y_test, pred, average='samples')
print(f"Accuracy: {acc}")
