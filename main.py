import numpy as np
from collections import defaultdict
import os
import json
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import treebank
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from skmultilearn.adapt import MLkNN
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support


class conversation:
    def __init__(self, i, title, category, dialog_time, frequency, utterances):
        self.i = i
        self.title = title.lower()
        self.category = category
        self.dialog_time = dialog_time
        self.frequency = frequency
        self.utterances = utterances
        self.utterances_count = len(utterances)


class utterance:
    def __init__(self, i, utterance_pos, actor_type, user_id, utterance, vote, utterance_time, affiliation, is_answer,
                 tags):
        self.i = i
        self.utterance_pos = utterance_pos
        self.actor_type = actor_type
        self.user_id = user_id
        self.utterance = utterance.lower()
        self.vote = vote
        self.utterance_time = utterance_time
        self.affiliation = affiliation
        self.is_answer = is_answer
        self.tags = [t for t in tags.strip().split(" ")]


def read_data(location="./MSDialog/MSDialog-Intent.json"):
    conversations = []
    with open(location) as json_file:
        data = json.load(json_file)
        for k in data.fromkeys(data):
            current = data[str(k)]
            utternaces = []
            utterances_data = current['utterances']
            for u in utterances_data:
                utternaces.append(utterance(i=u['id'], utterance_pos=u['utterance_pos'], actor_type=u['actor_type'],
                                            user_id=u['user_id'], utterance=u['utterance'], vote=u['vote'],
                                            utterance_time=u['utterance_time'], affiliation=u['affiliation'],
                                            is_answer=u['is_answer'], tags=u['tags']))
            conversations.append(conversation(i=k, title=current['title'], category=current['category'],
                                              dialog_time=current['dialog_time'], frequency=current['frequency'],
                                              utterances=utternaces))
    return conversations


# Embed the entire corpus as tf-idf
def idf_embedding(data):
    idf_corpus = []
    tok = treebank.TreebankWordTokenizer()

    for i, c in enumerate(data):
        for u in c.utterances:
            idf_corpus.append(c.title)
            idf_corpus.append(u.utterance)

    vectorizer = TfidfVectorizer(tokenizer=tok.tokenize, stop_words='english')
    vectorizer.fit(idf_corpus)
    return vectorizer


def q_mark(utterance):
    u = utterance.utterance
    return '?' in u


def duplicate(utterance):
    u = utterance.utterance
    return 'same' in u or 'similar' in u


def w5h1(utterance):
    u = utterance.utterance
    w5 = ['what', 'where', 'when', 'why', 'who', 'how']
    return [w in u for w in w5]


# Structural embedding methods
def norm_abs_pos(conversation, utterance):
    return utterance.utterance_pos / conversation.utterances_count


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


# Label preprocess
def remove_junk_labels(labels):
    if len(labels) > 1 and 'GG' in labels:
        labels.remove('GG')
    if len(labels) > 1 and 'O' in labels:
        labels.remove('O')
    if len(labels) > 1 and 'JK' in labels:
        labels.remove('JK')
    # return labels


def preprocess_labels(y):
    u, count = np.unique(y, return_counts=True)
    count_sort_ind = np.argsort(-count)[:32]
    label_dict = defaultdict(int, dict.fromkeys(map(str, list(u[count_sort_ind])), 1))

    for i in range(len(y)):
        if not label_dict[str(y[i])]:
            y[i] = [random.choice(y[i])]
    return y


# Combine features methods
def combine_content(conversations):
    x = []
    conv_count = len(conversations)
    vectorizer = idf_embedding(conversations)

    for i, c in enumerate(conversations):
        dialog = ' '.join([u.utterance for u in c.utterances])
        dialog_vec = vectorizer.transform([dialog]).toarray()
        init_vec = vectorizer.transform([c.utterances[0].utterance]).toarray()
        for u in c.utterances:
            u_vec = vectorizer.transform([u.utterance]).toarray()
            x.append([q_mark(u), duplicate(u), *w5h1(u), cosine_similarity(u_vec, init_vec)[0][0],
                      cosine_similarity(u_vec, dialog_vec)[0][0]])
        print('\r>>>> {}/{} done...'.format((i + 1), conv_count), end='')
    return np.asarray(x)


def combine_structural(conversations):
    x = []
    conv_count = len(conversations)
    stemmer = SnowballStemmer('english')

    for i, c in enumerate(conversations):
        for u in c.utterances:
            x.append([u.utterance_pos, norm_abs_pos(c, u), length(u), len_uni(u), len_stem(stemmer, u), is_starter(u)])
        print('\r>>>> {}/{} done...'.format((i + 1), conv_count), end='')
    return np.asarray(x)


def combine_sentimental(conversations):
    x = []
    conv_count = len(conversations)
    analyzer = SentimentIntensityAnalyzer()
    tok = treebank.TreebankWordTokenizer()

    for i, c in enumerate(conversations):
        for u in c.utterances:
            x.append([thank(u), e_mark(u), feedback(u), *sentiment_score(analyzer, u), *opinion_lex(tok, u)])
        print('\r>>>> {}/{} done...'.format((i + 1), conv_count), end='')
    return np.asarray(x)


def combine_2_feats(group1, group2):
    X_g1 = np.load('features/' + group1 + '.npy', allow_pickle=True)
    X_g2 = np.load('features/' + group2 + '.npy', allow_pickle=True)
    return np.hstack((X_g1, X_g2))


def combine_3_feats(group1, group2, group3):
    X_g1 = np.load('features/' + group1 + '.npy', allow_pickle=True)
    X_g2 = np.load('features/' + group2 + '.npy', allow_pickle=True)
    X_g3 = np.load('features/' + group3 + '.npy', allow_pickle=True)
    return np.hstack((X_g1, X_g2, X_g3))


# Load labels/features from saved npy files
def load_labels():
    if os.path.exists('features/labels.npy'):
        y = np.load('features/labels.npy', allow_pickle=True)
    else:
        data = read_data('/media/nommoinn/New Volume/veci/MSDialog/MSDialog-Intent.json')
        conv_count = len(data)
        y = []
        for i, c in enumerate(data):
            for u in c.utterances:
                remove_junk_labels(u.tags)
                y.append(u.tags)
            print('\r>>>> {}/{} done...'.format((i + 1), conv_count), end='')
        y = preprocess_labels(y)
        y = MultiLabelBinarizer().fit_transform(y)
        np.save('features/labels', y)
    return y


def load_embeddings(group):
    if os.path.exists('features/' + group + '.npy'):
        X = np.load('features/' + group + '.npy', allow_pickle=True)
    else:
        data = read_data('./MSDialog/MSDialog-Intent.json')
        if group == 'structural':
            X = combine_structural(data)
        if group == 'content':
            X = combine_content(data)
        elif group == 'sentimental':
            X = combine_sentimental(data)
        np.save('features/' + group, X)
    return X


X = combine_2_feats('content', 'structural')
y = load_labels()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

accs = np.zeros(10)
precs = np.zeros(10)
recs = np.zeros(10)
f1s = np.zeros(10)
for i in range(10):
    # model = ClassifierChain(LinearSVC(C=1, max_iter=1000, fit_intercept=True))
    # model = ClassifierChain(AdaBoostClassifier())
    # model = MLkNN(k=3, s=0.1)
    model = ClassifierChain(RandomForestClassifier(
        n_estimators=1500,
        min_samples_split=7,
        min_samples_leaf=7,
        max_features='sqrt'
    ))

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accs[i] = jaccard_score(y_test, pred, average='samples')
    precs[i], recs[i], f1s[i], _ = precision_recall_fscore_support(y_test, pred, average='samples')

print(f"Accuracy: {accs.mean()} +- {accs.std()}")
print(f"Precision: {precs.mean()} +- {precs.std()}")
print(f"Recall: {recs.mean()} +- {recs.std()}")
print(f"F1: {f1s.mean()} +- {f1s.std()}")
