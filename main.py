import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
import json

class conversation:
    def __init__(self, i, title, category, dialog_time, frequency, utterances):
        self.i = i;
        self.title = title;
        self.category = category;
        self.dialog_time = dialog_time;
        self.frequency = frequency;
        self.utterances = utterances;
        
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
        self.tags = [t for t in tags.split(" ")];


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

#Content embedding methods
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


#Structural embedding methods





#Sentiment embedding methods

