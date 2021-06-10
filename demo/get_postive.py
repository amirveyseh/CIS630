import json
import pickle
import spacy
import requests
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

with open('../../dataset/doc/test.json') as file:
    test = json.load(file)

with open('../caches/gold.pkl', 'rb') as file:
    gold = pickle.load(file)

positive = []
with open('positive') as file:
    lines = file.readlines()
    for l in lines:
        positive.append(int(l))

# print(len(test),len(gold))
#
# for i in range(len(gold)):
#     if gold[i] != test[i]['relation'] and gold[i] != 0 and i in positive:
#         print(' '.join(test[i]['token']),' ::::::::: ',test[i]['subj_start'], ' : ',test[i]['subj_end'],' : ',test[i]['relation'])
#         print('='*100)

#############################################################################################33
for i in tqdm(range(len(test))):
    text = ' '.join(test[i]['token'])
    subj_start = test[i]['subj_start']
    subj_end = test[i]['subj_end']
    tokens = []
    pos = []
    head = []
    docs = []
    for sent in nlp(text).sents:
        tok = []
        ps = []
        for t in sent:
            tokens.append(t.text)
            pos.append(t.pos_)
            if t.head.i == t.i:
                head.append(0)
            else:
                head.append(t.head.i+1)
            tok.append(t.text)
            ps.append(t.pos_)
        doc = {
            'pos': ps,
            'token': tok,
            'triggers': [0]*len(tok),
            'event': 0
        }
        docs.append(doc)
    d = {
        'token': tokens,
        'stanford_pos': pos,
        'stanford_deprel': [0]*len(tokens),
        'stanford_head': head,
        'doc': docs,
        'subj_start': subj_start,
        'subj_end': subj_end,
        'relation': 0
    }

    predictions_prob1 = requests.post('http://legendary1.cs.uoregon.edu:4000/process', data={'data': json.dumps(d)}).json()
    predictions_prob2 = requests.post('http://legendary2.cs.uoregon.edu:4000/process', data={'data': json.dumps(d)}).json()
    predictions_prob3 = requests.post('http://iq.cs.uoregon.edu:4000/process', data={'data': json.dumps(d)}).json()
    predictions_prob4 = requests.post('http://hal.cs.uoregon.edu:4000/process', data={'data': json.dumps(d)}).json()


    x = {
        'p1': predictions_prob1[0][0],
        'p2': predictions_prob2[0][0],
        'p3': predictions_prob3[0][0],
        'p4': predictions_prob4[0][0]
    }
    res = requests.post('http://legendary1.cs.uoregon.edu:4010/process_test', data=x).json()[0]

    if int(res) == int(test[i]['relation']) and int(res) != 0:
        print(' '.join(test[i]['token']),' ::::::::: ',test[i]['subj_start'], ' : ',test[i]['subj_end'],' : ',test[i]['relation'])
        print('='*100)