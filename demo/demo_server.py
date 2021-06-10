from flask import Flask, request, render_template, redirect, jsonify
import spacy
import requests
import json

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/process_test', methods=['POST'])
def process():
    text = request.form['text']
    subj_start = int(request.form['subj_start'])
    subj_end = int(request.form['subj_end'])
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

    if res == 13:
          res = 'Discover - Injection'
    elif res == 15:
          res = 'Attack - Trojan'
    else:
          res = 'None'

    return jsonify(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4016)
