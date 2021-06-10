from flask import Flask, request, render_template, redirect, jsonify
import random
import argparse
from tqdm import tqdm
import torch
import json

from data.loader import DataLoader
from model.trainer import GCNTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='saved_models/00', help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
trainer = GCNTrainer(opt)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

@app.route('/process_test', methods=['GET'])
def process():
    data_file = opt['data_dir'] + '/test.json'
    print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
    batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

    helper.print_config(opt)
    label2id = constant.LABEL_TO_ID
    id2label = dict([(v, k) for k, v in label2id.items()])

    predictions = []
    all_probs = []
    batch_iter = tqdm(batch)
    for i, b in enumerate(batch_iter):
        preds, probs, _ = trainer.predict(b)
        predictions += preds
        all_probs += probs

    p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)
    return jsonify([predictions,batch.gold(),(p,r,f1),all_probs])

@app.route('/process', methods=['POST'])
def process_data():
    data = request.form['data']
    d = json.loads(data)
    d['subj_start'] = int(d['subj_start'])
    d['subj_end'] = int(d['subj_end'])
    with open('data.json', 'w') as file:
        json.dump([d,d,d],file)
    batch = DataLoader('data.json', opt['batch_size'], opt, vocab, evaluation=True)

    predictions = []
    all_probs = []
    batch_iter = tqdm(batch)
    for i, b in enumerate(batch_iter):
        preds, probs, _ = trainer.predict(b)
        predictions += preds
        all_probs += probs

    return jsonify([predictions,all_probs])

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000)


