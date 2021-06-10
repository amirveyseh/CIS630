import json, random, glob, math
from tqdm import tqdm
import spacy, pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np
import difflib
from collections import defaultdict
from spacy.lang.en import English
# nlp = spacy.load("en_core_web_sm")
nlp = English()


# with open('raw/valid/seq.out') as file:
#     train_labels = file.readlines()
#
# with open('raw/valid/seq.in') as file:
#     train_sent = file.readlines()
#
# assert len(train_sent) == len(train_labels)
#
# dataset = []
#
# for i in range(len(train_labels)):
#     d = {
#         'tokens': train_sent[i].strip().split(),
#         'labels': train_labels[i].strip().split()
#     }
#     dataset.append(d)
#
# with open('dev.json', 'w') as file:
#     json.dump(dataset, file)

#############################################################################

# with open('../dataset/casie/processed/new_dataset.json') as file:
#     dataset = json.load(file)
#
# for d in tqdm(dataset):
#
#     doc = nlp(d['text'])
#     parse = []
#     tokens = []
#     pos = []
#     for i, token in enumerate(doc):
#         head = token.head.i
#         tokens.append(token.text)
#         pos.append(token.pos_)
#         if i == head:
#             head = -1
#         parse.append(head)
#
#     assert len(parse) == len(pos) == len(tokens)
#
#     d['tokens'] = tokens
#     d['head'] = parse
#     d['pos'] = pos
#
# with open('../dataset/casie/processed/parsed_dataset.json', 'w') as file:
#     json.dump(dataset, file)

#####################################################################################

# with open('train.json') as file:
#     dataset = json.load(file)
# with open('test.json') as file:
#     dataset += json.load(file)
#
# labels = {}
# i = 0
# short_labels = {}
# j = -1
#
# for d in dataset:
#     for l in d['labels']:
#         if l not in labels.keys():
#             i += 1
#             labels[l] = i
#         if len(l) > 1:
#             l = l[2:]
#             if l not in short_labels.keys():
#                 j += 1
#                 short_labels[l] = j
#
# print(labels)
# print(short_labels)

############################################################################################################

# with open('../dataset/casie/processed/parsed_dataset.json') as file:
#     dataset = json.load(file)
#
# filtered_dataset = []
#
# bads = 0
#
# for d in tqdm(dataset):
#     n = 0
#     event = d['text'][d['start']:d['end']]
#     event_token = [token.text for token in nlp(event)]
#     if not ' '.join(event_token) in ' '.join(d['tokens']):
#         bads += 1
#     else:
#         start = 0
#         for i, t in enumerate(d['tokens']):
#             if t == event_token[0] and all(d['tokens'][i+j] == event_token[j] for j in range(len(event_token))):
#                 n += 1
#                 start = i
#         if n > 1:
#             bads += 1
#         else:
#             d['subj_start'] = start
#             d['subj_end'] = start + len(event_token)
#             filtered_dataset.append(d)
#
# print(bads)
#
# with open('../dataset/casie/processed/final_dataset.json', 'w') as file:
#     json.dump(filtered_dataset, file)

#############################################################################################################

# with open('../dataset/casie/processed/stan/withneg/dataset.json') as file:
# with open('../dataset/casie/processed/stan/withneg/dataset_moreaccurate.json') as file:
with open('../dataset/casie/processed/stan/withneg/new_dataset.json') as file:
    dataset = json.load(file)

indices = list(range(len(dataset)))
random.shuffle(indices)
data = [dataset[i] for i in indices]

# with open('../dataset/casie/processed/stan/withneg/train.json', 'w') as file:
# with open('../dataset/casie/processed/stan/withneg/train_moreaccurate.json', 'w') as file:
with open('../dataset/casie/processed/stan/withneg/new/train.json', 'w') as file:
    json.dump(data[:int(len(data)*.8)], file)
# with open('../dataset/casie/processed/stan/withneg/dev.json', 'w') as file:
# with open('../dataset/casie/processed/stan/withneg/dev_moreaccurate.json', 'w') as file:
with open('../dataset/casie/processed/stan/withneg/new/dev.json', 'w') as file:
    json.dump(data[int(len(data)*.8):int(len(data)*.9)], file)
# with open('../dataset/casie/processed/stan/withneg/test.json', 'w') as file:
# with open('../dataset/casie/processed/stan/withneg/test_moreaccurate.json', 'w') as file:
with open('../dataset/casie/processed/stan/withneg/new/test.json', 'w') as file:
    json.dump(data[int(len(data)*.9):], file)

#####################################################################################################

# with open('../dataset/casie/processed/final_dataset.json') as file:
#     dataset = json.load(file)
#
# labels = {}
#
# for d in dataset:
#     if d['type'] not in labels:
#         labels[d['type']] = len(labels)
#
# print(labels)

#######################################################################################################

# with open('../dataset/casie/processed/train.json') as file:
#     dataset = json.load(file)
#
# new_data = []
#
# for d in dataset:
#     # data = {
#     #     'token': d['tokens'],
#     #     'stanford_pos': d['pos'],
#     #     'stanford_deprel': d['pos'],
#     #     'stanford_head': [h+1 for h in d['head']],
#     #     'subj_start': d['subj_start'],
#     #     'subj_end': d['subj_end'],
#     #     'relation': d['type']
#     # }
#     # new_data.append(data)
#     d['stanford_head'] = [h+1 for h in d['stanford_head']]
#
# with open('../dataset/casie/processed/train.json', 'w') as file:
#     json.dump(dataset, file)

###########################################################################################################

# # with open('../dataset/event/train.json') as file:
# with open('../dataset/casie/processed/stan/withneg/dataset.json') as file:
#     dataset = json.load(file)
#
# n = 0
# p = 0
#
# for d in dataset:
#     if d['relation'] == 'no_relation':
#         n += 1
#     else:
#         p += 1
#
# print(n/(n+p))

###########################################################################################################

# # with open('../dataset/casie/processed/stan/filtered_Dataset.json') as file:
# # with open('../dataset/casie/processed/stan/filtered_Dataset_moreaccurate.json') as file:
# with open('../dataset/casie/processed/stan/new_dataset.json') as file:
#     dataset = json.load(file)
#
# withneg = []
#
# for d in dataset:
#     withneg.append(d)
#     pop = [t for t in range(len(d['token'])) if not (d['subj_start'] <= t <= d['subj_end'])]
#     negs = random.sample(pop,min([8, len(pop)]))
#     for n in negs:
#         new_d = d.copy()
#         new_d['subj_start'] = n
#         new_d['subj_end'] = n
#         new_d['relation'] = 'no_relation'
#         withneg.append(new_d)
#
# # with open('../dataset/casie/processed/stan/withneg/dataset.json', 'w') as file:
# # with open('../dataset/casie/processed/stan/withneg/dataset_moreaccurate.json', 'w') as file:
# with open('../dataset/casie/processed/stan/withneg/new_dataset.json', 'w') as file:
#     json.dump(withneg, file)

#############################################################################################################

# with open('../dataset/casie/processed/parsed_dataset.pickle', 'rb') as file:
#     dataset = pickle.load(file)
#
# with open('../dataset/casie/processed/new_dataset.json') as file:
#     orig = json.load(file)
#
# stan_dataset = []
#
# for d in dataset:
#     old_d = orig[d[0]]
#     new_d = {
#         'token': d[1][0],
#         'stanford_pos': d[1][1],
#         'stanford_head': d[1][2],
#         'stanford_deprel': d[1][3],
#         'relation': old_d['type'],
#         'start': old_d['start'],
#         'end': old_d['end'],
#         'text': old_d['text'],
#         'i': d[0]
#     }
#     stan_dataset.append(new_d)
#
# # with open('../dataset/casie/processed/stan/dataset.json', 'w') as file:
# #     json.dump(stan_dataset, file)
# with open('../dataset/casie/processed/stan/dataset2.json', 'w') as file:
#     json.dump(stan_dataset, file)

###############################################################################################################

# # with open('../dataset/casie/processed/stan/dataset.json') as file:
# #     dataset = json.load(file)
# with open('../dataset/casie/processed/stan/dataset2.json') as file:
#     dataset = json.load(file)
# with open('../dataset/casie/processed/new_dataset.json') as file:
#     orig = json.load(file)
#
#
# bads = 0
# filtered_Dataset = []
# counted_is = set()
# bad_is = defaultdict(list)
# bad_events = set()
#
# print(len(dataset))
#
# for d in dataset:
#     n = 0
#     event = d['text'][d['start']:d['end']].split()
#     if ' '.join(event) not in ' '.join(d['token']):
#         bads += 1
#         bad_is[d['i']].append(d)
#         bad_events.add(' '.join(event))
#         # pass
#     else:
#         ind = 0
#         inds = []
#         for i, t in enumerate(d['token']):
#             if d['token'][i] == event[0] and all(d['token'][i+j] == event[j] for j in range(len(event))):
#                 n += 1
#                 inds += [i]
#         if n == 0:
#             # bads += 1
#             # bad_is[d['i']].append(d)
#             pass
#         elif n > 1:
#             chars = [len(w)+1 for w in d['token']]
#             ind_char = [math.fabs(sum(chars[:index])-d['start']) for index in inds]
#             ind = inds[ind_char.index(min(ind_char))]
#         else:
#             ind = inds[0]
#         d['subj_start'] = ind
#         d['subj_end'] = ind + len(event)-1
#         filtered_Dataset.append(d)
#         counted_is.add(d['i'])
#
#
# # for i in range(len(orig)):
# #     if i not in counted_is:
# #         print(orig[i])
# #         print(orig[i]['text'][orig[i]['start']:orig[i]['end']])
# #         print('='*26)
# #         for bad in bad_is[i]:
# #             print(bad)
# #             print('-'*13)
# #         exit(1)
#
#
# print(bads)
# print(len(filtered_Dataset))
# print(bad_events)
#
# # with open('../dataset/casie/processed/stan/filtered_Dataset.json', 'w') as file:
# #     json.dump(filtered_Dataset, file)
# with open('../dataset/casie/processed/stan/filtered_Dataset_moreAccurate.json', 'w') as file:
#     json.dump(filtered_Dataset, file)

##############################################################################################################3

# # with open('../dataset/casie/processed/stan/withneg/dev.json') as file:
# #     dataset = json.load(file)
# with open('../dataset/casie/processed/new_dataset.json') as file:
#     dataset = json.load(file)
#
# for d in dataset:
#     if d['text'] == "Bambenek suspects that attackers are harvesting credentials in hopes of gaining a small foothold into a company via an email account or to perpetuate further phishing scams.":
#         print(d)
#         print(d['text'][d['start']:d['end']])

################################################################################################################

# LABEL_TO_ID = {'PatchVulnerability': 0, 'DiscoverVulnerability': 1, 'Ransom': 2, 'Databreach': 3, 'Phishing': 4, 'no_relation': 5}
#
#
# with open('dataset/casie/processed/stan/withneg/bert/filtered_Dataset.json') as file:
#     data_bert = json.load(file)
#     berts = {}
#     for i in range(len(data_bert)):
#         berts[' '.join(data_bert[i]['token'])] = data_bert[i]
#
# bert = berts[' '.join(d['token'])]['bert']

##################################################################################################################

# with open('../dataset/casie/processed/stan/withneg/bert/filtered_Dataset.json') as file:
#     bert = json.load(file)
#
# with open('../dataset/casie/processed/stan/withneg/train.json') as file:
#     train = json.load(file)
# with open('../dataset/casie/processed/stan/withneg/dev.json') as file:
#     dev = json.load(file)
# with open('../dataset/casie/processed/stan/withneg/test.json') as file:
#     test = json.load(file)
#
# ids = {}
#
# for b in bert:
#     ids[' '.join(b['token'])] = len(ids)
#
#
#
# for t in train:
#     try:
#         t['id'] = ids[' '.join(t['token'])]
#     except:
#         pass
# with open('../dataset/casie/processed/stan/withneg/train.json', 'w') as file:
#     json.dump(train, file)
#
#
#
# for t in dev:
#     try:
#         t['id'] = ids[' '.join(t['token'])]
#     except:
#         pass
# with open('../dataset/casie/processed/stan/withneg/dev.json', 'w') as file:
#     json.dump(dev, file)
#
#
#
# for t in test:
#     try:
#         t['id'] = ids[' '.join(t['token'])]
#     except:
#         pass
# with open('../dataset/casie/processed/stan/withneg/test.json', 'w') as file:
#     json.dump(test, file)

##################################################################################################################

# # # nlp.add_pipe(nlp.create_pipe('sentencizer'))
# # import stanza
# #
# # nlp = stanza.Pipeline(lang='en', processors='tokenize')
# #
# # sent_to_doc = {}
# # doc_to_sent = defaultdict(set)
# #
# # num_docs = []
# #
# # for f in tqdm(glob.glob('../dataset/casie/*.json')):
# #     with open(f) as file:
# #         data = json.load(file)
# #         doc = nlp(data['content'])
# #         num_doc = 0
# #         # for sent in doc.sents:
# #         for sent in doc.sentences:
# #             num_doc += 1
# #             # sentence = ' '.join([t.text for t in nlp(sent.text)])
# #             sentence = ' '.join([t.text for t in sent.tokens])
# #             document = data['sourcefile'][:-4]
# #             sent_to_doc[sentence] = document
# #             doc_to_sent[document].add(sentence)
# #         num_docs.append(num_doc)
# #
# # print('avg sent in doc: ', sum(num_docs)/len(num_docs))
# #
# # with open('sent_to_doc.pickle', 'wb') as file:
# #     pickle.dump(sent_to_doc, file)
# # with open('doc_to_sent.pickle', 'wb') as file:
# #     pickle.dump(doc_to_sent, file)
#
# with open('sent_to_doc.pickle', 'rb') as file:
#     sent_to_doc = pickle.load(file)
# with open('doc_to_sent.pickle', 'rb') as file:
#     doc_to_sent = pickle.load(file)
#
# # for k,v in sent_to_doc.items():
# #     if 'Good Weather' in k:
# #         print(k)
# # exit(1)
#
# sentences = defaultdict(list)
# sentences2 = {}
# orig_dataset = []
#
# with open('../dataset/casie/processed/stan/withneg/train.json') as file:
#     dataset = json.load(file)
# with open('../dataset/casie/processed/stan/withneg/dev.json') as file:
#     dataset += json.load(file)
# with open('../dataset/casie/processed/stan/withneg/test.json') as file:
#     dataset += json.load(file)
#
# for i, d in enumerate(dataset):
#     sentences[' '.join(d['token'])].append(i)
#     if d['relation'] != 'no_relation':
#         sentences2[' '.join(d['token'])] = i
#
# dataset_doc = []
#
# bads = 0
# badbad = 0
#
# with open('../dataset/casie/processed/stan/withneg/test.json') as file:
#     orig_dataset = json.load(file)
#
# close_sent_to_doc = {}
# close_sent2 = {}
#
# for d in tqdm(orig_dataset):
#     new_d = d.copy()
#     new_d['doc'] = []
#     # try:
#     good = True
#     try:
#         doc = sent_to_doc[' '.join(d['token'])]
#     except:
#         query_sent = ' '.join(d['token'])
#         if query_sent in close_sent_to_doc:
#             close_sent = close_sent_to_doc[query_sent]
#         else:
#             close_sent = difflib.get_close_matches(query_sent, sent_to_doc.keys(), n=1)[0]
#             close_sent_to_doc[query_sent] = close_sent
#         doc = sent_to_doc[close_sent]
#     sents = doc_to_sent[doc]
#     for sent in sents:
#         try:
#             id = sentences2[sent]
#         except:
#             # try:
#             if sent in close_sent2:
#                 close_sent = close_sent2[sent]
#             else:
#                 close_sent = difflib.get_close_matches(sent, sentences2.keys(), n=1)
#                 close_sent2[sent] = close_sent
#             if len(close_sent) == 0:
#                 bads += 1
#                 continue
#             else:
#                 close_sent = close_sent[0]
#             # except:
#             #     bads += 1
#             #     continue
#             id = sentences2[close_sent]
#         d = dataset[id]
#         new_d['doc'].append({
#             'token': d['token'],
#             'pos': d['stanford_pos'],
#             'triggers': [1 if d['subj_start'] <= i <= d['subj_end'] else 0 for i in range(len(d['token']))],
#             'event': 1 if d['relation'] != 'no_relation' else 0
#         })
#         # except:
#         #     bads += 1
#         #     if good:
#         #         badbad += 1
#         #         good = False
#         #     pass
#     # except:
#     #     pass
#     dataset_doc.append(new_d)
#
# print(bads)
# print(badbad)
# print(len(orig_dataset))
# print(len(dataset_doc))
#
# lens = []
# for d in dataset_doc:
#     lens.append(len(d['doc']))
# print(sum(lens)/len(lens))
#
# with open('../dataset/casie/processed/stan/withneg/doc/test.json', 'w') as file:
#     json.dump(dataset_doc, file)

##############################################################################################################3

# with open('../dataset/casie/processed/new_dataset.json') as file:
#     dataset = json.load(file)
#
# relations = defaultdict(int)
#
# for d in dataset:
#     relations[d['type']] += 1
#
# print(relations)

#########################################################################################################3

# with open('../dataset/casie/processed/stan/new_dataset.json') as file:
#     dataset = json.load(file)
#
# for d in dataset:
#     d['subj_start'] = d['start']
#     d['subj_end'] = d['end']
#
# with open('../dataset/casie/processed/stan/new_dataset.json', 'w') as file:
#     json.dump(dataset, file)
