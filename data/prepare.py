import json, random, glob, os
from tqdm import tqdm

# with open('../dataset/event/data.ED') as file:
#     dataset = json.load(file)[0]
#
# indices = list(range(len(dataset)))
# random.shuffle(indices)
# data = [dataset[i] for i in indices]
#
# new_data = []
#
# for d in data:
#     new_data.append({
#         'id': d['entry_id'],
#         'relation': d['event_type'],
#         'token': d['word'],
#         'subj_start': d['anchor'][0],
#         'subj_end': d['anchor'][-1],
#         'stanford_pos': d['upos'],
#         'stanford_head': d['head'],
#         'stanford_deprel': d['dep_rel']
#     })
#
# with open('../dataset/event/train.json', 'w') as file:
#     json.dump(new_data, file)
#
#######################################################################################3

# with open('../dataset/event/dataset.json') as file:
#     dataset = json.load(file)
#
# train = dataset[:8*len(dataset)//10]
# dev = dataset[8*len(dataset)//10:9*len(dataset)//10]
# test = dataset[9*len(dataset)//10:10*len(dataset)//10]
#
# with open('../dataset/event/train.json', 'w') as file:
#     json.dump(train, file)
# with open('../dataset/event/dev.json', 'w') as file:
#     json.dump(dev, file)
# with open('../dataset/event/test.json', 'w') as file:
#     json.dump(test, file)

###########################################################################################

# with open('../dataset/event/dataset.json') as file:
#     dataset = json.load(file)
#
# labels = set()
# for d in dataset:
#     labels.add(d['relation'])
# print(labels)

############################################################################################
# lines = []
# for f in glob.glob('../../../data/*.parsed'):
#     with open(f) as file:
#         lines += file.readlines()
#
# with open('../dataset/event/dataset.json') as file:
#     dataset = json.load(file)
#
# print(len(dataset))
# sent = lines[0].strip().split('\t')[0]
# for d in dataset:
#     if ' '.join(d['token']) == sent:
#         print(d)
# t = 0
# for l in lines:
#     t += len(l.strip().split('\t')[-1].split())/2
#     if l.strip().split('\t')[0] == sent:
#         print(l)
# print(t)
# # print(lines[0].strip().split('\t')[-1].split())

############################################################################################

# with open('../dataset/event/train.json') as file:
#     dataset = json.load(file)
#
# new_data = []
#
# for d in tqdm(dataset):
#     docId = d['id'].split('.')[0]
#     doc = []
#     with open('../../../data/'+docId+'.parsed') as file:
#         lines = file.readlines()
#         for l in lines:
#             sent = l.strip().split('\t')[0].split()
#             pos = l.strip().split('\t')[2].split()
#             triggers = l.strip().split('\t')[-1].split()
#             labels = [0]*len(sent)
#             for i in range(len(triggers)):
#                 if i % 2 == 1:
#                     ts = triggers[i].split(',')
#                     for t in ts:
#                         labels[int(t)] = 1
#             hasevent = 0
#             if sum(labels) > 0:
#                 hasevent = 1
#             doc.append({
#                 'token': sent,
#                 'pos': pos,
#                 'triggers': labels,
#                 'event': hasevent
#             })
#     d['doc'] = doc
#     new_data.append(d)
#
# with open('../dataset/doc/train.json', 'w') as file:
#     json.dump(new_data, file)

###################################################################################################3

with open('../dataset/doc/bert/dev.json') as file:
    dataset = json.load(file)

for d in dataset:
    assert d['bert'] is not None
    for doc in d['doc']:
        assert doc['bert'] is not None