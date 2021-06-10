import json

from pytorch_transformers import BertModel
from pytorch_transformers import *
from collections import defaultdict
import torch
from tqdm import tqdm

#model = BertModel.from_pretrained('contractbert', output_hidden_states=True, output_attentions=True)
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizer = BertTokenizer('contractbert/vocab.txt')


def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    '''Aligns tokenized and untokenized sentence given subwords "##" prefixed
    Assuming that each subword token that does not start a new word is prefixed
    by two hashes, "##", computes an alignment between the un-subword-tokenized
    and subword-tokenized sentences.
    Args:
      tokenized_sent: a list of strings describing a subword-tokenized sentence
      untokenized_sent: a list of strings describing a sentence, no subword tok.
    Returns:
      A dictionary of type {int: list(int)} mapping each untokenized sentence
      index to a list of subword-tokenized sentence indices
    '''
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while (untokenized_sent_index < len(untokenized_sent) and
        tokenized_sent_index < len(tokenized_sent)):
      while (tokenized_sent_index + 1 < len(tokenized_sent) and
          tokenized_sent[tokenized_sent_index + 1].startswith('##')):
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        tokenized_sent_index += 1
      mapping[untokenized_sent_index].append(tokenized_sent_index)
      untokenized_sent_index += 1
      tokenized_sent_index += 1
    return mapping

with open('../dataset/doc/dataset.json') as file:
    dataset = json.load(file)

def extractbert(s, d):
    tokenized_sent = tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + ' '.join(s) + ' [SEP]')
    untok_tok_mapping = match_tokenized_to_untokenized(tokenized_sent, s)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
    segment_ids = [1 for x in tokenized_sent]

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])
    if tokens_tensor.shape[1] < 512:
        encoded_layers, _ = model(tokens_tensor, segments_tensors)[-2:]
        single_layer_features = torch.cat(encoded_layers[-1:], dim=2)
        representation = torch.stack(
            [torch.mean(single_layer_features[0, untok_tok_mapping[j][0]:untok_tok_mapping[j][-1] + 1, :],
                        dim=0)
            for j in range(len(s))], dim=0)
        representation = representation.view(1, *representation.size())[0]
        final_result = representation.data.cpu().numpy().tolist()
        assert len(final_result) == len(d['token'])
        return final_result
    else:
        return None

for d in tqdm(dataset):
    s = d['token']
    bert_vec = extractbert(s, d)
    if bert_vec is not None:
        d['bert'] = bert_vec
    # for i in range(min([len(d['doc']),5])):
    #     doc = d['doc'][i]
    #     s = doc['token']
    #     bert_vec = extractbert(s, doc)
    #     if bert_vec is not None:
    #         doc['bert'] = bert_vec
    #     d['doc'][i] = doc

with open('../dataset/doc/bert/dataset.json', 'w') as file:
    json.dump(dataset, file)




