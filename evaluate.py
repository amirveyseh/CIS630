import requests
from collections import Counter
import pickle
from tqdm import tqdm

from utils import scorer

# predictions1_gold_res_prob = requests.get('http://legendary1.cs.uoregon.edu:4000/process_test').json()
# predictions2_gold_res_prob = requests.get('http://legendary2.cs.uoregon.edu:4000/process_test').json()
# predictions3_gold_res_prob = requests.get('http://iq.cs.uoregon.edu:4000/process_test').json()
# predictions4_gold_res_prob = requests.get('http://hal.cs.uoregon.edu:4000/process_test').json()
#
# predictions1 = predictions1_gold_res_prob[0]
# gold = predictions1_gold_res_prob[1]
# res1 = predictions1_gold_res_prob[2]
# prob1 = predictions1_gold_res_prob[3]
#
# predictions2 = predictions2_gold_res_prob[0]
# res2 = predictions2_gold_res_prob[2]
# prob2 = predictions2_gold_res_prob[3]
#
# predictions3 = predictions3_gold_res_prob[0]
# res3 = predictions3_gold_res_prob[2]
# prob3 = predictions3_gold_res_prob[3]
#
# predictions4 = predictions4_gold_res_prob[0]
# res4 = predictions4_gold_res_prob[2]
# prob4 = predictions4_gold_res_prob[3]
#
# with open('caches/pred1.pkl', 'wb') as file:
#     pickle.dump((predictions1,res1, prob1),file)
# with open('caches/pred2.pkl', 'wb') as file:
#     pickle.dump((predictions2,res2, prob2),file)
# with open('caches/pred3.pkl', 'wb') as file:
#     pickle.dump((predictions3,res3, prob3),file)
# with open('caches/pred4.pkl', 'wb') as file:
#     pickle.dump((predictions4,res4, prob4),file)
# with open('caches/gold.pkl', 'wb') as file:
#     pickle.dump(gold,file)

## load predictions
with open('caches/pred1.pkl', 'rb') as file:
    pred1_res_prob = pickle.load(file)
    predictions1 = pred1_res_prob[0]
    res1 = pred1_res_prob[1]
    prob1 = pred1_res_prob[2]
with open('caches/pred2.pkl', 'rb') as file:
    pred1_res_prob = pickle.load(file)
    predictions2 = pred1_res_prob[0]
    res2 = pred1_res_prob[1]
    prob2 = pred1_res_prob[2]
with open('caches/pred3.pkl', 'rb') as file:
    pred1_res_prob = pickle.load(file)
    predictions3 = pred1_res_prob[0]
    res3 = pred1_res_prob[1]
    prob3 = pred1_res_prob[2]
with open('caches/pred4.pkl', 'rb') as file:
    pred1_res_prob = pickle.load(file)
    predictions4 = pred1_res_prob[0]
    res4 = pred1_res_prob[1]
    prob4 = pred1_res_prob[2]
with open('caches/gold.pkl', 'rb') as file:
    gold = pickle.load(file)

predictions = []

# # Majority Voting
# for i in range(len(predictions1)):
#     votes = []
#     votes.append(predictions1[i])
#     votes.append(predictions2[i])
#     votes.append(predictions3[i])
#     # votes.append(predictions4[i])
#     votes_count = Counter()
#     for vote in votes:
#         votes_count[vote] += 1
#     vote = votes_count.most_common(1)
#     predictions.append(vote[0][0])

## Prediction with highest confidence
for i in range(len(predictions1)):
    # probs = [max(prob1[i]),max(prob2[i]),max(prob3[i]),max(prob4[i])]
    probs = [max(prob1[i]),max(prob2[i]),max(prob3[i])]
    ind = probs.index(max(probs))
    # preds = [predictions1[i],predictions2[i],predictions3[i],predictions4[i]]
    preds = [predictions1[i],predictions2[i],predictions3[i]]
    predictions.append(preds[ind])


# ## Prediction using trained combiner
# for i in tqdm(range(len(predictions1))):
#     x = {
#         'p1': predictions1[i],
#         'p2': predictions2[i],
#         'p3': predictions3[i],
#         'p4': predictions4[i]
#     }
#     x = {
#         'p1': predictions1[i],
#         'p2': predictions2[i],
#         'p3': predictions3[i],
#         'p4': predictions4[i]
#     }
#     res = requests.post('http://legendary1.cs.uoregon.edu:4010/process_test', data=x).json()[0]
#     predictions.append(res)

# for i, p in enumerate(predictions):
#     if p != 0:
#         print(i)

p, r, f1 = scorer.score(gold, predictions, verbose=True)
print("evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(p,r,f1))

print("Evaluation ended.")