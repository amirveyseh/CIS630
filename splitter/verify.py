import pickle
from collections import defaultdict

with open('all_splits.pkl', 'rb') as file:
    splits = pickle.load(file)

counter = defaultdict(int)
for k,v in splits.items():
    counter[v] += 1

print(counter)