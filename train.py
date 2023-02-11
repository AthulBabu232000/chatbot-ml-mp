import json
from nltkutils import tokenize, stem, bag_of_words
with open('intents.json','r') as f:
    intents=json.load(f)

all_words=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words=['?',',','!','.']
all_words=sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags=sorted(set(tags))
print(tags)