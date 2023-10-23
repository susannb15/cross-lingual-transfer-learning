import spacy
from collections import defaultdict
from collections import Counter
from tqdm import tqdm

nlp = spacy.load("de_core_news_sm")
with open("probing/spacy_sents.txt", "r", encoding='utf-8') as f:
	text = f.readlines()

text = text[:10000]

tokens = list()
tok2pos = defaultdict(lambda: defaultdict(int))
pos2toks = defaultdict(set)

for sent in nlp.pipe(tqdm(text)):
	for word in sent:
		tokens.append(word.text)
		tok2pos[word.text][word.pos_] += 1
		pos2toks[word.pos_].add(word.text)

token_count = Counter(tokens)

print(list(tok2pos.items())[:5])
print(pos2toks["DET"])

