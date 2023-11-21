import spacy
from collections import defaultdict
from collections import Counter
from tqdm import tqdm

nlp = spacy.load("de_core_news_sm")
with open("probing/spacy_sents.txt", "r", encoding='utf-8') as f:
	text = f.readlines()

pos_prns = ["sein", "seines", "seinem", "seinen", "seine", "seiner", "ihr", "ihres", "ihrem", "ihren", "ihre", "ihrer"]

text_prns = list()
print("Filtering text...")
for sent in tqdm(text):
	if any(word in sent for word in pos_prns):
		text_prns.append(sent)

print(f"Before: {len(text)}, after: {len(text_prns)}.")

#text = text_prns[:10000] # for testing
text = text_prns

# create token - pos dicts
tokens = list()
tok2pos = defaultdict(lambda: defaultdict(int))
pos2toks = defaultdict(set)

for sent in nlp.pipe(tqdm(text)):
	for word in sent:
		tokens.append(word.text)
		tok2pos[word.text][word.pos_] += 1
		pos2toks[word.pos_].add(word.text)

token_count = Counter(tokens)

print("Created token - POS dicts.")

# get most common nouns
nouns = list()
verbs = list()
for word in tok2pos:
	if "NOUN" in tok2pos[word].keys():
		nouns.append((word, tok2pos[word]["NOUN"]))
	if "VERB" in tok2pos[word].keys():
		verbs.append((word, tok2pos[word]["VERB"]))
nouns_sorted = sorted(nouns, key=lambda x: x[1], reverse=True)
verbs_sorted = sorted(verbs, key=lambda x: x[1], reverse=True)
with open("nouns_most_common.txt", "w+", encoding='utf-8') as f:
	for noun, occ in nouns_sorted:
		f.write(noun+"\t"+str(occ)+"\n")
with open("verbs_most_common.txt", "w+", encoding='utf-8') as f:
	for verb, occ in verbs_sorted:
		f.write(verb+"\t"+str(occ)+"\n")
print("Created files with most common nouns, verbs.")
