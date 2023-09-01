#imports
import argparse
from datasets import load_dataset
import spacy
from tqdm import tqdm

# load args
parser = argparse.ArgumentParser(description='Options')

parser.add_argument('--tokens', type=str, help="File with tokens.")
parser.add_argument('--n_sents', type=int, help="Number of sentences extracted per token.")

args = parser.parse_args()

# load dataset
dataset = load_dataset("wikipedia", "20220301.de", split='train[95:100%]')
dataset = dataset.filter(lambda example, indice: indice < 15000, with_indices=True) # for testing

# split dataset into sentences

def create_text(text, data):
	sents = nlp(text)
	assert sents.has_annotation("SENT_START")
	data.extend(sents.sents)

nlp = spacy.load("de_core_news_sm")
wiki_sents = []
for article in tqdm(dataset["text"]):
	create_text(article, wiki_sents)
	
# for each word find n sentences that contain that word
probing_sents = list()
n = args.n_sents
with open(args.tokens, "r") as f:
	tokens = f.readlines()
print(f"Expecting ideally {n*len(tokens)} sentences.")
for token in tokens:
	counter = 0
	print(f"Trying to extract {n} sentences for the token: {token}")
	for sent in wiki_sents:
		if token in sent.text:
			counter += 1
			match = sent.text.strip()
			probing_sents.append(match)
		if counter >= n:
			break
	print(f"Found {counter}.")

# write test sentences to file
print(f"Writing {len(probing_sents)} sentences to file...")
with open("probing_sents.txt", "w+", encoding="utf-8") as f:
	for sent in probing_sents:
		f.write(sent+"\n")
