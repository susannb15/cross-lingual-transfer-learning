#imports
import argparse
from datasets import load_dataset
import spacy
from tqdm import tqdm

# load args
parser = argparse.ArgumentParser(description='Options')

parser.add_argument('--tokens', type=str, help="File with tokens.")
parser.add_argument('--sents', type=str, help="File with the spacy sentences.")
parser.add_argument('--n_sents', type=int, help="Number of sentences extracted per token.")

args = parser.parse_args()

# load sentences
with open(args.sents, "r", encoding="utf-8") as f:
	spacy_sents = f.readlines()

# for each word find n sentences that contain that word
probing_sents = list()
tokens = list()
with open(args.tokens, "r", encoding="utf-8") as f:
	text = f.readlines()
	for token in text:
		tokens.append(token.strip())
print(f"Expecting ideally {args.n_sents*len(tokens)} sentences.")

for token in tokens:
	counter = 0
	print(f"Trying to extract {args.n_sents} sentences for the token: {token}")
	for sent in spacy_sents:
		if token in sent.split() and counter >= args.n_sents:
			with open("probing_sents_test.txt", "a", encoding="utf-8") as f:
				f.write(sent)
			break
		if token in sent.split() and counter < args.n_sents:
			counter += 1
			match = sent
			probing_sents.append(match)
	print(f"Found {counter} and 1 for testing.")

# write test sentences to file
print(f"Writing {len(probing_sents)} sentences to file...")
with open("probing_sents.txt", "w+", encoding="utf-8") as f:
	for sent in probing_sents:
		f.write(sent)
