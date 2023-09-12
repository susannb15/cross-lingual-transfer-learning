#imports
import argparse
from datasets import load_dataset
import spacy
from tqdm import tqdm


# load dataset
dataset = load_dataset("wikipedia", "20220301.de", split='train[95:100%]')
#dataset = dataset.filter(lambda example, indice: indice < 5000, with_indices=True) # for testing

# split dataset into sentences

def create_text(text, data):
	sents = nlp(text)
	assert sents.has_annotation("SENT_START")
	data.extend(sents.sents)

nlp = spacy.load("de_core_news_sm")
wiki_sents = []
for article in tqdm(dataset["text"]):
	create_text(article, wiki_sents)
	
with open("spacy_sents.txt", "w+", encoding="utf-8") as f:
	for sent in wiki_sents:
		f.write(sent.text.strip()+"\n")
