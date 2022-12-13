import transformers
from datasets import load_dataset
from datasets import DatasetDict
from itertools import chain
import torch
import torch.nn as nn
import argparse
import math
from trainer_mod import My_Trainer
import wandb
import random
import spacy

random.seed(42)

nlp = spacy.load("de_core_news_sm")

datasets = load_dataset("wikipedia", "20220301.de", split='train[90:95%]')

from datasets import ClassLabel, Value
import random
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

pretrained = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")
native_model = AutoModelWithLMHead.from_pretrained("de_from_scratch/checkpoint-45000")
de_model = AutoModelWithLMHead.from_pretrained("de_tied/checkpoint-60000")
en_model = AutoModelWithLMHead.from_pretrained("en_tied/checkpoint-60000")
es_model = AutoModelWithLMHead.from_pretrained("es_tied/checkpoint-60000")

models = [("pretrained", pretrained), ("native", native_model),("de", de_model),("en", en_model),("es", es_model)]

#models = [("pretrained", pretrained)]

import math
from tqdm import tqdm

ppls = dict()
ppls["wikipedia"] = dict()
ppls["tiger"] = dict()
ppls["10kGNAD"] = dict()
#ppls["test"] = dict() 
# create Wikipedia sentences

wikipedia = []

for article in tqdm(datasets["text"]):
	sents = nlp(article)
	assert sents.has_annotation("SENT_START")
	wikipedia.extend(sents.sents)

wikipedia = wikipedia[:10000]
print(wikipedia[:3])

news = []
 
df = pd.read_csv("10kGNAD/articles.csv", delimiter="\t")
for article in tqdm(df["text"]):
	sents = nlp(article)
	assert sents.has_annotation("SENT_START")
	news.extend(sents.sents)

news = news[:10000]

with open("tiger_UTF-8.txt", "r", encoding='ISO-8859-1') as f:
	tiger = f.readlines()

tiger = tiger [:10000]

# for testing
#with open("xx.txt", "r") as f:
#	test = f.readlines()

def clean(dataset):
	clean_dataset = []
	for sent in dataset:
		sent = sent.lstrip()
		if sent and len(sent) <= 256:
			clean_dataset.append(sent)
	return clean_dataset
# for testing
#wikipedia = wikipedia[:1000]
#tiger = tiger[:1000]
#news = news[:1000]

valid = [("wikipedia", wikipedia), ("tiger", tiger), ("10kGNAD", news)]

#valid = [("test", test)]

problem_sents = []
other_problem_sents = []

for d, data in valid:
	dataset = clean(data)
	for name, model in models:
		ppl_total = 0
		dataset_len = len(dataset)
		for line in tqdm(dataset):
			with torch.no_grad():
				tokenized = tokenizer(line, return_tensors='pt')
				outputs = model(**tokenized, labels=tokenized["input_ids"])
				loss = outputs.loss
				ppl = math.exp(loss)
				if not math.isnan(ppl):
					ppl_total += math.exp(loss)
				else:
					problem_sents.append(line)
					dataset_len -= 1
				if ppl >= 10000:
					other_problem_sents.append(line)
		ppls[d][name] = ppl_total / len(dataset)

with open("problem_sents.txt", "w+") as f:
	for sent in problem_sents:
		f.write(sent)
		f.write("\n")
	f.write("\n\n")
	f.write("HIGH PPL\n\n")
	for sent in other_problem_sents:
		f.write(sent)
		f.write("\n")


with open("valid.txt", "w+") as f:
	for key in ppls:
		f.write(str(key)+"\n")
		for k in ppls[key]:
			f.write("AVG PPL of "+str(k)+": "+str(ppls[key][k])+"\n")
