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

random.seed(42)

datasets = load_dataset("wikipedia", "20220301.de", split='train[90:95%]')

from datasets import ClassLabel, Value
import random
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

native_model = AutoModelWithLMHead.from_pretrained("de_from_scratch/checkpoint-45000")
de_model = AutoModelWithLMHead.from_pretrained("de_tied/checkpoint-60000")
en_model = AutoModelWithLMHead.from_pretrained("en_tied/checkpoint-60000")
es_model = AutoModelWithLMHead.from_pretrained("es_tied/checkpoint-60000")

models = [("native", native_model),("de", de_model),("en", en_model),("es", es_model)]

import math
from tqdm import tqdm

ppls = dict()
ppls["wikipedia"] = dict()
ppls["tiger"] = dict()
ppls["10kGNAD"] = dict()

# create Wikipedia sentences

wikipedia = []

for article in tqdm(datasets["text"]):
	sents = article.split(".")
	wikipedia.extend(sents)

wikipedia = wikipedia[:50000]

news = []
 
df = pd.read_csv("10kGNAD/articles.csv", delimiter="\t")
for article in tqdm(df["text"]):
	sents = article.split(".")
	news.extend(sents)

news = news[:50000]

with open("tiger.txt", "r", encoding='ISO-8859-1') as f:
	tiger = f.readlines()

def clean(dataset):
	clean_dataset = []
	for sent in dataset:
        	if sent and len(sent) <= 256:
                	clean_dataset.append(sent)
	return clean_dataset
# for testing
#wikipedia = wikipedia[:100]
#tiger = tiger[:100]
#news = news[:100]

valid = [("wikipedia", wikipedia), ("tiger", tiger), ("10kGNAD", news)]

for d, data in valid:
	dataset = clean(data)
	for name, model in models:
		ppl_total = 0
		for line in tqdm(dataset):
			with torch.no_grad():
				tokenized = tokenizer(line, return_tensors='pt')
				outputs = model(**tokenized, labels=tokenized["input_ids"])
				loss = outputs.loss
				ppl = math.exp(loss)
				if not math.isnan(ppl):
					ppl_total += math.exp(loss)
		ppls[d][name] = ppl_total / len(dataset)

with open("valid.txt", "w+") as f:
	for key in ppls:
		f.write(str(key)+"\n")
		for k in ppls[key]:
			f.write("AVG PPL of "+str(k)+": "+str(ppls[key][k])+"\n")
