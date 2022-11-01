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

datasets = load_dataset("text", encoding='ISO-8859-1', data_files={'train': 'tiger.txt'})

from datasets import ClassLabel, Value
import random
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

native_model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")
de_model = AutoModelWithLMHead.from_pretrained("de_tied/checkpoint-100000")
en_model = AutoModelWithLMHead.from_pretrained("en_tied/checkpoint-100000")
es_model = AutoModelWithLMHead.from_pretrained("es_tied/checkpoint-100000")

models = [("native", native_model),("de", de_model),("en", en_model),("es", es_model)]

import math
from tqdm import tqdm

ppls = dict()

#with open("tiger.txt", "r", encoding='ISO-8859-1') as f:
	text = f.readlines()
	text = text[:10000]
	for name, model in models:
		ppl_total = 0
		for line in tqdm(text): 
			with torch.no_grad():
				tokenized = tokenizer(line, return_tensors="pt")
				outputs = model(**tokenized, labels=tokenized["input_ids"])
				loss = outputs.loss
				ppl_total += math.exp(loss)
		ppls[name] = ppl_total / len(text)

with open("ppls.txt", "w+") as f:
	for key in ppls:
		f.write("AVG PPL of "+str(key)+": "+str(ppls[key])+"\n")

