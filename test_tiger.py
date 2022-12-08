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
from scipy.special import softmax

random.seed(42)

#datasets = load_dataset("text", encoding='ISO-8859-1', data_files={'train': 'tiger.txt'})

from datasets import ClassLabel, Value
import random
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

pretrained_model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")
from_scratch_model = AutoModelWithLMHead.from_pretrained("de_from_scratch/checkpoint-45000")
de_model = AutoModelWithLMHead.from_pretrained("de_tied/checkpoint-60000")
en_model = AutoModelWithLMHead.from_pretrained("en_tied/checkpoint-60000")
es_model = AutoModelWithLMHead.from_pretrained("es_tied/checkpoint-60000")

models = [("pretrained", pretrained_model),("from scratch", from_scratch_model), ("de", de_model),("en", en_model),("es", es_model)]

import math
from tqdm import tqdm

files = ["wiki_logits.txt", "tiger_logits.txt"]

with open("tiger_ISO.txt", "r", encoding='ISO-8859-1') as f:
	iso = f.readlines()

with open("tiger_UTF-8.txt", "r") as f:
	utf = f.readlines()


sent1 = iso[0]
sent2 = utf[0]

print(sent1)
print(tokenizer.encode(sent1))
print(sent2)
print(tokenizer.encode(sent2))


"""
with open("prediction_analysis.txt", "w+") as g:
	for fi in files:
		g.write("---------\n"+str(fi)+"\n\n")
		with open(fi, "r") as f:
			text = f.readlines()
			for name, model in models:
				g.write(f"MODEL: {name}\n")
				for line in (text): 
					g.write(line+"\n-------\n")
					with torch.no_grad():
						tokenized = tokenizer(line, return_tensors="pt")
						outputs = model(**tokenized, labels=tokenized["input_ids"])
						loss = outputs.loss
						logits = outputs.logits
						next_token_logits = logits[:, -1, :]
						probs = softmax(next_token_logits)[0]
						#best_pred = np.argmax(probs[0])
						preds = []
						for word_id in range(next_token_logits.size(dim=1)):
							pred_word = tokenizer.decode(word_id)
							preds.append((pred_word, probs[word_id]))
						preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
					g.write(str(preds_sorted[:10])+"\n"+"-------------------------------"+"\n")
"""
