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
model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")

"""
def tokenize_function(examples):
	return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size=256

def group_texts(examples):
	# Concatenate all texts.
	concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
	total_length = len(concatenated_examples[list(examples.keys())[0]])
	# We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
	# customize this part to your needs.
	if total_length >= block_size:
		total_length = (total_length // block_size) * block_size
	# Split by chunks of max_len.
	result = {
		k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
		for k, t in concatenated_examples.items()
		}
	result["labels"] = result["input_ids"].copy()
	return result

lm_datasets = tokenized_datasets.map(
	group_texts,
	batched=True,
	batch_size=1000,
	num_proc=4,
)
"""

import math
from tqdm import tqdm

with open("tiger.txt", "r", encoding='ISO-8859-1') as f:
	text = f.readlines()
	ppl_total = 0
	for line in tqdm(text): 
		with torch.no_grad():
			tokenized = tokenizer(line, return_tensors="pt")
			outputs = model(**tokenized, labels=tokenized["input_ids"])
			loss = outputs.loss
			ppl_total += math.exp(loss)
	print(f"AVG PPL: {ppl_total/len(text)}")

