import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["WANDB_DISABLED"] = "true"

import transformers
from datasets import load_dataset
from datasets import DatasetDict
from datasets import Dataset
from datasets import concatenate_datasets
from itertools import chain
import torch
import torch.nn as nn
import argparse
import math
from trainer_mod import My_Trainer
import wandb
import random
import pandas as pd
import csv
import math
import numpy as np

random.seed(42)

parser = argparse.ArgumentParser(description="Options")

parser.add_argument('--name', type=str, help='Name of the output dir.')
#parser.add_argument('--config', type=str, help='Config file.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--group', type=str, help='Group parameter for wandb')
#parser.add_argument('--model_lng', type=str, help="Model language. Options: de, es, en, gpt2")
#parser.add_argument('--tied_weights', action='store_true')
#parser.add_argument('--no-tied_weights', dest='tied_weights', action='store_false')
#parser.set_defaults(tied_weights=True)

args = parser.parse_args()

wandb.init(group=args.group)


datasets = DatasetDict()
train = load_dataset("wikipedia", "20220301.de", split='train[:70%]')
validation = load_dataset("wikipedia", "20220301.de", split='train[90:95%]')
datasets = load_dataset("text", data_files={'validation2': 'tiger_UTF-8.txt'})
news_corpus = load_dataset("csv", delimiter="\t", data_files={'train': '10kGNAD/articles.csv'})
europarl = load_dataset("text", data_files={'train': 'europarl.txt'})
datasets["train"] = train
datasets["validation1"] = validation
datasets["validation3"] = news_corpus["train"]
datasets["validation4"] = europarl["train"]


from datasets import ClassLabel, Value
import random
import pandas as pd

def show_random_elements(dataset, num_examples=10):
	assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
	picks = []
	for _ in range(num_examples):
		pick = random.randint(0, len(dataset)-1)
		while pick in picks:
			pick = random.randint(0, len(dataset)-1)
		picks.append(pick)
	
	df = pd.DataFrame(dataset[picks])
	for column, typ in dataset.features.items():
		if isinstance(typ, ClassLabel):
			df[column] = df[column].transform(lambda i: typ.names[i])


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel, GPT2Config

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

#config = AutoConfig.from_pretrained(
#	"gpt2",
#	vocab_size=len(tokenizer),
#	n_ctx=256,
#	bos_token_id=tokenizer.bos_token_id,
#	eos_token_id=tokenizer.eos_token_id,
#	)

config = GPT2Config(n_ctx=256, vocab_size=len(tokenizer), bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
model = GPT2LMHeadModel(config)


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

train_dataset = lm_datasets["train"].filter(lambda example, indice: indice < 45000, with_indices=True)
eval_dataset={'wikipedia': lm_datasets["validation1"], 'tiger': lm_datasets["validation2"], '10kGNAD': lm_datasets["validation3"], 'europarl': lm_datasets["validation4"]}

from transformers import Trainer, TrainingArguments
from transformers.integrations import *

training_args = TrainingArguments(
	args.name,
	do_train=True,
	evaluation_strategy = "steps",
	learning_rate=args.lr,
	weight_decay=0.01,
	num_train_epochs=30,
	#max_steps=200000,
	eval_steps=15000,
	save_steps=15000,
	warmup_steps = 30000,
	seed=42
)


trainer = My_Trainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=eval_dataset
)

#print("----- Dataset Examples  -----")
#print(tokenizer.decode(lm_datasets["train"][0]["input_ids"]))

#for dataset in eval_dataset:
#	print(f'{dataset}:\t{tokenizer.decode(eval_dataset[dataset][0]["input_ids"])}')

print(train_dataset.shape)

#trainer.train()
#trainer.evaluate()
