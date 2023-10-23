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
"""
tiger = pd.read_csv("tiger_UTF-8.txt", sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8', names=["text"])
news_corpus = pd.read_csv("10kGNAD/articles.csv", sep="\t", header=0)
europarl = pd.read_csv("europarl.txt", sep="\t", names=["text"])

tiger.dropna(inplace=True)
tiger.reset_index(drop=True, inplace=True)
news_corpus.dropna(inplace=True)
news_corpus.reset_index(drop=True, inplace=True)
europarl.dropna(inplace=True)
europarl.reset_index(drop=True, inplace=True)

max_len = math.ceil(min(len(tiger), len(news_corpus), len(europarl))*0.9)

tiger = tiger[:max_len]
news_corpus = news_corpus[:max_len]
europarl = europarl[:max_len]

tiger.reset_index(drop=True, inplace=True)
news_corpus.reset_index(drop=True, inplace=True)
europarl.reset_index(drop=True, inplace=True)

#print(f"CORPUS LENGTHS: {len(tiger)}, {len(news_corpus)}, {len(europarl)}")
#print(f"CORPORA: {tiger} {news_corpus} {europarl}")

corpora = pd.concat([tiger, news_corpus, europarl])

corpora = corpora[["text"]]


#corpora.drop(corpora.columns[1], axis=1, inplace=True)
#corpora.dropna(inplace=True)

print(f"CONCAT: {len(corpora)} {corpora}")
print(f"Length of the pandas corpus: {len(corpora)}")
"""
datasets = DatasetDict()
train = load_dataset("wikipedia", "20220301.de", split='train[:10%]') #70
val_wiki = load_dataset("wikipedia", "20220301.de", split='train[90:95%]')
tiger = load_dataset("text", data_files={'train': 'tiger_UTF-8.txt'})
news_corpus = load_dataset("csv", delimiter="\t", data_files={'train': '10kGNAD/articles.csv'})
europarl = load_dataset("text", data_files={'train': 'europarl.txt'})
#d["train"] = concatenate_datasets([tiger, news_corpus, europarl], split='train[:95%]')
datasets["wiki_train"]= train
datasets["wiki_val"] = val_wiki
datasets["tiger"] = tiger ["train"]
datasets["news_corpus"] = news_corpus["train"]
datasets["europarl"] = europarl ["train"]

#dataset = Dataset.from_pandas(corpora)

#train = train.filter(lambda example, indice: indice < math.ceil(max_len*0.9), with_indices=True)

#datasets["train"] = concatenate_datasets([dataset, train])
#datasets["validation1"] = val_wiki
#datasets["validation2"] = Dataset.from_pandas(tiger[math.ceil(len(tiger)*0.9):])
#datasets["validation3"] = Dataset.from_pandas(news_corpus[math.ceil(len(news_corpus)*0.9):])
#datasets["validation4"] = Dataset.from_pandas(europarl[math.ceil(len(europarl)*0.9):])


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


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

config = AutoConfig.from_pretrained(
	"gpt2",
	vocab_size=len(tokenizer),
	n_ctx=256,
	bos_token_id=tokenizer.bos_token_id,
	eos_token_id=tokenizer.eos_token_id
	)
#config = args.config
model = GPT2LMHeadModel(config)

#tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
	return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

#counter = 0
block_size=256

#tokenized_datasets["train"] = tokenized_datasets["train"].map(remove_columns = ["__index_level_0__", "url", "title", "id"])

def group_texts(examples):
	# Concatenate all texts.
	concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
		#except TypeError:
		#	concatenated_examples = {k: list(chain(*[examples[k]])) for k in examples.keys()}
	#except TypeError:
	#	concatenated_examples = examples
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
	#except TypeError:
		#for k in examples.keys():
		#	print(k, examples[k])
	#	global counter
	#	counter += 1

lm_datasets = tokenized_datasets.map(
	group_texts,
	batched=True,
	batch_size=1000,
	num_proc=4,
)


max_len = max([len(lm_datasets[d]["input_ids"]) for d in lm_datasets])
train_len = math.ceil(max_len*0.9)
val_len = max_len


wiki_train_len = math.ceil(len(lm_datasets["wiki_train"]["input_ids"])*0.9) if len(lm_datasets["wiki_train"]["input_ids"]) < train_len else train_len
tiger_train_len = math.ceil(len(lm_datasets["tiger"]["input_ids"])*0.9) if len(lm_datasets["tiger"]["input_ids"]) < train_len else train_len
news_train_len = math.ceil(len(lm_datasets["news_corpus"]["input_ids"])*0.9) if len(lm_datasets["news_corpus"]["input_ids"]) < train_len else train_len
europarl_train_len = math.ceil(len(lm_datasets["europarl"]["input_ids"])*0.9) if len(lm_datasets["europarl"]["input_ids"]) < train_len else train_len

wiki_short = lm_datasets["wiki_train"].filter(lambda example, indice: indice < wiki_train_len, with_indices=True)
tiger_short = lm_datasets["tiger"].filter(lambda example, indice: indice < tiger_train_len, with_indices=True)
tiger_val = lm_datasets["tiger"].filter(lambda example, indice: indice < val_len, with_indices=True).filter(lambda example, indice: indice >=  tiger_train_len, with_indices=True)
news_short = lm_datasets["news_corpus"].filter(lambda example, indice: indice < news_train_len, with_indices=True)
news_val = lm_datasets["news_corpus"].filter(lambda example, indice: indice < val_len, with_indices=True).filter(lambda example, indice: indice >=  news_train_len, with_indices=True)
europarl_short = lm_datasets["europarl"].filter(lambda example, indice: indice < europarl_train_len, with_indices=True)
europarl_val = lm_datasets["europarl"].filter(lambda example, indice: indice < val_len, with_indices=True).filter(lambda example, indice: indice >=  europarl_train_len, with_indices=True)


train = concatenate_datasets([wiki_short, tiger_short, news_short, europarl_short])

# shuffle train data
train.shuffle()

from transformers import Trainer, TrainingArguments
from transformers.integrations import *

training_args = TrainingArguments(
	args.name,
	do_train=True,
	evaluation_strategy = "steps",
	learning_rate=args.lr,
	weight_decay=0.01,
	#num_train_epochs=20,
	max_steps=165000,
	eval_steps=15000,
	save_steps=15000,
	warmup_steps = 30000,
	seed=42
)


trainer = My_Trainer(
	model=model,
	args=training_args,
	train_dataset=train,
	eval_dataset={'wikipedia': lm_datasets['wiki_val'], 'tiger': tiger_val, '10kGNAD': news_val, 'europarl': europarl_val}
)


trainer.train()
trainer.evaluate()
