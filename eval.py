import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_DISABLED"] = "true"

import transformers
from datasets import load_dataset
from datasets import DatasetDict
from datasets import ClassLabel, Value
from itertools import chain
import torch
import torch.nn as nn
import argparse
import math
from trainer_mod import My_Trainer
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from transformers import Trainer, TrainingArguments
from transformers.integrations import *
import re
from tqdm import tqdm

# for now one model, one eval run

parser = argparse.ArgumentParser(description="Options")

parser.add_argument('--model', type=str, help="Path to the native model.")
parser.add_argument('--dataset', type=str, help="Path to the evaluation dataset.")
parser.add_argument('--tokenizer', type=str, help="Tokenizer of the model.")
parser.add_argument('--file_name', type=str, help="Save eval metrics to this file.")

def set_seed(seed: int = 123):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)

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

def main():

	args = parser.parse_args()

	set_seed(5) 

	validation = load_dataset("text", data_files={'train': args.dataset})
	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
	models = []
	chps = []
	results = dict()
	try:
		model = AutoModelForCausalLM.from_pretrained(args.model)
		models.append(model)
		chps.append(0)
	except:
		for checkpoint in os.listdir(args.model):
			chp_path = os.path.join(args.model, checkpoint)
			models.append(AutoModelForCausalLM.from_pretrained(chp_path))
			chp = re.split("-", chp_path)[1]
			chps.append(chp)


	def tokenize_function(examples):
		return tokenizer(examples["text"])

	# tokenize data
	tokenized_datasets = validation.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
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
	#test_datasets = lm_datasets["train"].filter(lambda example, indice: indice < 10, with_indices=True)


	training_args = TrainingArguments(
		output_dir=args.model+"eval",
		do_train=False,
	)

	for model, chp in zip(models, chps):	
		trainer = My_Trainer(
			model=model,
			args=training_args,
			eval_dataset=lm_datasets["train"]
		)

		model_eval = trainer.evaluate()
		ppl = model_eval["eval_perplexity"]
		results[chp] = ppl

	with open(args.file_name, "w+", encoding='utf-8') as f:
		for chp in results:
			f.write(str(chp)+"\t"+str(results[chp])+"\n")

if __name__ == '__main__':
	main()
