import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["WANDB_DISABLED"] = "true"

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
import wandb
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel
import pandas as pd
from transformers import Trainer, TrainingArguments
from transformers.integrations import *


parser = argparse.ArgumentParser(description="Options")

parser.add_argument('--name', type=str, help='Name of the output dir.')
#parser.add_argument('--config', type=str, help='Config file.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--block_size', type=int, default=256)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_training_epochs', type=int)
parser.add_argument('--max_steps', type=int)
parser.add_argument('--eval_steps', type=int, default=1000)
parser.add_argument('--save_steps', type=int, default=1000)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--group', type=str, help='Group parameter for wandb')
#parser.add_argument('--model_lng', type=str, help="Model language. Options: de, es, en, gpt2")
parser.add_argument('--language', type=str, help="Training language: de|en|es")
parser.add_argument('--native_model', type=str, help="Path to the native model that should be adapted.")
parser.add_argument('--tied_weights', action='store_true')
parser.add_argument('--no-tied_weights', dest='tied_weights', action='store_false')
parser.add_argument('--output_dir', type=str, help="Output directory to save the model to.")
parser.add_argument('--shuffle_perc', type=float, help='Shuffle only X% of the embeddings')
parser.set_defaults(tied_weights=True)
parser.add_argument('--seed', type=int, help="Seed for the training.")
parser.add_argument('--noise_intensity', type=float, help='Standard deviation for the generation of a normal distribution with mean 0. The generated distribution is added to the embeddings to generate noisy embeddings.')

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

def tokenize_function(examples):
    return tokenizer(examples["text"])

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


def shuffle_part(weights, perc):
    array = np.arange(weights.shape[0])
    idx_dict = dict()
    num_samples = int(perc * len(array))
    shuffle_indeces = random.sample(array.tolist(), num_samples)
    shuffled = random.sample(shuffle_indeces, len(shuffle_indeces))
    for idx, idy in zip(shuffle_indeces, shuffled):
        idx_dict[idx] = idy
    array_new = []
    for el in array:
        if el in idx_dict:
            array_new.append(idx_dict[el])
        else:
            array_new.append(el)
    return weights[array_new]

def shuffle_embeddings(embeddings):
    """
    Shuffles the embedding matrix.
    """
    idx = torch.randperm(embeddings.shape[0])
    shuffled = embeddings[idx]
    return shuffled

def gauss(embeddings):
    """
    Returns an embedding matrix with gaussian noise.
    """
    mean = 0
    std = args.noise_intensity # intensity has to be defined in the args
    noise = np.random.normal(mean, std, size=(embeddings.shape))
    gauss_embeddings = embeddings.detach().numpy() + noise
    return torch.from_numpy(gauss_embeddings).float()

def freeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def main():

	args = parser.parse_args()

	wandb.init(group=args.group)
	set_seed(args.seed)

	datasets = DatasetDict()

	if args.language == "de":
		# load German data
		train = load_dataset("wikipedia", "20220301.de", split='train[70:90%]')
		validation = load_dataset("wikipedia", "20220301.de", split='train[90:95%]')
		datasets = load_dataset("text", data_files={'validation2': 'tiger_UTF-8.txt'})
		news_corpus = load_dataset("csv", delimiter="\t", data_files={'train': '10kGNAD/articles.csv'})
		europarl = load_dataset("text", data_files={'train': 'europarl.txt'})
		datasets["train"] = train
		datasets["validation1"] = validation
		datasets["validation3"] = news_corpus["train"]
		datasets["validation4"] = europarl["train"]

		# load German tokenizer
		tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")


	elif args.language == "en":
		# load English data
		train = load_dataset("wikipedia", "20220301.en", split='train[70:90%]')
		validation = load_dataset("wikipedia", "20220301.en", split='train[90:95%]')
		datasets["train"] = train
		datasets["validation"] = validation

		# load English tokenizer
		tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

	elif args.language == "es":
		# load Spanish data
		train = load_dataset("wikipedia.py", "20220301.es", split='train[70:90%]', beam_runner="DirectRunner")
		validation = load_dataset("wikipedia.py", "20220301.es", split='train[90:95%]', beam_runner="DirectRunner")
		datasets["train"] = train
		datasets["validation"] = validation

		# load Spanish tokenizer
		tokenizer = AutoTokenizer.from_pretrained("datificate/gpt2-small-spanish")

	else:
		print(f"{args.language} is not a valid language: de|en|es")

	# load native model
	model = AutoModelForCausalLM.from_pretrained(args.native_model)

	# adjust embedding size of the native model to match the tokenizer of the adaptation language
	embed_prior = model.transformer.wte.weight
	if model.transformer.wte.weight.shape[0] < len(tokenizer):
		# embeddings have to be filled up; e.g. EN model adapted on DE
		diff = len(tokenizer)-model.transformer.wte.weight.shape[0]
		print(f"DIFF: {diff}")
		perm = torch.randperm(embed_prior.shape[0])
		copy_idx = perm[:diff]
		copies = embed_prior[copy_idx]
		embed_new = torch.cat((embed_prior, copies), 0)
		model.transformer.wte.weight = embed_new
		print(f"Adjusted wte size: {embed_prior.shape != model.wte.shape}")

	if model.transforer.wte.weight.shape[0] > len(tokenizer):
		# embeddings have to be reduced; e.g. DE model adapted on EN
		embed_new = embed_prior[:len(tokenizer),:]
		model.transformer.wte.weight = embed_new
		print(f"Adjusted wte size: {embed_prior.shape != model.wte.shape}")

	# modify embeddings
	embed_prior = model.transformer.wte.weight

	if args.shuffle_perc is not None:
		print(f"Shuffle {args.shuffle_perc*100}% of the embedding matrix.")
		shuffled_embeddings = shuffle_part(embed_prior, args.shuffle_perc)
		model.transformer.wte.weight = nn.Parameter(shuffled_embeddings)
	elif args.noise_intensity is not None:
		print(f"Apply Gauss noise {args.noise_intensity} to embedding layer.")	
		gauss_embeddings = gauss(embed_prior)
		model.transformer.wte.weight = nn.Parameter(gauss_embeddings)
	else:
		shuffled_embeddings = shuffle_embeddings(embed_prior)
		model.transformer.wte.weight = nn.Parameter(shuffled_embeddings)
		print(f"WEIGHTS ARE SHUFFLED: {embed_prior != model.wte}")

	print(len(tokenizer) == model.transformer.wte.weight.shape[0])
	assert len(tokenizer) == model.transformer.wte.weight.shape[0]

	# tie weights
	if args.tied_weights:
		model.tie_weights()
	else:
		print("Untied weights are currently not supported!!!")

	# freeze parameters 
	freeze_model(model)
	model.transformer.wte.weight.requires_grad = True
	model.transformer.wpe.weight.requires_grad = True
	
	print(f"TIED WEIGHTS: {model.wte == model.lm_head} {model.transformer.wte.weight is model.lm_head.weight}")
	print("ADAPTED PARAMETERS:")
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name, param.shape)

	# tokenize data
	tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
	block_size=args.block_size

	lm_datasets = tokenized_datasets.map(
		group_texts,
		batched=True,
		batch_size=1000,
		num_proc=4,
	)

	# define fixed size for the dataset (because wikipedias differ in size)
	print(lm_datasets["train"].shape)

	training_args = TrainingArguments(
		args.output_dir,
		do_train=True,
		evaluation_strategy = "steps",
		learning_rate=args.lr,
		weight_decay=args.weight_decay,
		#num_train_epochs=args.num_train_epochs,
		max_steps=args.max_steps,
		eval_steps=args.eval_steps,
		save_steps=args.save_steps,
		warmup_steps=args.warmup_steps,
		seed=args.seed
	)


	trainer = My_Trainer(
		model=model,
		args=training_args,
		train_dataset=lm_datasets["train"],
		eval_dataset={'wikipedia': lm_datasets["validation1"], 'tiger': lm_datasets["validation2"], '10kGNAD': lm_datasets["validation3"]}
	)

	#trainer.train()
	#trainer.evaluate()

if __name__ == '__main__':
	main()
