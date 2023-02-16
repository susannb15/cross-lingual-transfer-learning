import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["WANDB_DISABLED"] = "true"

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

parser = argparse.ArgumentParser(description="Options")

parser.add_argument('--name', type=str, help='Name of the output dir.')
#parser.add_argument('--config', type=str, help='Config file.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--group', type=str, help='Group parameter for wandb')
parser.add_argument('--model_lng', type=str, help="Model language. Options: de, es, en")
parser.add_argument('--tied_weights', action='store_true')
parser.add_argument('--no-tied_weights', dest='tied_weights', action='store_false')
parser.set_defaults(tied_weights=True)

args = parser.parse_args()

wandb.init(group=args.group)

datasets = DatasetDict()
#train = load_dataset("wikipedia.py", "20220301.es", split='train[:3%]', beam_runner="DirectRunner")
train = load_dataset("wikipedia.py", "20220301.es", split='train[30:45%]', beam_runner="DirectRunner")
#validation = load_dataset("wikipedia.py", "20220301.es", split='train[4:5%]', beam_runner="DirectRunner")
validation = load_dataset("wikipedia.py", "20220301.es", split='train[90:95%]', beam_runner="DirectRunner")
datasets["train"] = train
datasets["validation"] = validation

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

tokenizer = AutoTokenizer.from_pretrained("datificate/gpt2-small-spanish")

if args.model_lng == "es":
	model = AutoModelForCausalLM.from_pretrained("es_from_scratch/checkpoint-45000")
	#embeddings = model.transformer.wte.weight
	#perm = torch.randperm(embeddings.shape[0])
	#copy_idx = perm[:8]
	#copies = embeddings[copy_idx]
	#extended_emb = torch.cat((embeddings, copies), 0)
	#model.transformer.wte.weight = nn.Parameter(extended_emb)

	#if not args.tied_weights:
		#lm_head = model.lm_head.weight
		#perm = torch.randperm(lm_head.shape[0])
		#copy_idx = perm[:8]
		#copies = lm_head[copy_idx]
		#extended_head = torch.cat((lm_head, copies), 0)
		#model.lm_head.weight = nn.Parameter(extended_head)


elif args.model_lng == "en":
	model = AutoModelWithLMHead.from_pretrained("en_from_scratch/checkpoint-45000")
	#embeddings = model.transformer.wte.weight
	#perm = torch.randperm(embeddings.shape[0])
	#copy_idx = perm[:8]
	#copies = embeddings[copy_idx]
	#extended_emb = torch.cat((embeddings, copies), 0)
	#model.transformer.wte.weight = nn.Parameter(extended_emb)

	#if not args.tied_weights:
		#lm_head = model.lm_head.weight
		#perm = torch.randperm(lm_head.shape[0])
		#copy_idx = perm[:8]
		#copies = lm_head[copy_idx]
		#extended_head = torch.cat((lm_head, copies), 0)
		#model.lm_head.weight = nn.Parameter(extended_head)

elif args.model_lng == "de":
	model = AutoModelWithLMHead.from_pretrained("de_from_scratch/checkpoint-45000")
	embeddings = model.transformer.wte.weight
	embeddings_shortened = embeddings[:50257,:]
	model.transformer.wte.weight = nn.Parameter(embeddings_shortened)
	"""
	config = AutoConfig.from_pretrained(
	"gpt2",
	vocab_size=len(tokenizer),
	n_ctx=128,
	bos_token_id=tokenizer.bos_token_id,
	eos_token_id=tokenizer.eos_token_id,
	)
	model = GPT2LMHeadModel(config)
	"""
else:
	print("Illegal option.")

"""
# shuffle parts of WTE
embeddings = model.transformer.wte.weight
shuffle_len = round(embeddings.shape[0]*0.1)
shuffle_idx = torch.randperm(shuffle_len)
rest = embeddings[shuffle_len:]
shuffled = embeddings[shuffle_idx]
shuffled_partly = torch.cat((shuffled, rest),0)
model.transformer.wte.weight = nn.Parameter(shuffled_partly)
print(embeddings.shape == model.transformer.wte.weight.shape)

"""
# shuffle WTE
embeddings = model.transformer.wte.weight 
idx = torch.randperm(embeddings.shape[0])
shuffled = embeddings[idx]
model.transformer.wte.weight = nn.Parameter(shuffled)

if args.tied_weights:
	model.tie_weights()
else:
	
	# shuffle output embeddings
	lm_head = model.lm_head.weight
	idx = torch.randperm(lm_head.shape[0])
	shuffled = lm_head[idx]
	model.lm_head.weight = nn.Parameter(shuffled)
	
	print("Weights are NOT tied and output is shuffled")

print("TIED EMBEDDINGS?", model.transformer.wte.weight is model.lm_head.weight)	


def freeze_model(model):
	for name, param in model.named_parameters():
		param.requires_grad = False
		#print(name, param.shape)

# unfreeze all
def unfreeze_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = True

# freeze es model parameters except embeddings
freeze_model(model)
# or unfreeze model
#unfreeze_model(model)

model.transformer.wte.weight.requires_grad = True
model.transformer.wpe.weight.requires_grad = True
if not args.tied_weights:
	model.lm_head.weight.requires_grad = True

#model.lm_head.weight.requires_grad = False

print("ADAPTED PARAMETERS:")
for name, param in model.named_parameters():
	if param.requires_grad:
		print(name, param.shape)

def tokenize_function(examples):
	return tokenizer(examples["text"])

"""
def pos_function(dataset):
	pos = [" sein ", " seinen ", " seinem ", " seine ", " seines ", " seiner ", " ihr ", " ihren ", " ihrem ", " ihre ", " ihres ", " ihrer "]
	for example in dataset:
		count = 0
		for p in pos:
			count += example["text"].count(p)
		example["pos"] = count

pos_function(datasets["train"])
pos_function(datasets["validation"])
#datasets = datasets.map(pos_function,batched=True, num_proc=4)
"""
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
#tokenized_datasets = tok_datasets.map(pos_function, batched=True, num_proc=4)
"""
train_tok = train.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
val_tok = validation.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
"""
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

#test_seq = tokenizer.decode(lm_datasets["validation2"]["input_ids"][0])
#print("MODEL")
#print(model)

BAD_counter = 0
GOOD_counter = 0
indeces_train = []
indeces_val = []
for i, example in enumerate(lm_datasets["train"]["input_ids"]):
        if len(example) != 256:
                indeces_train.append(i)

for i, example in enumerate(lm_datasets["validation"]["input_ids"]):
        if len(example) != 256:
                indeces_val.append(i)


lm_datasets["train"] = lm_datasets["train"].select((i for i in range(len(lm_datasets["train"])) if i not in set(indeces_train)))
lm_datasets["validation"] = lm_datasets["validation"].select((i for i in range(len(lm_datasets["validation"])) if i not in set(indeces_val)))

for i, example in enumerate(lm_datasets["train"]["input_ids"]):
        if len(example) == 256:
                GOOD_counter += 1
        else:
                BAD_counter += 1
                print(i, len(example))


print(f"BAD!!!! {BAD_counter}")
print(f"GOOD!!! {GOOD_counter}")

from transformers import Trainer, TrainingArguments
from transformers.integrations import *

training_args = TrainingArguments(
	args.name,
	do_train=True,
	evaluation_strategy = "steps",
	learning_rate=args.lr,
	weight_decay=0.01,
	#num_train_epochs=1.0,
	max_steps=100000,
	eval_steps=5000,
	save_steps=5000,
	warmup_steps = 30000,
	seed=42
)


trainer = My_Trainer(
	model=model,
	args=training_args,
	train_dataset=lm_datasets["train"],
	eval_dataset={'wikipedia': lm_datasets["validation"]}
)
"""
model = AutoModelWithLMHead.from_pretrained("de_from_scratch/checkpoint-340000")

inputs = tokenizer(test_seq, return_tensors="pt")
labels = lm_datasets["validation2"]["input_ids"][0]
outputs = model(**inputs, labels=labels)
print(outputs)
"""

#print(model.transformer.wte.weight.shape)

trainer.train()
trainer.evaluate()