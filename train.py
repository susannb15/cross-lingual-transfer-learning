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


parser = argparse.ArgumentParser(description="Options")

parser.add_argument('--model', type=str, required=True, 
					help="Model. Options: de, es")

args = parser.parse_args()

datasets = DatasetDict()
train = load_dataset("wikipedia", "20220301.de", split='train[:10%]')
validation = load_dataset("wikipedia", "20220301.de", split='train[11:13%]')
datasets["train"] = train
datasets["validation"] = validation

# add new column to store pos values 
datasets["train"] = datasets["train"].add_column("pos", [[0]] * len(datasets["train"]))
datasets["validation"] = datasets["validation"].add_column("pos", [[0]] * len(datasets["validation"]))

print(datasets["train"].features)

"""
train = load_dataset("wikipedia", "20220301.de", split='train[:3%]')
validation = load_dataset("wikipedia", "20220301.de", split='train[4:5%]')
"""
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

#print(show_random_elements(datasets["train"]))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead

de_model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")

if args.model == "es":
	#tokenizer = AutoTokenizer.from_pretrained("datificate/gpt2-small-spanish")
	tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
	model = AutoModelForCausalLM.from_pretrained("datificate/gpt2-small-spanish")
	embeddings = model.transformer.wte.weight
	perm = torch.randperm(embeddings.shape[0])
	copy_idx = perm[:8]
	copies = embeddings[copy_idx]
	extended_emb = torch.cat((embeddings, copies), 0)
	idx = torch.randperm(extended_emb.shape[0])
	shuffled = extended_emb[idx]
	model.transformer.wte.weight = nn.Parameter(shuffled)

	#lm_head = model.transformer.wte.weight
	model.tie_weights()
	print("TIED EMBEDDINGS?", hasattr(model, "tie_weights"), model.transformer.wte.weight is model.lm_head.weight)
	"""
	lm_head = model.lm_head.weight
	perm = torch.randperm(lm_head.shape[0])
	copy_idx = perm[:8]
	copies = lm_head[copy_idx]
	extended_head = torch.cat((lm_head, copies), 0)
	idx = torch.randperm(extended_head.shape[0])
	shuffled = extended_head[idx]
	model.lm_head.weight = nn.Parameter(shuffled)
	"""
elif args.model == "de":
	tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
	model = de_model
	de_embeddings = de_model.transformer.wte.weight # DE word token embeddings
	idx = torch.randperm(de_embeddings.shape[0])
	de_emb_shuffled = de_embeddings[idx]
	model.transformer.wte.weight = nn.Parameter(de_emb_shuffled)

	model.tie_weights()
	"""
	lm_head = model.lm_head.weight
	idx = torch.randperm(lm_head.shape[0])
	shuffled = lm_head[idx]
	model.lm_head.weight = nn.Parameter(shuffled)
	"""
	#print("TIED EMBEDDINGS?", hasattr(model, "tie_weights"), model.transformer.wte.weight is model.lm_head.weight)
	

else:
	print("Illegal model name. Choose between: es, de")

#print("EMB SHAPE")
#print(de_emb_shuffled.shape)
#de_emb_shuffled = de_emb_shuffled[:50257]
#print(de_emb_shuffled.shape)

# replace es embeddings with the shuffled de embeddings
#print(model.transformer.wte.weight.shape)
#print(model.lm_head.weight.shape)

def freeze_model(model):
	for name, param in model.named_parameters():
		param.requires_grad = False
		#print(name, param.shape)

# freeze es model parameters except embeddings
freeze_model(model)
model.transformer.wte.weight.requires_grad = True
model.lm_head.weight.requires_grad = True
#print("test freeze")
#for name, param in model.named_parameters():
    #if param.requires_grad:
        #print(name)

def tokenize_function(examples):
	return tokenizer(examples["text"])

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
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
#tokenized_datasets = tok_datasets.map(pos_function, batched=True, num_proc=4)
print(tokenized_datasets["train"][1])
print(tokenized_datasets["train"].features)
"""
train_tok = train.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
val_tok = validation.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
"""
block_size=128

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


#lm_train = train_tok.map(group_texts, batched=True, batch_size=1000, num_proc=4)
#lm_val = val_tok.map(group_texts, batched=True, batch_size=1000, num_proc=4)

#print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))



pos_1 = lm_datasets["train"].filter(lambda example: example["pos"] != 0)
print(tokenizer.decode(pos_1[1]["input_ids"]))

from transformers import Trainer, TrainingArguments
from transformers.integrations import *

training_args = TrainingArguments(
    args.model+"_model",
	do_train=True,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
	num_train_epochs=25,
	eval_steps=1000,
	save_steps=1000,
	seed=42
)


trainer = My_Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)


trainer.train()
trainer.evaluate()
