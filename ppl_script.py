# A script for ppl output per word in a sentence

import transformers
from datasets import load_dataset
import torch
import torch.nn as nn
import argparse
import math
from trainer_mod import My_Trainer
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel
import matplotlib.pyplot as plt

random.seed(42)

parser = argparse.ArgumentParser(description="Options")

parser.add_argument('--data', help='Dataset to be tested. Should contain sample sentences')
parser.add_argument('--model', default='from_scratch', help='Model to be tested. Specify: pre-trained, from_scratch, de, en, es')

args = parser.parse_args()

# load model and tokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

if args.model == "pre-trained":
	model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")
elif args.model == "de":
	model = AutoModelWithLMHead.from_pretrained("de_tied/checkpoint-60000")
elif args.model == "en":
	model = AutoModelWithLMHead.from_pretrained("en_tied/checkpoint-60000")
elif args.model == "es":
	model = AutoModelWithLMHead.from_pretrained("es_tied/checkpoint-60000")
else:
	model = AutoModelWithLMHead.from_pretrained("de_from_scratch/checkpoint-45000")
	print("No model specified or unknown model name. Taking DE model trained from scratch")

def read_file(file_name):
	with open(file_name, "r") as f:
		text = f.readlines()
	return text

content = read_file(args.data)

for sentence in content:
	words = sentence.split()
	label, sent = sentence.split("\t")
	sentence = sent
	with torch.no_grad():
		tokenized = tokenizer(sentence, return_tensors='pt')
		labels = tokenized["input_ids"]
		#print("LABELS", labels)
		outputs = model(**tokenized, labels=labels)
		logits = outputs.get("logits") 
		print("PPL overall: ", math.exp(outputs.loss))
		shift_logits = logits[..., :-1, :].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		#print(shift_logits.shape, shift_labels)
		loss_fct = nn.CrossEntropyLoss(reduction="none")
		loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
		#print("SHIFT", shift_labels.view(-1))
	ppls = [math.exp(l) for l in loss] 
	#print("LOSS", len(loss))
	token_list = tokenized["input_ids"].flatten().tolist()
	words = [tokenizer.decode(word) for word in token_list]
	del words[0]
	#print("WORDS", words)
	plt.plot(words, ppls)
	plt.title(f"PPL per word for: {sentence}")
	#plt.show()
