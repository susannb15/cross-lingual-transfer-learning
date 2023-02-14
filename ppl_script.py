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
from scipy.special import softmax

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

test_logits = {0: [5, 8], 1: [2,5], 2: [3], 3: [3, 12], 4: [8], 5: [3], 6: [3]} 
test_logits_words = ["... nach seiner gelungenen >>Täter(suche)<<", "... sogleich >>befördert<<", "Diese Familien >>konzentrierten<<", "konzentrierten sich auf >>Einfluss<<", "... Rest ist >>Kohlendioxid<<", "... macht sie >>ris(iko)<<", "... auch >>Chancen<<", "... gegen Verlust >>versichert<<", "... stammt der >>Kommand(it)<<", "... begannen >>1793<<"]


for i, sentence in enumerate(content):
	words = sentence.split()
	with torch.no_grad():
		tokenized = tokenizer(sentence, return_tensors='pt')
		labels = tokenized["input_ids"]
		outputs = model(**tokenized, labels=labels)
		logits = outputs.get("logits")
		for j in test_logits[i]:
			token_logits = logits[:, j-1, :]
			probs = softmax(token_logits)[0]
			prob_j = probs[labels.tolist()[0][j]].item()
			preds = []
			for k in range(token_logits.size(dim=1)):
				pred_word = tokenizer.decode(k)
				preds.append((pred_word, probs[k]))
			preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
			best_preds = preds_sorted[:15]
			words = [k[0] for k in best_preds]
			probs = [k[1].item() for k in best_preds]
			title = test_logits_words.pop(0)
			title += f"({str(prob_j)})"
			plt.title(title)
			plt.plot(words, probs)
			plt.show()
			
"""

concat = ""


for sentence in content:
	words = sentence.split()
	#label, sent = sentence.split("\t")
	#sentence = sent
	concat += sentence

# shift everything 1 tab forward!

with torch.no_grad():
	words = concat.split() # del
	tokenized = tokenizer(concat, return_tensors='pt') #sentence
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
words1 = words[:(len(words)//2)]
words2 = words[(len(words)//2):]
ppls1 = ppls[:(len(words)//2)]
ppls2 = ppls[:(len(words)//2)]
#plt.plot(words, ppls)
plt.title("PPL per word for: concat") # f{sentence}
plt.plot(words1, ppls1)
plt.show()
plt.plot(words2, ppls2)
plt.show()
"""
