# imports
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
import torch
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
import argparse

# argparse testfile and maybe models

parser = argparse.ArgumentParser()
parser.add_argument('--test_file')
parser.add_argument('--models', nargs='+')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

#checkpoints = np.arange(5000, 105000, 5000)
checkpoints = np.arange(5000, 20000, 5000)

# read file

def read_file(file):
	with open(file, "r", encoding='utf-8') as f:
		text = f.readlines()
	return text

# return path of model checkpoint

def get_chpt(model_type, chpt):
	print(f"Evaluating model type {model_type} on checkpoint {chpt}")
	return AutoModelForCausalLM.from_pretrained(model_type+"/checkpoint-"+str(chpt))

# eval on testsents function

def eval_on_testsents(model, test_file):
	poss_m = [" sein", " seinen", " seines", " seinem", " seiner", " seine"]
	poss_f = [" ihr", " ihren" , " ihres", " ihrem", " ihrer", " ihre"]
	poss = poss_m + poss_f
	corr = 0
	text = read_file(test_file)
	for line in text:
		label, test_sent = line.split("\t")
		tokenized = tokenizer(test_sent, return_tensors='pt')
		with torch.no_grad():
			outputs = model(**tokenized, labels=tokenized["input_ids"])
		logits = outputs.logits
		next_token_logits = logits[:, -1, :]
		probs = softmax(next_token_logits)[0]
		poss_probs = [probs[tokenizer(word)["input_ids"]] for word in poss]
		best_pred = poss[np.argmax(poss_probs)]
		if label == "m":
			if best_pred in poss_m:
				corr += 1
		elif label == "f":
			if best_pred in poss_f:
				corr += 1
		else:
			print("Unexpected label. Check test file.")
			break

	total = len(text)
	corr_total = (corr / total) * 100
	return corr_total
		

# iterate over all checkpoints

def eval_on_checkpoints(model_type):
	datapoints = []
	for chpt in checkpoints:
		model = get_chpt(model_type, chpt)
		corr_chpt_x = eval_on_testsents(model, args.test_file)
		datapoints.append(corr_chpt_x)
	return datapoints

# plot function; plots all checkpoints of x models into one graph
# bar plots for each checkpoints in diff colors depending on model type

def plot_checkpoints():
	n_groups = len(args.models)
	index = np.arange(n_groups)
	rects = []
	width = 0

	fig = plt.figure()
	ax = fig.add_subplot(111)

	colors = ['r', 'g', 'b']
	colors = colors[:n_groups]

	for model_type in args.models:
		datapoints = eval_on_checkpoints(model_type)
		rects.append(ax.bar(index+width, datapoints, width, color=colors.pop()))
		width += 0.27

	ax.set_ylabel('Percentage of correct genus prediction')
	ax.set_xticks(index+0.27)
	ax.set_xticklabels(checkpoints)
	ax.legend(tuple([rec[0] for rec in rects]), (tuple(args.models)))
	
	def autolabel(rs):
		for rect in rs:
			h = rect.get_height()
			ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h), ha='center', va='bottom')
	for rect in rects:
		autolabel(rect)

	plt.show()

plot_checkpoints()
