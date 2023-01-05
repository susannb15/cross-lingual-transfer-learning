# imports
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
import torch
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from collections import defaultdict

# argparse testfile and maybe models

parser = argparse.ArgumentParser()
parser.add_argument('--test_file')
parser.add_argument('--models', nargs='+')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

checkpoints = np.arange(5000, 55000, 5000)
#checkpoints = np.arange(5000, 20000, 5000)

# read file

def read_file(file):
	with open(file, "r", encoding='utf-8') as f:
		text = f.readlines()
	return text

# return path of model checkpoint

def get_chpt(model_type, chpt):
	print(f"Evaluating model type {model_type} on checkpoint {chpt}")
	return model_type+"/checkpoint-"+str(chpt)

# autolabel

def autolabel(rs, ax):
	for rect in rs:
		h = rect.get_height()
		ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h), ha='center', va='bottom')


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
		model = AutoModelForCausalLM.from_pretrained(get_chpt(model_type, chpt))
		corr_chpt_x = eval_on_testsents(model, args.test_file)
		datapoints.append(corr_chpt_x)
	return datapoints

# get ppl of checkpoints

def get_ppl_on_checkpoints(model_type, dataset):
	ppl = []
	path = get_chpt(model_type, checkpoints[-1])	
	with open(path+'/trainer_state.json', 'r') as f:
		state = json.load(f)
		for item in state['log_history']:
			if 'eval-'+dataset+'_perplexity' in item:
				ppl.append(item['eval-'+dataset+'_perplexity'])
	return ppl

# plot function; plots all checkpoints of x models into one graph
# bar plots for each checkpoints in diff colors depending on model type

def plot_testsents():
	n_groups = len(checkpoints)
	index = np.arange(n_groups)
	rects = []
	width = 0

	fig = plt.figure()
	ax = fig.add_subplot(111)

	colors = ['r', 'g', 'b']

	for model_type in args.models:
		datapoints = eval_on_checkpoints(model_type)
		color = colors.pop()
		#rects.append(ax.bar(index+width, datapoints, width=0.27, color=color))
		ax.plot(index, datapoints, color=color)
		width += 0.27

	ax.set_ylabel('Percentage of correct genus prediction')
	ax.set_xticks(index)
	ax.set_xticklabels(checkpoints)
	#ax.legend(tuple([rec[0] for rec in rects]), (tuple(args.models)))
	ax.legend(tuple(args.models))
	
	#for rect in rects:
	#	autolabel(rect, ax)

	plt.savefig("checkpoints_testsents.png")

	plt.show()

def plot_ppl():
	n_groups = len(checkpoints)
	index = np.arange(n_groups)
	datasets = defaultdict(lambda x: defaultdict(x))
	datasets["wikipedia"] = defaultdict(lambda x: defaultdict(x))
	datasets["tiger"] = defaultdict(lambda x: defaultdict(x))
	datasets["10kGNAD"] = defaultdict(lambda x: defaultdict(x))
	datasets["wikipedia"]["rects"] = list()
	datasets["tiger"]["rects"] = list()
	datasets["10kGNAD"]["rects"] = list()

	fig = plt.figure()
	ax_wiki = fig.add_subplot(221)
	ax_tiger = fig.add_subplot(222)
	ax_10kGNAD = fig.add_subplot(223)

	datasets["wikipedia"]["plot"] = ax_wiki
	datasets["tiger"]["plot"] = ax_tiger
	datasets["10kGNAD"]["plot"] = ax_10kGNAD

	for dataset in datasets:
		colors = ['r', 'g', 'b']
		width = 0
		for model_type in args.models:
			datapoint = get_ppl_on_checkpoints(model_type, dataset)
			color = colors.pop()
			#datasets[dataset]["rects"].append(datasets[dataset]["plot"].bar(index+width, datapoint, width=0.27, color=color))
			datasets[dataset]["plot"].plot(index, datapoint, color=color)
			width += 0.27

		for dataset in datasets:
			datasets[dataset]["plot"].set_ylabel(f'PPL on {dataset} testset')
			datasets[dataset]["plot"].set_xticks(index)
			datasets[dataset]["plot"].set_xticklabels(checkpoints)
			#datasets[dataset]["plot"].legend(tuple([rec[0] for rec in datasets[dataset]["rects"]]), (tuple(args.models)))
			datasets[dataset]["plot"].legend(tuple(args.models))

			#for rect in datasets[dataset]["rects"]:
			#	autolabel(rect, datasets[dataset]["plot"])
	
	plt.yscale('log')
	plt.savefig("checkpoints_ppl.png")
	plt.show()

plot_testsents()
plot_ppl()
