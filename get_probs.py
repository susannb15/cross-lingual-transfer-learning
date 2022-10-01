# imports
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
import torch
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from collections import defaultdict

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
native_model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")

#tokenizer = AutoTokenizer.from_pretrained("datificate/gpt2-small-spanish")
de_model = AutoModelForCausalLM.from_pretrained("de_untied_wiki20wtewpe/checkpoint-100000")
from_scratch_model = AutoModelForCausalLM.from_pretrained("de_from_scratch_adapted_wte_wpe_untied/checkpoint-100000")
from_scratch_model_wte = AutoModelForCausalLM.from_pretrained("de_from_scratch_adapted_wte_untied/checkpoint-100000")
from_scratch_model_traindata = AutoModelForCausalLM.from_pretrained("de_from_scratch_adapted_on_train_data/checkpoint-100000")
from_scratch_model_no_adaptation = AutoModelForCausalLM.from_pretrained("de_from_scratch/checkpoint-230000")

# input 
# TODO: make this loadable from command line
input = "testsents.txt"

def get_all_preds(input):
	tokenized = tokenizer(input, return_tensors="pt")
	with torch.no_grad():
		outputs = model(**tokenized, labels=tokenized["input_ids"])
	logits = outputs.logits
	next_token_logits = logits[:, -1, :]
	probs = softmax(next_token_logits)
	print("INPUT:\t", input)
	probs = probs[0]
	preds = []
	for id in range(next_token_logits.size(dim=1)):
		pred_word = tokenizer.decode(id)
		preds.append((pred_word, probs[id]))
	preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
	return preds_sorted

def predict(input, word_list):
	tokenized = tokenizer(input, return_tensors="pt")
	with torch.no_grad():
		outputs = model(**tokenized, labels=tokenized["input_ids"])
	logits = outputs.logits
	next_token_logits = logits[:, -1, :]
	probs = softmax(next_token_logits)
	probs = probs[0]
	print("INPUT:\t", input)
	#print("WORDS OF INTEREST:\t", [word for word in word_list])
	prob_list = []
	for word in word_list:
		#print(word)
		ids = tokenizer(word)["input_ids"]
		print("TEST:\t", tokenizer.convert_ids_to_tokens(ids))
		#print("IDs", ids)
		#prob = 0
		prob = probs[ids]
		#print(ids, prob)
		#for id in ids:
			#print("ID", id, probs[id])
		#	prob += probs[id]
		prob_list.append(prob)
	return prob_list

def plot_preds(cons):
	poss_m = [" sein", " seinen", " seines", " seinem", " seiner", " seine"]
	poss_f = [" ihr", " ihren" , " ihres", " ihrem", " ihrer", " ihre"]
	poss = poss_m + poss_f
	names = ["native_model", "from_scratch", "from_scratch_adapted_wte+wpe", "from_scratch_adapted_wte", "from_scratch_adaptated_on_train_data", "DE_adapted_wte+wpe"]
	#names = ["native_model", "de_model"]
	models = [native_model, from_scratch_model_no_adaptation, from_scratch_model, from_scratch_model_wte, from_scratch_model_traindata, de_model]
	#models = [native_model, de_model]
	y_corr = defaultdict(list)
	y_incorr = defaultdict(list)
	y_probs = defaultdict(list)
	with open("detailed.txt", "w+", encoding="utf-8") as f:
		for model, name in zip (models, names):
			f.write("DETAILS FOR:\t"+name+"\n")
			for con in cons:
				f.write("CONDITION:\t"+con+"\n")
				text = read_file(con+".txt")
				corr = 0
				incorr = 0
				for line in text:
					label, text = line.split("\t")
					tokenized = tokenizer(text, return_tensors="pt")
					with torch.no_grad():
						outputs = model(**tokenized, labels=tokenized["input_ids"])
					logits = outputs.logits
					next_token_logits = logits[:, -1, :]
					probs = softmax(next_token_logits)
					probs = probs[0]
					prob_list = []
					for word in poss:
						ids = tokenizer(word)["input_ids"]
						prob = probs[ids]
						prob_list.append(prob)
					best_pred = poss[np.argmax(prob_list)]
					norm = sum(prob_list)
					if label == "m":
						y_probs[name].extend((prob_list[0]+prob_list[1]) / norm) 
						if best_pred in poss_m:
							corr += 1
						else:
							incorr += 1
							f.write("INCORRECT PRED AT:\t"+line+"\n")
					elif label == "f":
						y_probs[name].extend((prob_list[2]+prob_list[3]) / norm) 
						if best_pred in poss_f:
							corr += 1
						else:
							incorr += 1
							f.write("INCORRECT PRED AT:\t"+line+"\n")
					else:
						print("UNKNOWN LABEL!")
					#print(label, best_pred)
				total = corr + incorr
				corr = (corr / total) * 100
				incorr = (incorr / total) * 100
				print(name+"\n", "CORRECT", corr, "\tINCORRECT", incorr)
				f.write("CORRECT\t"+str(corr)+"\tINCORRECT\t"+str(incorr)+"\n")
				y_corr[name].append(corr)
				y_incorr[name].append(incorr)
			x_axis = np.arange(len(cons))
			plt.bar(x_axis - 0.2, y_corr[name], 0.4, label='CORRECT')
			plt.bar(x_axis + 0.2, y_incorr[name], 0.4, label='INCORRECT')
			plt.xticks(x_axis, cons)
			plt.xlabel("Conditions")
			plt.ylabel("Number of correct/incorrect predictions in %")
			plt.title(name+" predictions of possessive pronouns")
			plt.legend()
			plt.savefig(name+".png")
			plt.show()
		labels, data = [*zip(*y_probs.items())]
		plt.boxplot(data)
		plt.xticks(range(1, len(labels) + 1), labels)
		plt.savefig("boxplot.png")
		plt.show()	

def read_file(file):
	with open(file, "r", encoding="utf-8") as f:
		text = f.readlines()
	return text

#plot_preds(["simple", "adv", "genitiv", "nebensatz"])
plot_preds(["testsents"])

"""
text = read_file(input)
word_list = [" seinen", " seine", " ihren", " ihre"]
with open("out.txt", "w+", encoding="utf-8") as f:
	for line in text:
		f.write("INPUT:\t"+line+"\n")
		prob_list = predict(line, word_list)
		for i in range(len(prob_list)):
			print("WORD:\t", word_list[i])
			print("PROB:\t", prob_list[i] / sum(prob_list))
			f.write("WORD:\t"+word_list[i]+"\n")
			#f.write("PROB:\t"+str(prob_list[i] / sum(prob_list))+"\n") 
			f.write("PROB:\t"+str(prob_list[i])+"\n")
		f.write("\n")
		print("Top 10 Predictions:")
		preds = get_all_preds(line)
		print(preds[:10])
"""
