#imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# load sentences
with open("probing_sents.txt", "r") as f:
	sentences = f.readlines()

with open("probing_sents_test.txt", "r") as f:
	test_sentences = f.readlines()

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model = AutoModelForCausalLM.from_pretrained("native_models/native_de/checkpoint-45000")

# create id - token dict
tok2id = dict()
id2tok = dict()
with open("tokens_for_probing.txt", "r") as f:
	tokens = f.readlines()
	for line in tokens:
		tokenized = tokenizer(line)["input_ids"]
		und, token = line.split()
		tok_id = tokenized[1]
		tok2id[token] = tok_id
		id2tok[tok_id] = token

# get representations from model
with open("probing.meta", "w+", encoding='utf-8') as f:
	f.write("sentence number,token,token index in sentence")
reps = dict()
for token_num, token in enumerate(tqdm(list(tok2id.keys()))):
	for sent_num, sent in enumerate(sentences):
		while sent_num in np.arange(token_num*20,(token_num+1)*20,1):
			tokenized = tokenizer(sent, return_tensors="pt")
			with torch.no_grad():
				outputs = model(**tokenized, labels=tokenized["input_ids"], output_hidden_states=True, return_dict=True)
			hidden_states = outputs['hidden_states'] # [1,seq_len,hidden_size]
			tokenized_list = tokenized["input_ids"].tolist()[0]
			with open("probing.meta", "a", encoding='utf-8') as f:
				try:
					ids = tok2id[token]
					idx = tokenized_list.index(ids)
					for i, layer in enumerate(hidden_states):
						# extract representation
						rep = layer[:,idx,:]
						try:
							reps[i][sent_num] = rep
						except KeyError:
							reps[i] = torch.empty(size=(693, 768)) # 1809
							reps[i][sent_num] = rep
					f.write("\n"+str(sent_num)+","+id2tok[ids]+","+str(idx))
				except:
					print(token, sent)
					break
			break

# test senteces
for token_num, token in enumerate(tqdm(list(tok2id.keys()))):
	sent = test_sentences[token_num]
	tokenized = tokenizer(sent, return_tensors="pt")
	with torch.no_grad():
		outputs = model(**tokenized, labels=tokenized["input_ids"], output_hidden_states=True, return_dict=True)
	hidden_states = outputs['hidden_states'] # [1,seq_len,hidden_size]
	tokenized_list = tokenized["input_ids"].tolist()[0]
	with open("probing.meta", "a", encoding='utf-8') as f:
		try:
			ids = tok2id[token]
			idx = tokenized_list.index(ids)
			for i, layer in enumerate(hidden_states):
				# extract representation
				rep = layer[:,idx,:]
				reps[i][660+token_num] = rep
			f.write("\n"+str(660+token_num)+","+id2tok[ids]+","+str(idx))
		except:
			print(token, sent)
			break

# save representation
for layer in reps:
	torch.save(reps[layer], 'representations_'+str(layer))

