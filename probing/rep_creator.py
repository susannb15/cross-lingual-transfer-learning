#imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os

class RepCreator:
	def __init__(self,tokenizer, model, output_dir):
		# load sentences
		with open("probing_sents.txt", "r") as f:
			self.sentences = f.readlines()
		with open("probing_sents_test.txt", "r") as f:
			self.test_sentences = f.readlines()
		# load model and tokenizer
		self.tokenizer = tokenizer
		self.model = model
		self.tok2id, self.id2tok = self._create_dicts()
		self.reps = dict()
		self.output_dir = output_dir
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

	def _create_dicts(self):
		# create id - token dicts
		tok2id = dict()
		id2tok = dict()
		with open("tokens_for_probing.txt", "r") as f:
			tokens = f.readlines()
		for line in tokens:
			tokenized = self.tokenizer(line)["input_ids"]
			und, token = line.split()
			tok_id = tokenized[1]
			tok2id[token] = tok_id
			id2tok[tok_id] = token
		return tok2id, id2tok

	def _model_rep_train(self):
		# get representations from model
		with open("probing.meta", "w+", encoding='utf-8') as f:
			f.write("sentence number,token,token index in sentence")
		for token_num, token in enumerate(tqdm(list(self.tok2id.keys()))):
			for sent_num, sent in enumerate(self.sentences):
				while sent_num in np.arange(token_num*20,(token_num+1)*20,1):
					tokenized = self.tokenizer(sent, return_tensors="pt")
					with torch.no_grad():
						outputs = self.model(**tokenized, labels=tokenized["input_ids"], output_hidden_states=True, return_dict=True)
					hidden_states = outputs['hidden_states'] # [1,seq_len,hidden_size]
					tokenized_list = tokenized["input_ids"].tolist()[0]
					with open("probing.meta", "a", encoding='utf-8') as f:
						try:
							ids = self.tok2id[token]
							idx = tokenized_list.index(ids)
							for i, layer in enumerate(hidden_states):
								# extract representation
								rep = layer[:,idx,:]
								try:
									self.reps[i][sent_num] = rep
								except KeyError:
									self.reps[i] = torch.empty(size=(693, 768)) # 1809
									self.reps[i][sent_num] = rep
							f.write("\n"+str(sent_num)+","+self.id2tok[ids]+","+str(idx))
						except:
							print(token, sent)
							break
					break
	def _model_rep_test(self):
		# test senteces
		for token_num, token in enumerate(tqdm(list(self.tok2id.keys()))):
			sent = self.test_sentences[token_num]
			tokenized = self.tokenizer(sent, return_tensors="pt")
			with torch.no_grad():
				outputs = self.model(**tokenized, labels=tokenized["input_ids"], output_hidden_states=True, return_dict=True)
			hidden_states = outputs['hidden_states'] # [1,seq_len,hidden_size]
			tokenized_list = tokenized["input_ids"].tolist()[0]
			with open("probing.meta", "a", encoding='utf-8') as f:
				try:
					ids = self.tok2id[token]
					idx = tokenized_list.index(ids)
					for i, layer in enumerate(hidden_states):
						# extract representation
						rep = layer[:,idx,:]
						self.reps[i][660+token_num] = rep
					f.write("\n"+str(660+token_num)+","+self.id2tok[ids]+","+str(idx))
				except:
					print(token, sent)
					break

	def _write_reps_to_file(self):
		# save representation
		for layer in self.reps:
			path = os.path.join(self.output_dir, 'representations_'+str(layer))
			torch.save(self.reps[layer], path)

