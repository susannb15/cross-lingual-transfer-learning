# imports
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import torch
import matplotlib.cm as cm
import pyconll
import copy

np.random.seed(42)

# load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
german_native =  AutoModelWithLMHead.from_pretrained("de_from_scratch/checkpoint-45000")

# load dataset
file = pyconll.load_from_file("/local/susannb/UD_German-PUD/de_pud-ud-test.conllu")
print("File loaded successfully.")
for sentence in file:
	print(sentence.text)
	pos_list = [token.upos for token in sentence]
	print(pos_list[:5], "...")
	break

# take (10 for testing) 100 random sents 
indeces = np.random.choice(len(file), 10).tolist()
sentences = [file[index] for index in indeces]
print(f"Picked {len(indeces)} random sentences.")

word_list = []
tag_list = []

for sentence in sentences:
	words = [word.form for word in sentence]
	tags = [word.upos for word in sentence] 
	word_list.append(words)
	tag_list.append(tags)

assert len(word_list) == len(tag_list)

print(f"Extracted words and tags from {len(word_list)} sentences.")
print(word_list[0][:5])
print(tag_list[0][:5])

# tokenize each word with the DE tokenizer 

# deepcopy word and tag lists for inserting and iterating
word_copy = copy.deepcopy(word_list)
tag_copy = copy.deepcopy(tag_list)

tokenized = []
inputs = []
for depth, sent in enumerate(word_list):
	toks = []
	count = 0 # increase for correct indexing
	inputs.append(tokenizer(sent, return_tensors='pt'))
	for index, word in enumerate(sent):
		tok = tokenizer(word)["input_ids"]
		toks.extend(tok)
		# if a word is split bc of tokenizer we need to copy its pos tag
		if len(tok) > 1:
			for ids in tok:
				word_copy[depth].insert(index+count,word)
				curr_tag = tag_list[depth][index] # get tag of current word
				tag_copy[depth].insert(index+count,curr_tag) # copy tag	
				count += 1
	tokenized.append(toks)

print("Tokenized successfully.")
print(tokenized[0][:10])
print(word_copy[0][:10])
print(tag_copy[0][:10])

# TSNE
for sent in inputs:
	with torch.no_grad():
		outputs = german_native(**sent, labels=sent["input_ids"])
		print(outputs)
	break
#tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embeddings)

# plot 

# pos tags = color coding for the clusters
# plot for each model
# joint plots
