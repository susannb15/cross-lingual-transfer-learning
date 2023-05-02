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
import matplotlib as mpl

np.random.seed(42)

# load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
german_native =  AutoModelWithLMHead.from_pretrained("de_from_scratch/checkpoint-45000")
german_adapted = AutoModelWithLMHead.from_pretrained("de_tied/checkpoint-45000")
spanish_adapted = AutoModelWithLMHead.from_pretrained("es_tied/checkpoint-45000")
english_adapted = AutoModelWithLMHead.from_pretrained("en_tied/checkpoint-45000")

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


word_counter = 0
tag_counter = 0

for sentence in sentences:
	words = [word.form for word in sentence]
	tags = [word.upos for word in sentence] 
	word_list.append(words)
	tag_list.append(tags)
	word_counter += len(words)
	tag_counter += len(tags)

assert len(word_list) == len(tag_list)

print(f"Extracted {word_counter} words and {tag_counter} tags from {len(word_list)} sentences.")
print(word_list[0][:5])
print(tag_list[0][:5])

# tokenize each word with the DE tokenizer 

word_copy = []
tag_copy = []

tokenized = []
inputs = []
for depth, sent in enumerate(word_list):
	toks = []
	words = []
	tags = []
	sent_tok = " ".join(sent)
	inputs.append(tokenizer(sent_tok, return_tensors='pt'))
	for index, word in enumerate(sent):
		tok = tokenizer(word)["input_ids"]
		# if a word is split bc of tokenizer we need to copy its pos tag
		for ids in tok:
			curr_tag = tag_list[depth][index] # get tag of current word
			if curr_tag is not None:
				words.append(word)
				tags.append(curr_tag) 
				toks.append(ids)
	tokenized.append(toks)
	word_copy.append(words)
	tag_copy.append(tags)

print("Tokenized successfully.")
print(tokenized[0][:10])
print(word_copy[0][:10])
print(tag_copy[0][:10])

print(f"Sizes after tokenization: {sum(len(x) for x in word_copy)} words, {sum(len(x) for x in tag_copy)} tags and {sum(len(x) for x in tokenized)} tokens.")

def tsne(model_name, model):

	# get embeddings from model
	embeddings = []
	for sent in tokenized:
		#with torch.no_grad():
		#	outputs = german_native(**sent, labels=sent["input_ids"])
		#embs_german_native = []
		for tok in sent:
			embedding = model.transformer.wte.weight[tok].detach().numpy().tolist()
			embeddings.append(embedding)

	embeddings = np.array(embeddings)
	print(f"Shape of the embedding array: {embeddings}")

	# TSNE
	tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embeddings)

	# plot 

	label_dict = dict()

	count = 0
	for sent in tag_copy:
		for tag in sent:
			if tag not in label_dict:
				label_dict[tag] = count
				count += 1

	labels = []

	for sent in tag_copy:
		for tag in sent:
			labels.append(label_dict[tag])
	N = len(label_dict.keys())

	print(f"{N} labels: {label_dict.keys()}")

	cmap = plt.cm.jet
	cmaplist = [cmap(i) for i in range(cmap.N)]
	cmap = cmap.from_list('Costum cmap', cmaplist, cmap.N)

	xs = embeddings[:,0]
	ys = embeddings[:,1]

	fig, ax = plt.subplots(figsize=(20,10))

	bounds = np.linspace(0,N,N+1)
	norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

	scatter = plt.scatter(xs, ys, c=labels, cmap=cmap, norm=norm)

	cb = plt.colorbar(scatter, spacing='proportional', ticks=bounds)

	for j, label in enumerate(label_dict.keys()):
		cb.ax.text(.5, (8 * j + 3) / 8.0, label, ha='center', va='center')
	cb.set_label('POS Tags')

	plt.legend()

	word_list_flat = []
	for sent in word_copy:
		for word in sent:
			word_list_flat.append(word)

	print(f"Length of flat word list: {len(word_list_flat)}")

	for i, txt in enumerate(word_list_flat):
		ax.annotate(txt, (xs[i], ys[i]), fontsize=8)

	plt.title(f"TSNE embedding plot of {model_name}  model")
	fig_name = "tsne_UD_"+model_name+".png"
	plt.savefig(fig_name)
	plt.show()

tsne("german_native", german_native) 
tsne("german_adapted", german_adapted)
tsne("spanish_adapted", spanish_adapted)
tsne("english_adapted", english_adapted)

