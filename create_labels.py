#imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# load sentences
with open("probing_sents.txt", "r") as f:
	sentences = f.readlines()

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model = AutoModelForCausalLM.from_pretrained("native_models/native_de/checkpoint-45000")

# create id - token dict
tok2id = dict()
id2tok = dict()
with open("tokens_for_probing.txt", "r") as f:
	tokens = f.readlines()
	for token in tokens:
		token = token.strip()
		tok_id = frozenset(tokenizer(token)["input_ids"])
		tok2id[token] = tok_id
		id2tok[tok_id] = token
print(tok2id.items())

# get representations from model
for sent in sentences:
	tokenized = tokenizer(sent, return_tensors="pt")
	with torch.no_grad():
		outputs = model(**tokenized, labels=tokenized["input_ids"], return_representations=True)

# save representation - word file
