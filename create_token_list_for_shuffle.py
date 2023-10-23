# imports
import transformers
from datasets import load_dataset
from datasets import DatasetDict
from datasets import ClassLabel, Value
from itertools import chain
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel, GPT2Tokenizer
from collections import Counter

# build a list of the n most frequent token (ids) in the wiki corpus

# load corpus
data = load_dataset("wikipedia", "20220301.de", split='train[70:90%]') #70:90%
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

# tokenize corpus

def tokenize_function(examples):
	return tokenizer(examples["text"])

tokenized = data.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

counter_all = Counter()
# build token/id/count dict
for example in tokenized["input_ids"]:
	counts = Counter(example)
	counter_all.update(counts)

# pick the top n tokens 
sorted_tokens = sorted(counter_all.items(), key=lambda x: x[1], reverse=True)

# for mid tokens:
sorted_tokens = sorted_tokens[5000:]

# write the tokens to output file
with open('shuffle_tokens_medium50.txt', 'w+', encoding='utf-8') as f:
	for i in range(50):	
		f.write(str(sorted_tokens[i][0])+'\n')
