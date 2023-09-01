# get unigram probabilities from corpus
# imports
from transformers import AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

# load wikipedia test corpus
corpus = load_dataset("wikipedia", "20220301.de", split='train[95:100%]')
print(f"{corpus['text'][0]}")

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

# create (token, freq) pairs for plotting
tok_freq = defaultdict(int)
for example in tqdm(corpus['text']):
	tokenized = tokenizer.tokenize(example)
	for tok in tokenized:
		tok_freq[tok] += 1
print(f"Created a token/freq dictionary: {list(tok_freq.items())[:10]}")

# create unigram probabilities
tok_prob = defaultdict(float)
sum_toks = sum(tok_freq.values())
for tok in tok_freq:
	tok_prob[tok] = tok_freq[tok] / sum_toks

print(f"Unigram probabilities created. {list(tok_prob.items())[:10]}")

# sort frequencies

freq_sorted = sorted(list(tok_freq.items()), key=lambda x: x[1], reverse=True)

print(f"Total number of tokens: {len(freq_sorted)}")
print(f"Most frequent tokens: {freq_sorted[:10]}")

# write n tokens into a file
with open("tokens.txt", "w+", encoding="utf-8") as f:
	for i in range(1000):
		f.write(str(i)+"\t"+freq_sorted[i][0]+"\t"+str(freq_sorted[i][1])+"\n")
