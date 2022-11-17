from datasets import load_dataset
from collections import Counter
import itertools
from tqdm import tqdm
import pandas as pd

# load corpora
datasets = load_dataset("wikipedia", "20220301.de", split='train[90:95%]')
wikipedia = []

for article in tqdm(datasets["text"]):
        sents = article.split(".")
        wikipedia.extend(sents)

news = []

df = pd.read_csv("10kGNAD/articles.csv", delimiter="\t")
for article in tqdm(df["text"]):
        sents = article.split(".")
        news.extend(sents)

with open("tiger.txt", "r", encoding='ISO-8859-1') as f:
        tiger = f.readlines()


def avg_sen_len(corpus):
	counter = Counter()
	num_words = 0
	sent_len_min = 1000
	sent_len_max = 0
	for sen in corpus:
		words = sen.split(" ")
		sent_len = len(words)
		if sent_len < sent_len_min:
			sent_len_min = sent_len
		if sent_len > sent_len_max:
			sent_len_max = sent_len
		c = Counter(words)
		counter.update(c)
		num_words += sent_len
	avg_len = num_words / len(corpus)
	stats = (avg_len, sent_len_min, sent_len_max)
	return stats, counter

wiki_stats, wiki_words = avg_sen_len(wikipedia)
tiger_stats, tiger_words = avg_sen_len(tiger)
news_stats, news_words = avg_sen_len(news)

print(f"STATS FOR WIKIPEDIA")
print(f"Average sentence length of Wikipedia: {wiki_stats[0]}")
print(f"Shortest sentence has length {wiki_stats[1]}, longest {wiki_stats[2]}")
print(f"Most common words in Wikipedia: {wiki_words.most_common(50)}")

print(f"STATS FOR TIGER")
print(f"Average sentence length of Tiger: {tiger_stats[0]}")
print(f"Shortest sentence has length {tiger_stats[1]}, longest {tiger_stats[2]}")
print(f"Most common words in Tiger: {tiger_words.most_common(50)}")

print(f"STATS FOR 10kGNAD")
print(f"Average sentence length of German news corpus: {news_stats[0]}")
print(f"Shortest sentence has length {news_stats[1]}, longest {news_stats[2]}")
print(f"Most common words in 10kGNAD: {news_words.most_common(50)}")

"""
#poss = [" sein ", " seinen ", " seinem ", " seines ", " seiner ", " seine ", " ihr ", " ihren ", " ihrem ", " ihres ", " ihrer ", " ihre "]
poss = ["sein", "seinen", "seinem", "seines", "seiner", "seine", "ihr", "ihren", "ihrem", "ihres", "ihrer", "ihre"]
for p in poss:
	print(p, wiki_words[p])

with open("testsents_wiki.txt", "w+") as f:
	for art in wikipedia:
		sentences = art.split(".")
		for sen in sentences:
			for p in poss:
				if p in sen.split(" "):
					f.write(sen+"\n")
"""
