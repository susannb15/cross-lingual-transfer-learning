from datasets import load_dataset
from collections import Counter
import itertools

# load corpora
wikipedia = load_dataset("wikipedia", "20220301.de", split='train[95:100%]')
#tiger = load_dataset("text", encoding='ISO-8859-1', data_files={'validation': 'tiger.txt'})

wikipedia = wikipedia['text']
#tiger = tiger['validation']['text']

#words_wikipedia = Counter(wikipedia)
#words_tiger = Counter(tiger)

def avg_sen_len(text):
	counter = Counter()
	len_words = 0
	len_sen = 0
	for sen in text:
		sentences = sen.split(".")
		words = sen.split(" ")
		c = Counter(words)
		counter.update(c)
		len_words += len(words)
		len_sen += len(sentences)
	avg_len = len(words) / len(sentences)
	return avg_len, counter

wiki_len, wiki_words = avg_sen_len(wikipedia)
#tiger_len, tiger_words = avg_sen_len(tiger)
print(f"Average sentence length of Wikipedia: {wiki_len}")
#print(f"Average sentence length of Tiger: {tiger_len}")

print(f"Most common words in Wikipedia: {wiki_words.most_common(50)}")
#print(f"Most common words in Tiger: {tiger_words.most_common(20)}")

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
