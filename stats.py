from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
train = load_dataset("wikipedia", "20220301.de", split='train[:1%]')
poss = [" sein ", " seine ", " seinen ", " seines ", " seinem ", " seiner ", " ihr ", " ihre ", " ihren ", " ihres ", " ihrem ", " ihrer "]

text = train["text"]
all_texts = len(text)
text_len = 0
print(all_texts)
"""
with open("stats.txt", "w+", encoding="utf-8") as f:
	for p in poss:
		count = 0
		for t in text:
			count += t.count(p)
		f.write(p+"\t"+str(count)+"\n")
		f.write("\t"+str(count/all_texts)+"\n")
	#for t in text:
	#	text_len += len(tokenizer.tokenize(t))
	#f.write("AVG TEXT LENGTH:\t"+str(text_len / all_texts))
"""
for t in text:
	text_len = len(tokenizer.tokenize(t))
	print(tokenizer.tokenize(t))
print("AVG TEXT LENGTH:\t"+str(text_len / all_texts))		
