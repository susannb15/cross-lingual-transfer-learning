from transformers import AutoTokenizer, AutoModelWithLMHead

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

vocab = list(tokenizer.keys())

print(vocab[:25])
