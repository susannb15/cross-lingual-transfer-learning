from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import torch
import matplotlib.cm as cm

np.random.seed(12)

tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")

model_before =  AutoModelWithLMHead.from_pretrained("de_from_scratch/checkpoint-45000")
model_after =  AutoModelWithLMHead.from_pretrained("es_tied/checkpoint-60000")

embeddings_before = model_before.transformer.wte.weight.detach().numpy()
embeddings_after = model_after.transformer.wte.weight.detach().numpy()

emb_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embeddings_before)
ema_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embeddings_after)

perm = torch.randperm(emb_tsne.shape[0])
idx = perm[:60]


emb_sample = emb_tsne[idx]
ema_sample = ema_tsne[idx]

labels = [tokenizer.decode(w) for w in idx.tolist()]

xs = [emb_sample[:,0], ema_sample[:,0]]

ys = [emb_sample[:,1], ema_sample[:,1]]

print(xs)

fig, ax = plt.subplots(figsize=(20, 10))

colors = iter(cm.rainbow(np.linspace(0, 1, len(ys))))
for x, y in zip(xs, ys):
	ax.scatter(x, y, color=next(colors))

for i, txt in enumerate(labels):
	ax.annotate(txt, (xs[0][i], ys[0][i]), fontsize=7)
	ax.annotate(txt, (xs[1][i], ys[1][i]), fontsize=7)

#plt.scatter(emb_sample, ema_sample)
plt.savefig("tsne_es.png")
plt.show()
