from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model_before =  AutoModelWithLMHead.from_pretrained("de_from_scratch/checkpoint-45000")
model_after =  AutoModelWithLMHead.from_pretrained("de_tied/checkpoint-60000")

embeddings_before = model_before.transformer.wte.weight.detach().numpy()
embeddings_after = model_after.transformer.wte.weight.detach().numpy()

emb_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embeddings_before)
ema_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embeddings_after)

plt.scatter(emb_tsne, ema_tsne)
plt.show()
