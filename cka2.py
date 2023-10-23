import torch
import numpy as np
from CKA import CKA, CudaCKA
import argparse
import numpy as np
import tqdm
import matplotlib
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead, AutoConfig, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from datasets import DatasetDict
from datasets import ClassLabel, Value
from itertools import chain

#parser = argparse.ArgumentParser(description="Options")

#parser.add_argument('--model1', type=str)
#parser.add_argument('--model2', type=str)
#args = parser.parse_args()
device = torch.device('cuda:0')
#model1 = AutoModelForCausalLM.from_pretrained(args.model1)
#model2 = AutoModelForCausalLM.from_pretrained(args.model2)

native_model = AutoModelForCausalLM.from_pretrained("native_models/native_de/checkpoint-45000")
de = AutoModelForCausalLM.from_pretrained("adapted_models/de_on_de1/checkpoint-60000")
en = AutoModelForCausalLM.from_pretrained("adapted_models/en_on_de4/checkpoint-50000")
es = AutoModelForCausalLM.from_pretrained("adapted_models/es_on_de1/checkpoint-50000")

adapted_models = [de, en, es]
names = ["de", "en", "es"]

with open("cka_langs.txt", "w+", encoding="utf-8") as f:
	f.write("CKA between native DE and adapted models de, en, es\n")
	for name, model in zip(names, adapted_models):
		lins = list()
		kernels = list()
		
		max_len = native_model.transformer.wte.weight.shape[0]
		chunk_len = max_len//10
		chunks = list()
		for i in range(10):
			chunks.append(max_len-chunk_len*i)
		chunks.reverse()
		start = 0
		for c in chunks:
			torch.cuda.empty_cache()
			torch.cuda.set_device(0)
			cka = CudaCKA(device)
			with torch.no_grad():
				X = native_model.transformer.wte.weight[start:c].cuda(0)
				Y = model.transformer.wte.weight[start:c].cuda(0)
			start = c
			
			try:
				lins.append(cka.linear_CKA(X, Y))
				kernels.append(cka.kernel_CKA(X, Y))
			except RuntimeError:
				print(f"chunk {c} triggered OOM.")
			
		cka_lin = sum(lins)/len(lins)
		cka_kernel = sum(kernels)/len(kernels)
		f.write(f'Linear CKA, between native de and adapted {name}: {cka_lin}\n')
		f.write(f'RBF Kernel CKA, between native de and adapted {name}: {cka_kernel}\n')
