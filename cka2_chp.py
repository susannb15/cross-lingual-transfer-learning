import torch
import numpy as np
from CKA import CKA, CudaCKA
import argparse
import numpy as np
from tqdm import tqdm
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
cka = CudaCKA(device=device)
torch.cuda.set_device(0)
#model1 = AutoModelForCausalLM.from_pretrained(args.model1)
#model2 = AutoModelForCausalLM.from_pretrained(args.model2)

native_model = AutoModelForCausalLM.from_pretrained("native_models/native_de/checkpoint-45000")
#de = "adapted_models/de_on_de1"
#en = "adapted_models/en_on_de4"
es = "adapted_models/es_on_de1"

checkpoints = np.arange(5000, 105000, 5000)

def get_chpt(model_type, chpt):
    print(f"Evaluating model type {model_type} on checkpoint {chpt}")
    return model_type+"/checkpoint-"+str(chpt)

#adapted_models = [de, en, es]
#names = ["de", "en", "es"]

# write into a pandas readable file
#with open("cka_de.txt", "w+", encoding="utf-8") as f:
with open("cka_es.csv", "w+", encoding="utf-8") as g:
	#f.write("CKA between native DE and all checkpoints of the adapted de model\n")
	g.write("step,cka")	
	for chp in tqdm(checkpoints):
		lins = list()
		kernels = list()

		model = AutoModelForCausalLM.from_pretrained(get_chpt(es, chp))

		max_len = native_model.transformer.wte.weight.shape[0]
		chunk_len = max_len//10
		chunks = list()
		for i in range(10):
			chunks.append(max_len-chunk_len*i)
		chunks.reverse()
		start = 0
		for c in chunks:
			with torch.no_grad():
				X = native_model.transformer.wte.weight[start:c].cuda(0)
				Y = model.transformer.wte.weight[start:c].cuda(0)
			lins.append(cka.linear_CKA(X, Y))
			kernels.append(cka.kernel_CKA(X, Y))
			start = c

		cka_lin = sum(lins)/len(lins)
		cka_kernel = sum(kernels)/len(kernels)
		#f.write(f'Linear CKA, between native de and adapted de at step {chp}: {cka_lin}\n')
		#f.write(f'RBF Kernel CKA, between native de and adapted de at step {chp}: {cka_kernel}\n')
		g.write(f"\n{chp},{cka_lin}")
