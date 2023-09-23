from torch_cka import CKA
import argparse
import torch
import numpy as np
import torchvision
import tqdm
import matplotlib

parser = argparse.ArgumentParser(description="Options")

parser.add_argument('--model1', type=str)
parser.add_argument('--model2', type=str)
parser.add_argument('--model1_name', type=str)
parser.add_argument('--model2_name', type=str)

def set_seed(seed: int = 123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

args = parser.parse_args()
set_seed()

model1 = args.model1
model2 = args.model2

dataset = "probing/spacy_sents.txt"

dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

cka = CKA(model1, model2, model1_name=args.model1_name, model2_name=args.model2_name, model1_layers=model1.transformer.wte, model2_layers=model2.transformer.wte, device='cuda:2') 

cka.compare(dataloader)

results = cka.export()

print(results.keys())
