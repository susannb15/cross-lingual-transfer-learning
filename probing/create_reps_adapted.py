from rep_creator import RepCreator
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
# create de on de representations

chps = np.arange(5000,105000, 5000)
dirs = [os.path.join("representations_es", str(i)) for i in chps]
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
models = ["../adapted_models/es_on_de1/checkpoint-"+str(i) for i in chps]

for (m_path, d) in zip(models, dirs):
	m = AutoModelForCausalLM.from_pretrained(m_path)
	print(f"Creating representations for {m_path} in dir {d}")
	rc = RepCreator(tokenizer, m, d)
	rc._model_rep_train()
	rc._model_rep_test()
	rc._write_reps_to_file()
