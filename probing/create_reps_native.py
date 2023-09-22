from rep_creator import RepCreator
from transformers import AutoTokenizer, AutoModelForCausalLM

output_dir = "representations_native"
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model = AutoModelForCausalLM.from_pretrained("native_models/native_de/checkpoint-45000")

rc = RepCreator(tokenizer, model, output_dir)
rc._model_rep_train()
rc._model_rep_test()
rc._write_reps_to_file()
