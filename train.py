import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["WANDB_DISABLED"] = "true"

import transformers
from datasets import load_dataset
from datasets import DatasetDict
from itertools import chain
import torch
import torch.nn as nn
import argparse
import math
from trainer_mod import Mod_Trainer


parser = argparse.ArgumentParser(description="Options")

parser.add_argument('--model', type=str, required=True, 
					help="Model. Options: de, es")

args = parser.parse_args()

datasets = DatasetDict()
train = load_dataset("wikipedia", "20220301.de", split='train[:10%]')
validation = load_dataset("wikipedia", "20220301.de", split='train[11:13%]')
datasets["train"] = train
datasets["validation"] = validation

# add new column to store pos values 
datasets["train"] = datasets["train"].add_column("pos", ["0"] * len(datasets["train"]))
datasets["validation"] = datasets["validation"].add_column("pos", ["0"] * len(datasets["validation"]))

"""
train = load_dataset("wikipedia", "20220301.de", split='train[:3%]')
validation = load_dataset("wikipedia", "20220301.de", split='train[4:5%]')
"""
from datasets import ClassLabel
import random
import pandas as pd

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])

#print(show_random_elements(datasets["train"]))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelWithLMHead

de_model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")

if args.model == "es":
	#tokenizer = AutoTokenizer.from_pretrained("datificate/gpt2-small-spanish")
	tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
	model = AutoModelForCausalLM.from_pretrained("datificate/gpt2-small-spanish")
	embeddings = model.transformer.wte.weight
	perm = torch.randperm(embeddings.shape[0])
	copy_idx = perm[:8]
	copies = embeddings[copy_idx]
	extended_emb = torch.cat((embeddings, copies), 0)
	idx = torch.randperm(extended_emb.shape[0])
	shuffled = extended_emb[idx]
	model.transformer.wte.weight = nn.Parameter(shuffled)

	lm_head = model.lm_head.weight
	perm = torch.randperm(lm_head.shape[0])
	copy_idx = perm[:8]
	copies = lm_head[copy_idx]
	extended_head = torch.cat((lm_head, copies), 0)
	idx = torch.randperm(extended_head.shape[0])
	shuffled = extended_head[idx]
	model.lm_head.weight = nn.Parameter(shuffled)

elif args.model == "de":
	tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
	model = de_model
	de_embeddings = de_model.transformer.wte.weight # DE word token embeddings
	idx = torch.randperm(de_embeddings.shape[0])
	de_emb_shuffled = de_embeddings[idx]
	model.transformer.wte.weight = nn.Parameter(de_emb_shuffled)

	lm_head = model.lm_head.weight
	idx = torch.randperm(lm_head.shape[0])
	shuffled = lm_head[idx]
	model.lm_head.weight = nn.Parameter(shuffled)
	

else:
	print("Illegal model name. Choose between: es, de")

#print("EMB SHAPE")
#print(de_emb_shuffled.shape)
#de_emb_shuffled = de_emb_shuffled[:50257]
#print(de_emb_shuffled.shape)

# replace es embeddings with the shuffled de embeddings
#print(model.transformer.wte.weight.shape)
#print(model.lm_head.weight.shape)

def freeze_model(model):
	for name, param in model.named_parameters():
		param.requires_grad = False
		#print(name, param.shape)

# freeze es model parameters except embeddings
freeze_model(model)
model.transformer.wte.weight.requires_grad = True
model.lm_head.weight.requires_grad = True
#print("test freeze")
#for name, param in model.named_parameters():
    #if param.requires_grad:
        #print(name)

def tokenize_function(examples):
	return tokenizer(examples["text"])

def pos_function(dataset):
	pos = [" sein ", " seinen ", " seinem ", " seine ", " seines ", " seiner ", " ihr ", " ihren ", " ihrem ", " ihre ", " ihres ", " ihrer "]
	for example in dataset:
		count = 0
		for p in pos:
			count += example["text"].count(p)
		example["pos"] = str(count)

pos_function(datasets["train"])
pos_function(datasets["validation"])
#datasets = datasets.map(pos_function,batched=True, num_proc=4)
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
#tokenized_datasets = tok_datasets.map(pos_function, batched=True, num_proc=4)
print(tokenized_datasets["train"][1])
"""
train_tok = train.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
val_tok = validation.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
"""
block_size=128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
        }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)


#lm_train = train_tok.map(group_texts, batched=True, batch_size=1000, num_proc=4)
#lm_val = val_tok.map(group_texts, batched=True, batch_size=1000, num_proc=4)

#print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))
pos_1 = lm_datasets["train"].filter(lambda example: example["pos"] != "0")
print(tokenizer.decode(pos_1[1]["input_ids"]))

from transformers import Trainer, TrainingArguments
from transformers.integrations import *

class MyCallback(WandbCallback):
    """
    A [`TrainerCallback`] that sends the logs to [Weight and Biases](https://www.wandb.com/).
    """

    def __init__(self):
        has_wandb = is_wandb_available()
        if not has_wandb:
            raise RuntimeError("WandbCallback requires wandb to be installed. Run `pip install wandb`.")
        if has_wandb:
            import wandb

            self._wandb = wandb
        self._initialized = False
        # log outputs
        self._log_model = os.getenv("WANDB_LOG_MODEL", "FALSE").upper() in ENV_VARS_TRUE_VALUES.union({"TRUE"})

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name

            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=run_name,
                    **init_args,
                )
            # add config parameters (run may have been created manually)
            self._wandb.config.update(combined_dict, allow_val_change=True)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, args.logging_steps)
                )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        hp_search = state.is_hyper_param_search
        if hp_search:
            self._wandb.finish()
            self._initialized = False
            args.run_name = None
        if not self._initialized:
            self.setup(args, state, model, **kwargs)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model and self._initialized and state.is_world_process_zero:
            from .trainer import Trainer

            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                artifact = self._wandb.Artifact(name=f"model-{self._wandb.run.id}", type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})
            ppl = math.exp(logs["train/loss"])
            self._wandb.log({"perplexity": ppl})

    def on_step_begin(self, args, state, control, model=None, logs=None, **kwargs):
        parameters = [(n, p) for (n, p) in model.named_parameters() if p.grad is not None and p.requires_grad]
        for n, p in parameters:
            param_norm = p.grad.detach().data.norm(2).item()
            self._wandb.log({str(n)+"grad_norm": param_norm})
        #self._wandb.log({"sents_with_poss": model.batch["pos"]})

training_args = TrainingArguments(
    args.model+"_model",
	do_train=True,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
	num_train_epochs=10,
	seed=42
)


trainer = Mod_Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)


trainer.train()
trainer.evaluate()
