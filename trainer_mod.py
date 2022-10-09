from transformers import Trainer
import math

import contextlib
import functools
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict

from tqdm.auto import tqdm

# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
	default_hp_search_backend,
	get_reporting_integration_callbacks,
	hp_params,
	is_fairscale_available,
	is_optuna_available,
	is_ray_tune_available,
	is_sigopt_available,
	is_wandb_available,
	run_hp_search_optuna,
	run_hp_search_ray,
	run_hp_search_sigopt,
	run_hp_search_wandb,
)

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, deepspeed_reinit, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.optimization import Adafactor, get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
	CallbackHandler,
	DefaultFlowCallback,
	PrinterCallback,
	ProgressCallback,
	TrainerCallback,
	TrainerControl,
	TrainerState,
)
from transformers.trainer_pt_utils import (
	DistributedLengthGroupedSampler,
	DistributedSamplerWithLoop,
	DistributedTensorGatherer,
	IterableDatasetShard,
	LabelSmoother,
	LengthGroupedSampler,
	SequentialDistributedSampler,
	ShardSampler,
	distributed_broadcast_scalars,
	distributed_concat,
	find_batch_size,
	get_parameter_names,
	nested_concat,
	nested_detach,
	nested_numpify,
	nested_truncate,
	nested_xla_mesh_reduce,
	reissue_pt_warnings,
)
from transformers.trainer_utils import (
	PREFIX_CHECKPOINT_DIR,
	BestRun,
	EvalLoopOutput,
	EvalPrediction,
	HPSearchBackend,
	HubStrategy,
	IntervalStrategy,
	PredictionOutput,
	ShardedDDPOption,
	TrainerMemoryTracker,
	TrainOutput,
	default_compute_objective,
	default_hp_space,
	denumpify_detensorize,
	get_last_checkpoint,
	has_length,
	number_of_arguments,
	set_seed,
	speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
	CONFIG_NAME,
	WEIGHTS_INDEX_NAME,
	WEIGHTS_NAME,
	find_labels,
	get_full_repo_name,
	is_apex_available,
	is_datasets_available,
	is_in_notebook,
	is_sagemaker_dp_enabled,
	is_sagemaker_mp_enabled,
	is_torch_tpu_available,
	logging,
)


_is_torch_generator_available = False
_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
	from transformers.utils.notebook import NotebookProgressCallback

	DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
	from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
	_is_torch_generator_available = True
	_is_native_amp_available = True
	from torch.cuda.amp import autocast

if is_datasets_available():
	import datasets

if is_torch_tpu_available():
	import torch_xla.core.xla_model as xm
	import torch_xla.debug.metrics as met
	import torch_xla.distributed.parallel_loader as pl

if is_fairscale_available():
	dep_version_check("fairscale")
	import fairscale
	from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
	from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
	from fairscale.nn.wrap import auto_wrap
	from fairscale.optim import OSS
	from fairscale.optim.grad_scaler import ShardedGradScaler


if is_sagemaker_mp_enabled():
	import smdistributed.modelparallel.torch as smp

	from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat


if TYPE_CHECKING:
	import optuna

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


from transformers import AutoTokenizer

class My_Trainer(Trainer):

	def __init__(
		self,
		model: Union[PreTrainedModel, nn.Module] = None,
		args: TrainingArguments = None,
		data_collator: Optional[DataCollator] = None,
		train_dataset: Optional[Dataset] = None,
		eval_dataset: Optional[Dataset] = None,
		tokenizer: Optional[PreTrainedTokenizerBase] = None,
		model_init: Callable[[], PreTrainedModel] = None,
		compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
		callbacks: Optional[List[TrainerCallback]] = None,
		optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
		preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,):
		super().__init__(
		model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics
)

		self._tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
		self._pronoun_counter = defaultdict(int)
		#self._pronouns = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize("sein"))
		self._pronouns = np.array(self._tokenizer([" sein", " seine", " seiner", " seinen", " seinem", " seines", " ihr", " ihre", " ihrer", " ihren", " ihrem", " ihres"])["input_ids"]).flat
		setattr(self, 'additional_eval_datasets', True)


	def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
		if not self.args.remove_unused_columns:
			return dataset
		if self._signature_columns is None:
			# Inspect model forward signature to keep only the arguments it accepts.
			signature = inspect.signature(self.model.forward)
			self._signature_columns = list(signature.parameters.keys())
			# Labels may be named label or label_ids, the default data collator handles that.
			self._signature_columns += ["label", "label_ids"] 

		ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
		if len(ignored_columns) > 0:
			dset_description = "" if description is None else f"in the {description} set "
			logger.info(
				f"The following columns {dset_description} don't have a corresponding argument in "
				f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
				f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
				f" you can safely ignore this message."
			)

		columns = [k for k in self._signature_columns if k in dataset.column_names]

		if version.parse(datasets.__version__) < version.parse("1.4.0"):
			dataset.set_format(
				type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
			)
			return dataset
		else:
			return dataset.remove_columns(ignored_columns)

	def train(
		self,
		resume_from_checkpoint: Optional[Union[str, bool]] = None,
		trial: Union["optuna.Trial", Dict[str, Any]] = None,
		ignore_keys_for_eval: Optional[List[str]] = None,
		**kwargs,
	):
		"""
		Main training entry point.

		Args:
			resume_from_checkpoint (`str` or `bool`, *optional*):
				If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
				`bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
				of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
			trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
				The trial run or the hyperparameter dictionary for hyperparameter search.
			ignore_keys_for_eval (`List[str]`, *optional*)
				A list of keys in the output of your model (if it is a dictionary) that should be ignored when
				gathering predictions for evaluation during the training.
			kwargs:
				Additional keyword arguments used to hide deprecated arguments
		"""
		resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

		# memory metrics - must set up as early as possible
		self._memory_tracker.start()

		args = self.args

		self.is_in_train = True

		# do_train is not a reliable argument, as it might not be set and .train() still called, so
		# the following is a workaround:
		if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
			self._move_model_to_device(self.model, args.device)

		if "model_path" in kwargs:
			resume_from_checkpoint = kwargs.pop("model_path")
			warnings.warn(
				"`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
				"instead.",
				FutureWarning,
			)
		if len(kwargs) > 0:
			raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
		# This might change the seed so needs to run first.
		self._hp_search_setup(trial)

		# Model re-init
		model_reloaded = False
		if self.model_init is not None:
			# Seed must be set before instantiating the model when using model_init.
			set_seed(args.seed)
			self.model = self.call_model_init(trial)
			model_reloaded = True
			# Reinitializes optimizer and scheduler
			self.optimizer, self.lr_scheduler = None, None

		# Load potential model checkpoint
		if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
			resume_from_checkpoint = get_last_checkpoint(args.output_dir)
			if resume_from_checkpoint is None:
				raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

		if resume_from_checkpoint is not None:
			if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
				raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

			logger.info(f"Loading model from {resume_from_checkpoint}).")

			if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
				config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
				checkpoint_version = config.transformers_version
				if checkpoint_version is not None and checkpoint_version != __version__:
					logger.warning(
						f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
						f"Transformers but your current version is {__version__}. This is not recommended and could "
						"yield to errors or unwanted behaviors."
					)

			if args.deepspeed:
				# will be resumed in deepspeed_init
				pass
			else:
				# We load the model state dict on the CPU to avoid an OOM error.
				state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
				# If the model is on the GPU, it still works!
				self._load_state_dict_in_model(state_dict)

				# release memory
				del state_dict

		# If model was re-initialized, put it on the right device and update self.model_wrapped
		if model_reloaded:
			if self.place_model_on_device:
				self._move_model_to_device(self.model, args.device)
			self.model_wrapped = self.model

		# Data loader and number of training steps
		train_dataloader = self.get_train_dataloader()

		# Setting up training control variables:
		# number of training epochs: num_train_epochs
		# number of training steps per epoch: num_update_steps_per_epoch
		# total number of training steps to execute: max_steps
		total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

		len_dataloader = None
		if has_length(train_dataloader):
			len_dataloader = len(train_dataloader)
			num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
			num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
			num_examples = self.num_examples(train_dataloader)
			if args.max_steps > 0:
				max_steps = args.max_steps
				num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
					args.max_steps % num_update_steps_per_epoch > 0
				)
				# May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
				# the best we can do.
				num_train_samples = args.max_steps * total_train_batch_size
			else:
				max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
				num_train_epochs = math.ceil(args.num_train_epochs)
				num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
		elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
			max_steps = args.max_steps
			# Setting a very large number of epochs so we go as many times as necessary over the iterator.
			num_train_epochs = sys.maxsize
			num_update_steps_per_epoch = max_steps
			num_examples = total_train_batch_size * args.max_steps
			num_train_samples = args.max_steps * total_train_batch_size
		else:
			raise ValueError(
				f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
			)

		if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
			if self.args.n_gpu > 1:
				# nn.DataParallel(model) replicates the model, creating new variables and module
				# references registered here no longer work on other gpus, breaking the module
				raise ValueError(
					"Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
				)
			else:
				debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

		delay_optimizer_creation = (
			self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE or is_sagemaker_mp_enabled()
		)
		if args.deepspeed:
			deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
				self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
			)
			self.model = deepspeed_engine.module
			self.model_wrapped = deepspeed_engine
			self.deepspeed = deepspeed_engine
			self.optimizer = optimizer
			self.lr_scheduler = lr_scheduler
		elif not delay_optimizer_creation:
			self.create_optimizer_and_scheduler(num_training_steps=max_steps)

		self.state = TrainerState()
		self.state.is_hyper_param_search = trial is not None

		# Activate gradient checkpointing if needed
		if args.gradient_checkpointing:
			self.model.gradient_checkpointing_enable()

		model = self._wrap_model(self.model_wrapped)

		# for the rest of this function `model` is the outside model, whether it was wrapped or not
		if model is not self.model:
			self.model_wrapped = model

		if delay_optimizer_creation:
			self.create_optimizer_and_scheduler(num_training_steps=max_steps)

		# Check if saved optimizer or scheduler states exist
		self._load_optimizer_and_scheduler(resume_from_checkpoint)

		# important: at this point:
		# self.model		 is the Transformers Model
		# self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

		# Train!
		logger.info("***** Running training *****")
		logger.info(f"  Num examples = {num_examples}")
		logger.info(f"  Num Epochs = {num_train_epochs}")
		logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
		logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
		logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
		logger.info(f"  Total optimization steps = {max_steps}")

		self.state.epoch = 0
		start_time = time.time()
		epochs_trained = 0
		steps_trained_in_current_epoch = 0
		steps_trained_progress_bar = None

		# Check if continuing training from a checkpoint
		if resume_from_checkpoint is not None and os.path.isfile(
			os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
		):
			self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
			epochs_trained = self.state.global_step // num_update_steps_per_epoch
			if not args.ignore_data_skip:
				steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
				steps_trained_in_current_epoch *= args.gradient_accumulation_steps
			else:
				steps_trained_in_current_epoch = 0

			logger.info("  Continuing training from checkpoint, will skip to saved global_step")
			logger.info(f"  Continuing training from epoch {epochs_trained}")
			logger.info(f"  Continuing training from global step {self.state.global_step}")
			if not args.ignore_data_skip:
				logger.info(
					f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
					"batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
					"flag to your launch command, but you will resume the training on data already seen by your model."
				)
				if self.is_local_process_zero() and not args.disable_tqdm:
					steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
					steps_trained_progress_bar.set_description("Skipping the first batches")

		# Update the references
		self.callback_handler.model = self.model
		self.callback_handler.optimizer = self.optimizer
		self.callback_handler.lr_scheduler = self.lr_scheduler
		self.callback_handler.train_dataloader = train_dataloader
		self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
		if trial is not None:
			assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
			self.state.trial_params = hp_params(assignments)
		else:
			self.state.trial_params = None
		# This should be the same if the state has been saved but in case the training arguments changed, it's safer
		# to set this after the load.
		self.state.max_steps = max_steps
		self.state.num_train_epochs = num_train_epochs
		self.state.is_local_process_zero = self.is_local_process_zero()
		self.state.is_world_process_zero = self.is_world_process_zero()

		# tr_loss is a tensor to avoid synchronization of TPUs through .item()
		tr_loss = torch.tensor(0.0).to(args.device)
		# _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
		self._total_loss_scalar = 0.0
		self._globalstep_last_logged = self.state.global_step
		model.zero_grad()

		self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

		# Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
		if not args.ignore_data_skip:
			for epoch in range(epochs_trained):
				is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
					train_dataloader.sampler, RandomSampler
				)
				if version.parse(torch.__version__) < version.parse("1.11") or not is_random_sampler:
					# We just need to begin an iteration to create the randomization of the sampler.
					# That was before PyTorch 1.11 however...
					for _ in train_dataloader:
						break
				else:
					# Otherwise we need to call the whooooole sampler cause there is some random operation added
					# AT THE VERY END!
					_ = list(train_dataloader.sampler)

		for epoch in range(epochs_trained, num_train_epochs):
			if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
				train_dataloader.sampler.set_epoch(epoch)
			elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
				train_dataloader.dataset.set_epoch(epoch)

			if is_torch_tpu_available():
				parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
				epoch_iterator = parallel_loader
			else:
				epoch_iterator = train_dataloader

			# Reset the past mems state at the beginning of each epoch if necessary.
			if args.past_index >= 0:
				self._past = None

			steps_in_epoch = (
				len(epoch_iterator)
				if len_dataloader is not None
				else args.max_steps * args.gradient_accumulation_steps
			)
			self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

			step = -1
			for step, inputs in enumerate(epoch_iterator):

				self._pronouns = np.array(self._tokenizer([" sein", " seine", " seiner", " seinen", " seinem", " seines", " ihr", " ihre", " ihrer", " ihren", " ihrem", " ihres"])["input_ids"]).flat
				self._count_pronouns(inputs)


				# Skip past any already trained steps if resuming training
				if steps_trained_in_current_epoch > 0:
					steps_trained_in_current_epoch -= 1
					if steps_trained_progress_bar is not None:
						steps_trained_progress_bar.update(1)
					if steps_trained_in_current_epoch == 0:
						self._load_rng_state(resume_from_checkpoint)
					continue
				elif steps_trained_progress_bar is not None:
					steps_trained_progress_bar.close()
					steps_trained_progress_bar = None

				if step % args.gradient_accumulation_steps == 0:
					self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

				if (
					((step + 1) % args.gradient_accumulation_steps != 0)
					and args.local_rank != -1
					and args._no_sync_in_gradient_accumulation
				):
					# Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
					with model.no_sync():
						tr_loss_step = self.training_step(model, inputs)
				else:
					tr_loss_step = self.training_step(model, inputs)

				if (
					args.logging_nan_inf_filter
					and not is_torch_tpu_available()
					and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
				):
					# if loss is nan or inf simply add the average of previous logged losses
					tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
				else:
					tr_loss += tr_loss_step

				self.current_flos += float(self.floating_point_ops(inputs))

				# Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
				if self.deepspeed:
					self.deepspeed.step()

				if (step + 1) % args.gradient_accumulation_steps == 0 or (
					# last step in epoch but step is always smaller than gradient_accumulation_steps
					steps_in_epoch <= args.gradient_accumulation_steps
					and (step + 1) == steps_in_epoch
				):
					# Gradient clipping
					if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
						# deepspeed does its own clipping

						if self.do_grad_scaling:
							# Reduce gradients first for XLA
							if is_torch_tpu_available():
								gradients = xm._fetch_gradients(self.optimizer)
								xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
							# AMP: gradients need unscaling
							self.scaler.unscale_(self.optimizer)

						if hasattr(self.optimizer, "clip_grad_norm"):
							# Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
							self.optimizer.clip_grad_norm(args.max_grad_norm)
						elif hasattr(model, "clip_grad_norm_"):
							# Some models (like FullyShardedDDP) have a specific way to do gradient clipping
							model.clip_grad_norm_(args.max_grad_norm)
						else:
							# Revert to normal clipping otherwise, handling Apex or full precision
							nn.utils.clip_grad_norm_(
								amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
								args.max_grad_norm,
							)

					# Optimizer step
					optimizer_was_run = True
					if self.deepspeed:
						pass  # called outside the loop
					elif is_torch_tpu_available():
						if self.do_grad_scaling:
							self.scaler.step(self.optimizer)
							self.scaler.update()
						else:
							xm.optimizer_step(self.optimizer)
					elif self.do_grad_scaling:
						scale_before = self.scaler.get_scale()
						self.scaler.step(self.optimizer)
						self.scaler.update()
						scale_after = self.scaler.get_scale()
						optimizer_was_run = scale_before <= scale_after
					else:
						self.optimizer.step()

					if optimizer_was_run and not self.deepspeed:
						self.lr_scheduler.step()


					
					 # log gradient norm
					#parameters = [(n, p) for (n, p) in model.named_parameters() if p.grad is not None]
					#grad_norms = {n: torch.linalg.norm(p.grad, ord=2).item() for (n, p) in parameters}	
					#grad_norms = {"WTE_grad_norm": torch.linalg.norm(model.transformer.wte.weight.grad, ord=2).item()} 
					total_norm = 0
					for n, p in self.model.named_parameters():
						if p.grad is None:
							continue
						grads = p.grad.view(-1)
						grads_norm = torch.norm(grads, p=2, dim=0)
						total_norm += grads_norm.item() ** 2
					total_norm = total_norm ** 0.5
					grad_norms = {"Gradient Norm": total_norm}

					model.zero_grad()
					self.state.global_step += 1
					self.state.epoch = epoch + (step + 1) / steps_in_epoch
					self.control = self.callback_handler.on_step_end(args, self.state, self.control)

					self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)#, grad_norms=grad_norms)
				else:
					self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

				if self.control.should_epoch_stop or self.control.should_training_stop:
					break
			if step < 0:
				logger.warning(
					f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
					f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
					f" num_steps ({max_steps}) higher than the number of available samples."
				)
				self.control.should_training_stop = True

			self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
			self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)#, grad_norms=grad_norms)

			if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
				if is_torch_tpu_available():
					# tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
					xm.master_print(met.metrics_report())
				else:
					logger.warning(
						"You enabled PyTorch/XLA debug metrics but you don't have a TPU "
						"configured. Check your training configuration if this is unexpected."
					)
			if self.control.should_training_stop:
				break

		if args.past_index and hasattr(self, "_past"):
			# Clean the state at the end of training
			delattr(self, "_past")

		logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
		if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
			# Wait for everyone to get here so we are sur the model has been saved by process 0.
			if is_torch_tpu_available():
				xm.rendezvous("load_best_model_at_end")
			elif args.local_rank != -1:
				dist.barrier()

			logger.info(
				f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
			)

			best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
			if os.path.exists(best_model_path):
				if self.deepspeed:
					# temp hack until Deepspeed fixes the problem with resume from an existing engine that did some stepping
					deepspeed_engine, optimizer, lr_scheduler = deepspeed_reinit(self)
					self.model = deepspeed_engine.module
					self.model_wrapped = deepspeed_engine
					self.deepspeed = deepspeed_engine
					self.optimizer = optimizer
					self.lr_scheduler = lr_scheduler
					self.deepspeed.load_checkpoint(
						self.state.best_model_checkpoint, load_optimizer_states=True, load_lr_scheduler_states=True
					)
				else:
					# We load the model state dict on the CPU to avoid an OOM error.
					state_dict = torch.load(best_model_path, map_location="cpu")
					# If the model is on the GPU, it still works!
					self._load_state_dict_in_model(state_dict)
			else:
				logger.warning(
					f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
					"on multiple nodes, you should activate `--save_on_each_node`."
				)

		# add remaining tr_loss
		self._total_loss_scalar += tr_loss.item()
		train_loss = self._total_loss_scalar / self.state.global_step

		metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
		self.store_flos()
		metrics["total_flos"] = self.state.total_flos
		metrics["train_loss"] = train_loss

		self.is_in_train = False

		self._memory_tracker.stop_and_update_metrics(metrics)

		self.log(metrics)

		self.control = self.callback_handler.on_train_end(args, self.state, self.control)

		return TrainOutput(self.state.global_step, train_loss, metrics)


	def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, grad_norms={}):
		if self.control.should_log:
			if is_torch_tpu_available():
				xm.mark_step()

			logs: Dict[str, float] = {}

			# all_gather + mean() to get average loss over all processes
			tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

			# reset tr_loss to zero
			tr_loss -= tr_loss

			logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
			logs["learning_rate"] = self._get_learning_rate()

			logs["perplexity"] = math.exp(logs["loss"])
			for n in grad_norms:
				logs[n] = grad_norms[n]


			self._total_loss_scalar += tr_loss_scalar
			self._globalstep_last_logged = self.state.global_step
			self.store_flos()

			self.log(logs)

		metrics = None
		if self.control.should_evaluate:
			if self.additional_eval_datasets:
				for dataset in self.eval_dataset:
					metrics = None
					print(f'Evaluating on {dataset}')
					metrics = self.evaluate(self.eval_dataset[dataset], ignore_keys=ignore_keys_for_eval, metric_key_prefix=f"eval-{dataset}")
					self._report_to_hp_search(trial, epoch, metrics)
			else:
				metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
				self._report_to_hp_search(trial, epoch, metrics)

		if self.control.should_save:
			self._save_checkpoint(model, trial, metrics=metrics)
			self.control = self.callback_handler.on_save(self.args, self.state, self.control)

	def _count_pronouns(self, batch):
		for p in self._pronouns:
			self._pronoun_counter[self._tokenizer.decode(p)] += torch.numel(batch["input_ids"][batch["input_ids"]== p])


	def evaluation_loop(
		self,
		dataloader: DataLoader,
		description: str,
		prediction_loss_only: Optional[bool] = None,
		ignore_keys: Optional[List[str]] = None,
		metric_key_prefix: str = "eval",
	) -> EvalLoopOutput:
		"""
		Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

		Works both with or without labels.
		"""
		args = self.args

		prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

		# if eval is called w/o train init deepspeed here
		if args.deepspeed and not self.deepspeed:

			# XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
			# from the checkpoint eventually
			deepspeed_engine, _, _ = deepspeed_init(
				self, num_training_steps=0, resume_from_checkpoint=None, inference=True
			)
			self.model = deepspeed_engine.module
			self.model_wrapped = deepspeed_engine
			self.deepspeed = deepspeed_engine

		model = self._wrap_model(self.model, training=False)

		# if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
		# while ``train`` is running, cast it to the right dtype first and then put on device
		if not self.is_in_train:
			if args.fp16_full_eval:
				model = model.to(dtype=torch.float16, device=args.device)
			elif args.bf16_full_eval:
				model = model.to(dtype=torch.bfloat16, device=args.device)

		batch_size = self.args.per_device_eval_batch_size

		logger.info(f"***** Running {description} *****")
		if has_length(dataloader):
			logger.info(f"  Num examples = {self.num_examples(dataloader)}")
		else:
			logger.info("  Num examples: Unknown")
		logger.info(f"  Batch size = {batch_size}")

		model.eval()

		self.callback_handler.eval_dataloader = dataloader
		# Do this before wrapping.
		eval_dataset = getattr(dataloader, "dataset", None)

		if is_torch_tpu_available():
			dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

		if args.past_index >= 0:
			self._past = None

		# Initialize containers
		# losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
		losses_host = None
		preds_host = None
		labels_host = None
		# losses/preds/labels on CPU (final containers)
		all_losses = None
		all_preds = None
		all_labels = None
		# Will be useful when we have an iterable dataset so don't know its length.

		observed_num_examples = 0
		# Main evaluation loop
		for step, inputs in enumerate(dataloader):
			# Update the observed num examples
			observed_batch_size = find_batch_size(inputs)
			if observed_batch_size is not None:
				observed_num_examples += observed_batch_size
				# For batch samplers, batch_size is not known by the dataloader in advance.
				if batch_size is None:
					batch_size = observed_batch_size

			# Prediction step
			loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

			if is_torch_tpu_available():
				xm.mark_step()

			# Update containers on host
			if loss is not None:
				losses = self._nested_gather(loss.repeat(batch_size))
				losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
			if labels is not None:
				labels = self._pad_across_processes(labels)
				labels = self._nested_gather(labels)
				labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
			if logits is not None:
				logits = self._pad_across_processes(logits)
				logits = self._nested_gather(logits)
				if self.preprocess_logits_for_metrics is not None:
					logits = self.preprocess_logits_for_metrics(logits, labels)
				preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
			self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

			# Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
			if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
				if losses_host is not None:
					losses = nested_numpify(losses_host)
					all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
				if preds_host is not None:
					logits = nested_numpify(preds_host)
					all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
				if labels_host is not None:
					labels = nested_numpify(labels_host)
					all_labels = (
						labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
					)

				# Set back to None to begin a new accumulation
				losses_host, preds_host, labels_host = None, None, None

		if args.past_index and hasattr(self, "_past"):
			# Clean the state at the end of the evaluation loop
			delattr(self, "_past")

		# Gather all remaining tensors and put them back on the CPU
		if losses_host is not None:
			losses = nested_numpify(losses_host)
			all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
		if preds_host is not None:
			logits = nested_numpify(preds_host)
			all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
		if labels_host is not None:
			labels = nested_numpify(labels_host)
			all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

		# Number of samples
		if has_length(eval_dataset):
			num_samples = len(eval_dataset)
		# The instance check is weird and does not actually check for the type, but whether the dataset has the right
		# methods. Therefore we need to make sure it also has the attribute.
		elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
			num_samples = eval_dataset.num_examples
		else:
			if has_length(dataloader):
				num_samples = self.num_examples(dataloader)
			else:  # both len(dataloader.dataset) and len(dataloader) fail
				num_samples = observed_num_examples

		# Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
		# samplers has been rounded to a multiple of batch_size, so we truncate.
		if all_losses is not None:
			all_losses = all_losses[:num_samples]
		if all_preds is not None:
			all_preds = nested_truncate(all_preds, num_samples)
		if all_labels is not None:
			all_labels = nested_truncate(all_labels, num_samples)

		# Metrics!
		if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
			metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
		else:
			metrics = {}

		# To be JSON-serializable, we need to remove numpy types or zero-d tensors
		metrics = denumpify_detensorize(metrics)

		if all_losses is not None:
			metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
			metrics[f"{metric_key_prefix}_perplexity"] = math.exp(all_losses.mean().item())

		# Prefix all keys with metric_key_prefix + '_'
		for key in list(metrics.keys()):
			if not key.startswith(f"{metric_key_prefix}_"):
				metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

		return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
