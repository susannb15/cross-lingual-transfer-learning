export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR="/local/susannb/cross-lingual-transfer-learning/adapted_models/es_on_de"
NATIVE_PATH="/local/susannb/cross-lingual-transfer-learning/native_models/native_es/checkpoint-45000"

python /local/susannb/cross-lingual-transfer-learning/adapt.py \
	--seed 12 \
	--name es_on_de \
	--group adaptation_on_de \
	--lr 1e-4 \
	--tied_weights \
	--output_dir $OUTPUT_DIR \
	--native_model $NATIVE_PATH \
	--block_size 256 \
	--weight_decay 0.01 \
	--max_steps 100000 \
	--eval_steps 5000 \
	--save_steps 5000 \
	--warmup_steps 30000 \
	--language de
