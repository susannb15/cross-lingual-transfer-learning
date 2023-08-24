export CUDA_VISIBLE_DEVICES=1

OUTPUT_DIR="/local/susannb/cross-lingual-transfer-learning/adapted_models/de_on_de_shuf75_5"
NATIVE_PATH="/local/susannb/cross-lingual-transfer-learning/native_models/native_de/checkpoint-45000"

python /local/susannb/cross-lingual-transfer-learning/adapt.py \
	--seed 5 \
	--name de_on_de_shuf75 \
	--group adaptation_on_de_shuf_perc \
	--lr 1e-4 \
	--tied_weights \
	--shuffle_perc 0.75 \
	--output_dir $OUTPUT_DIR \
	--native_model $NATIVE_PATH \
	--block_size 256 \
	--weight_decay 0.01 \
	--max_steps 100000 \
	--eval_steps 5000 \
	--save_steps 5000 \
	--warmup_steps 30000 \
	--language de
