export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NGPU=8
export WORK_DIR=./output/train_from_sana
export LOAD_FROM=hf://Efficient-Large-Model/Sana_1600M_1024px_BF16/checkpoints/Sana_1600M_1024px_BF16.pth

torchrun --nproc_per_node=$NGPU --master_port=21540 scripts/train.py \
  --config_path $1 \
  --model.load_from $LOAD_FROM \
  --work_dir $WORK_DIR \
  --resume_from latest \
  --resume_from_sana true \
  --train.train_batch_size 4 \
  --model.use_pe true
