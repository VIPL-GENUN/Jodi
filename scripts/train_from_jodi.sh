export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NGPU=8
export WORK_DIR=./output/train_from_jodi
export LOAD_FROM=hf://VIPL-GENUN/Jodi/Jodi.pth

torchrun --nproc_per_node=$NGPU --master_port=21540 scripts/train.py \
  --config_path $1 \
  --model.load_from $LOAD_FROM \
  --work_dir $WORK_DIR \
  --resume_from latest \
  --resume_from_sana false \
  --train.train_batch_size 4 \
  --model.use_pe true
