output_dir=logs/cogvideox_5b_1_5_dp_adamw_lora_8p

msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir scripts/train.py \
    --config configs/cogvideox_5b-v1-5/train/train_t2v.yaml \
    --csv_path datasets/mixkit-100videos/video_caption_train.csv \
    --video_folder datasets/mixkit-100videos/mixkit \
    --text_embed_folder datasets/mixkit-100videos/t5_224 \
    --use_parallel True \
    --warmup_steps 100 \
    --start_learning_rate 1e-4 \
    --lora_dim 128
