output_dir=logs/cogvideox_5b_1_5_sp_adamw_8p

msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir scripts/train.py \
    --config configs/cogvideox_5b-v1-5/train/train_t2v.yaml \
    --csv_path datasets/mixkit-100videos/video_caption_train.csv \
    --video_folder datasets/mixkit-100videos/mixkit \
    --text_embed_folder datasets/mixkit-100videos/t5_224 \
    --vae_latent_folder datasets/mixkit-100videos/vae_768_1360 \
    --use_parallel True \
    --enable_sequence_parallelism True \
    --sequence_parallel_shards 8
