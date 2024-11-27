output_dir=outputs/cogvideox_5b_1_5_sp_came_8p

msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir scripts/train.py \
    --config configs/cogvideox_5b-v1-5/train/train_t2v.yaml \
    --csv_path datasets/mixkit-100videos/video_caption_train.csv \
    --video_folder datasets/mixkit-100videos/mixkit \
    --text_embed_folder  datasets/mixkit-100videos/t5_emb_224 \
    --vae_latent_folder datasets/mixkit-100videos/vae_448_720 \
    --use_parallel True \
    --zero_stage 2 \
    --num_frames 69 \
    --num_latent_frames 18 \
    --enable_sequence_parallelism True \
    --sequence_parallel_shards 8 \
    --optim came \
    --betas 0.9 0.95 0.999 \
    --optim_eps 1e-8 1e-8
