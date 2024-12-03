output_dir=logs/cogvideox_vae_encode

msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir scripts/infer_vae.py \
    --csv_path datasets/mixkit-100videos/video_caption_train.csv \
    --video_folder datasets/mixkit-100videos/mixkit \
    --output_path datasets/mixkit-100videos/vae_768_1360 \
    --vae_type CogVideoX-VAE \
    --image_size 768 1360 \
    --vae_checkpoint models/CogVideoX-1.5-5b/vae/diffusion_pytorch_model.safetensors \
    --dtype fp32 \
    --mode 1 \
    --max_frames 85 \
    --num_parallel_workers 1 \
    --use_parallel True \
    --transform_name crop_resize
