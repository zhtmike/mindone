python scripts/infer_vae_decode.py \
    --latent_folder samples/denoised_latents \
    --output_path samples \
    --vae_type CogVideoX-VAE \
    --sd_scale_factor 0.7 \
    --vae_checkpoint models/CogVideoX-1.5-5b/vae/diffusion_pytorch_model.safetensors \
    --dtype fp32 \
    --mode 1 \
    --save_format mp4 \
    --fps 8
