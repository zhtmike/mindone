# Efficient CogVideoX based on MindSpore

## Features

- [x] CogVideoX v1.5 2B/5B inference
    - [ ] Linear-quadratic t-schedule (new)
- [x] CogVideoX v1.5 2B/5B SFT supporting 736P videos of 77 frames
- [x] Memory efficiency optimization (new)
    - [x] VAE cache in training (to be optimized)
    - [x] CAME optimizer in training
    - [x] Context parallelism + Zero2 for distributed training based on MindSpore
    - [ ] Simplied VAE tiling
- [ ] Arbitary aspect-ratio training (new)
- [ ] LoRA Fine-tuning


TODO: A table comparing to original CogVideoX, including video resolution, frames, memory reqirement...

## Requirements

| mindspore | ascend driver   | firmware       | cann toolkit/kernel   |
|:---------:|:---------------:|:--------------:|:---------------------:|
| 2.4.0     | 24.1.RC3        | 7.5.T15.1.B120 | 8.0.RC3               |


Python: 3.9 or later.

Then run `pip install -r requirements.txt` to install the necessary packages.

## Prepare Model Weights

Please refer to the [official repository of CogVideo](https://github.com/THUDM/CogVideo) for downloading pretrained checkpoints. Download the [Transformer](https://huggingface.co/THUDM/CogVideoX1.5-5B/tree/main/transformer) and [VAE](https://huggingface.co/THUDM/CogVideoX1.5-5B/tree/main/vae), and place them under the `models/CogVideoX-1.5-5b` directory.

Additionally, download the [Tokenizer](https://huggingface.co/THUDM/CogVideoX1.5-5B/tree/main/tokenizer) and [Text Encoder](https://huggingface.co/THUDM/CogVideoX1.5-5B/tree/main/text_encoder), and place all files in the `models/t5-v1_1-xxl` directory.

After downloading, the checkpoints under `models/` should look like this:
```bash
models/
├── CogVideoX-1.5-5b/
│  ├── Transformer/
│  └── VAE/
└── t5-v1_1-xxl/
```

## Inference

You can then run the inference using `scripts/inference.py`. For examples, to generate a 768x1360 resolution video with 81 frames, you can run:

```bash
python scripts/inference.py -c configs/cogvideox_5b-v1-5/inference/sample_t2v.yaml --captions "your caption"
```

## Prapare Dataset for Training

We support fine-tuning the CogVideoX v1.5 model with context parallel on at least 8 NPU* devices. To start the fine-tuning, you should prepare the video-text dataset with T5 and VAE Cache.

### Video-text Dataset

Organize the text-video pair data as follows:

```bash
.
├── video_caption.csv
├── video_folder
│   ├── part01
│   │   ├── vid001.mp4
│   │   ├── vid002.mp4
│   │   └── ...
│   └── part02
│       ├── vid001.mp4
│       ├── vid002.mp4
│       └── ...
```

The `video_folder` contains all the video files. The csv file `video_caption.csv` records the relative video paths and their text captions, like this:

```text
video,caption
video_folder/part01/vid001.mp4,a cartoon character is walking through
video_folder/part01/vid002.mp4,a red and white ball with an angry look on its face
```

### T5 Cache

Store the T5 cache using the following command:

```bash
python scripts/infer_t5.py \
    --csv_path /path/to/video_caption.csv \
    --output_path /path/to/text_embed_folder \
    --model_max_length 224 \
    --t5_model transformers \
    --dtype fp32 \
    --mode 1 \
    --require_mask False \
    --predict_empty_text_embedding True

```

After running the script, the text embeddings will be saved as `.npz` files for each caption in the specified `output_path`.

### VAE Cache

Save the VAE cache using the following commands:

```bash
python scripts/infer_vae.py \
    --csv_path /path/to/video_caption.csv \
    --video_folder /path/to/video \
    --output_path /path/to/vae_cache \
    --vae_type CogVideoX-VAE \
    --image_size 768 1360 \
    --vae_checkpoint models/CogVideoX-1.5-5b/vae/diffusion_pytorch_model.safetensors \
    --dtype fp32 \
    --mode 1 \
    --max_frames 85 \
    --num_parallel_workers 1 \
    --transform_name crop_resize
```

The `--max_frames 85` parameter specifies the maximum number of frames for each video you may want to encode. During the training stage, the number of frames should be less than or equal to the `max_frames` specified here.

To speed up cache generation, you can run VAE encoding with 8 NPU* devices using the following command:

```bash
msrun --worker_num=8 --local_worker_num=8 --log_dir="logs/vae" scripts/infer_vae.py \
    --csv_path /path/to/video_caption.csv \
    --video_folder /path/to/video \
    --output_path /path/to/vae_cache \
    --vae_type CogVideoX-VAE \
    --image_size 768 1360 \
    --vae_checkpoint models/CogVideoX-1.5-5b/vae/diffusion_pytorch_model.safetensors \
    --dtype fp32 \
    --mode 1 \
    --max_frames 85 \
    --num_parallel_workers 1 \
    --transform_name crop_resize \
    --use_parallel True
```

After running the script, the VAE latents will be saved as `.npz` files for each video in the specified `output_path`.

Each `.npz` file contains data for the following keys:
- `latent_mean`: Mean of the VAE latent distribution
- `latent_std`: Standard deviation of the VAE latent distribution
- `fps`: Video frames per second
- `ori_size`: Original size (height, width) of the video

Finally, the training data structure should look like this:
```bash
.
├── video_caption.csv
├── video_folder
│   ├── part01
│   │   ├── vid001.mp4
│   │   ├── vid002.mp4
│   │   └── ...
│   └── part02
│       ├── vid001.mp4
│       ├── vid002.mp4
│       └── ...
├── text_embed_folder
│   ├── part01
│   │   ├── vid001.npz
│   │   ├── vid002.npz
│   │   └── ...
│   └── part02
│       ├── vid001.npz
│       ├── vid002.npz
│       └── ...
├── video_embed_folder
│   ├── part01
│   │   ├── vid001.npz
│   │   ├── vid002.npz
│   │   └── ...
│   └── part02
│       ├── vid001.npz
│       ├── vid002.npz
│       └── ...

```

## Training (SFT)

After preprare the T5 and VAE cache, you may then start the training with 8 NPU* cards using the command

```bash
msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir scripts/train.py \
    --config configs/cogvideox_5b-v1-5/train/train_t2v.yaml \
    --csv_path /path/to/video_caption.csv \
    --video_folder /path/to/video \
    --text_embed_folder /path/to/text_embed_folder \
    --vae_latent_folder /path/to/vae_cache \
    --use_parallel True \
    --zero_stage 2 \
    --num_frames 77 \
    --num_latent_frames 20 \
    --enable_sequence_parallelism True \
    --sequence_parallel_shards 8 \
    --optim came \
    --betas 0.9 0.95 0.99 \
    --optim_eps 1e-8 1e-8 \
    --max_device_memory 59GB
```

## Performance

### Training Performance

Experiments are tested on ascend 910* with mindspore 2.4.0 graph mode

| task           | video size | frames  | batch size | flash attention | jit level | step time(s) | train. videos/s |
|:--------------:|:----------:|:-------:|:----------:|:---------------:|:---------:|:------------:|:---------------:|
| Text-To-Video  | 768x1360   |   77    |    1       |       ON        |    O1     |    10.2      |    0.10         |
