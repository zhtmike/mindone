# Efficient CogVideoX based on MindSpore

## Features

- [x] CogVideoX v1.5 2B/5B inference
    - [] Dynamic Condition-free Guidance (CFG)
    - [] Linear-quadratic t-schedule (new)
- [x] CogVideoX v1.5 2B/5B SFT supporting 736P videos of 77 frames
- [x] Memory efficiency optimization (new)
    - [x] VAE cache in training (to be optimized)
    - [x] CAME optimizer in training (verifying)
    - [x] Context parallelism + Zero2 for distributed training based on MindSpore
    - [ ] Simplied VAE tiling
- [] Arbitary aspect-ratio training (new)
- [] LoRA Fine-tuning


TODO: A table comparing to original CogVideoX, including video resolution, frames, memory reqirement...

## Requirements

## Prepare Model Weights

## Inference

## Prapare Dataset for Training

### Video-text Dataset

### VAE Cache

## Training (SFT)


- mindspore 2.3.1

|             task             | image size | frames  | batch size | flash attention | jit level | step time(s) | train. imgs/s |
|:----------------------------:|:----------:|:-------:|:----------:|:---------------:|:---------:|:------------:|:-------------:|
|         MM training          |    512     |   16    |    x      |       ON        |    O0     |    1.320     |    x     |
