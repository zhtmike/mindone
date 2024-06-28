# Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis (PixArt-Alpha)

## Getting Start

### Pretrained Checkpoints

We refer to the [official repository of PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) for pretrained checkpoints downloading.

After downloading the `PixArt-XL-2-{}x{}.pth` file and `PixArt-XL-2-1024-MS.pth`, please place it under the `models/` directory, and then run `tools/convert.py`. For example, to convert `models/PixArt-XL-2-1024-MS.pth.pth`, you can run:

```bash
python tools/convert.py --source models/PixArt-XL-2-1024-MS.pth --target models/PixArt-XL-2-1024-MS.pth
```

In addition, please download the [VAE checkpoint](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema) and [T5 checkpoint](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) and put them under `models` directory, and convert the checkpoints by running:

```bash
python tools/convert_vae.py --source models/sd-vae-ft-ema/diffusion_pytorch_model.bin --target models/sd-vae-ft-ema.ckpt
```

and

```bash
python tools/convert_t5.py --src models/t5-v1_1-xxl/pytorch_model-00001-of-00002.bin  models/t5-v1_1-xxl/pytorch_model-00002-of-00002.bin --target models/t5-v1_1-xxl/model.ckpt
```

And download the T5 checkpoit

After conversion, the checkpoints under `models/` should be like:
```bash
models/
├── PixArt-XL-2-256x256.ckpt
├── PixArt-XL-2-512x512.ckpt
├── PixArt-XL-2-1024-MS.ckpt
├── sd-vae-ft-ema.ckpt
└── t5-v1_1-xxl/model.ckpt
```
