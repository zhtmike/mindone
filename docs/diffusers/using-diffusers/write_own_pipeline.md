<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Understanding pipelines, models and schedulers

🧨 Diffusers is designed to be a user-friendly and flexible toolbox for building diffusion systems tailored to your use-case. At the core of the toolbox are models and schedulers. While the [`DiffusionPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline) bundles these components together for convenience, you can also unbundle the pipeline and use the models and schedulers separately to create new diffusion systems.

In this tutorial, you'll learn how to use models and schedulers to assemble a diffusion system for inference, starting with a basic pipeline and then progressing to the Stable Diffusion pipeline.

## Deconstruct a basic pipeline

A pipeline is a quick and easy way to run a model for inference, requiring no more than four lines of code to generate an image:

```python
from mindone.diffusers import DDPMPipeline

ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True)
image = ddpm(num_inference_steps=1000)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/2d737a78-e806-4e41-86b3-ca136a905cac" alt="Image of cat created from DDPMPipeline"/>
</div>

That was super easy, but how did the pipeline do that? Let's breakdown the pipeline and take a look at what's happening under the hood.

In the example above, the pipeline contains a [`UNet2DModel`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/unet2d/#mindone.diffusers.UNet2DModel) model and a [`DDPMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/ddpm/#mindone.diffusers.DDPMScheduler). The pipeline denoises an image by taking random noise the size of the desired output and passing it through the model several times. At each timestep, the model predicts the *noise residual* and the scheduler uses it to predict a less noisy image. The pipeline repeats this process until it reaches the end of the specified number of inference steps.

To recreate the pipeline with the model and scheduler separately, let's write our own denoising process.

1. Load the model and scheduler:

```python
from mindone.diffusers import DDPMScheduler, UNet2DModel

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True)
```

2. Set the number of timesteps to run the denoising process for:

```python
scheduler.set_timesteps(50)
```

3. Setting the scheduler timesteps creates a tensor with evenly spaced elements in it, 50 in this example. Each element corresponds to a timestep at which the model denoises an image. When you create the denoising loop later, you'll iterate over this tensor to denoise an image:

```python
scheduler.timesteps
Tensor(shape=[50], dtype=Int64, value=[980, 960, 940, 920, 900, 880, 860,
  840, 820, 800, 780, 760, 740, 720, 700, 680, 660, 640, 620, 600, 580,
  560, 540, 520, 500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300,
  280, 260, 240, 220, 200, 180, 160, 140, 120, 100,  80,  60,  40,  20,
  0])
```

4. Create some random noise with the same shape as the desired output:

```python
import mindspore

sample_size = model.config.sample_size
noise = mindspore.ops.randn((1, 3, sample_size, sample_size))
```

5. Now write a loop to iterate over the timesteps. At each timestep, the model does a [`UNet2DModel.construct`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/unet2d/#mindone.diffusers.UNet2DModel.construct) pass and returns the noisy residual. The scheduler's [`step`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/ddpm/#mindone.diffusers.DDPMScheduler.step) method takes the noisy residual, timestep, and input and it predicts the image at the previous timestep. This output becomes the next input to the model in the denoising loop, and it'll repeat until it reaches the end of the `timesteps` array.

```python
input = noise

for t in scheduler.timesteps:
noisy_residual = model(input, t)[0]
previous_noisy_sample = scheduler.step(noisy_residual, t, input)[0]
input = previous_noisy_sample
```

    This is the entire denoising process, and you can use this same pattern to write any diffusion system.

6. The last step is to convert the denoised output into an image:

```python
from PIL import Image
import numpy as np

image = (input / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).round().to(mindspore.uint8).numpy()
image = Image.fromarray(image)
image
```

In the next section, you'll put your skills to the test and breakdown the more complex Stable Diffusion pipeline. The steps are more or less the same. You'll initialize the necessary components, and set the number of timesteps to create a `timestep` array. The `timestep` array is used in the denoising loop, and for each element in this array, the model predicts a less noisy image. The denoising loop iterates over the `timestep`'s, and at each timestep, it outputs a noisy residual and the scheduler uses it to predict a less noisy image at the previous timestep. This process is repeated until you reach the end of the `timestep` array.

Let's try it out!

## Deconstruct the Stable Diffusion pipeline

Stable Diffusion is a text-to-image *latent diffusion* model. It is called a latent diffusion model because it works with a lower-dimensional representation of the image instead of the actual pixel space, which makes it more memory efficient. The encoder compresses the image into a smaller representation, and a decoder converts the compressed representation back into an image. For text-to-image models, you'll need a tokenizer and an encoder to generate text embeddings. From the previous example, you already know you need a UNet model and a scheduler.

As you can see, this is already more complex than the DDPM pipeline which only contains a UNet model. The Stable Diffusion model has three separate pretrained models.

!!! tip

    💡 Read the [How does Stable Diffusion work?](https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work) blog for more details about how the VAE, UNet, and text encoder models work.

Now that you know what you need for the Stable Diffusion pipeline, load all these components with the [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/overview/#mindone.diffusers.ModelMixin.from_pretrained) method. You can find them in the pretrained [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) checkpoint, and each component is stored in a separate subfolder:

```python
from PIL import Image
import mindspore
from transformers import CLIPTokenizer
from mindone.transformers import CLIPTextModel
from mindone.diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
   "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
)
unet = UNet2DConditionModel.from_pretrained(
   "CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True
)
```

Instead of the default [`PNDMScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/pndm/#mindone.diffusers.PNDMScheduler), exchange it for the [`UniPCMultistepScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/unipc/#mindone.diffusers.UniPCMultistepScheduler) to see how easy it is to plug a different scheduler in:

```python
from mindone.diffusers import UniPCMultistepScheduler

scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
```

### Create text embeddings

The next step is to tokenize the text to generate embeddings. The text is used to condition the UNet model and steer the diffusion process towards something that resembles the input prompt.

!!! tip

    💡 The `guidance_scale` parameter determines how much weight should be given to the prompt when generating an image.

Feel free to choose any prompt you like if you want to generate something else!

```python
prompt = ["a photograph of an astronaut riding a horse"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = np.random.Generator(np.random.PCG64(seed=0))  # Seed generator to create the initial latent noise
batch_size = len(prompt)
```

Tokenize the text and generate the embeddings from the prompt:

```python
text_input = tokenizer(
   prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="np"
)

text_embeddings = text_encoder(mindspore.Tensor(text_input.input_ids))[0]
```

You'll also need to generate the *unconditional text embeddings* which are the embeddings for the padding token. These need to have the same shape (`batch_size` and `seq_length`) as the conditional `text_embeddings`:

```python
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np")
uncond_embeddings = text_encoder(mindspore.Tensor(uncond_input.input_ids))[0]
```

Let's concatenate the conditional and unconditional embeddings into a batch to avoid doing two forward passes:

```python
text_embeddings = mindspore.ops.cat([uncond_embeddings, text_embeddings])
```

### Create random noise

Next, generate some initial random noise as a starting point for the diffusion process. This is the latent representation of the image, and it'll be gradually denoised. At this point, the `latent` image is smaller than the final image size but that's okay though because the model will transform it into the final 512x512 image dimensions later.

!!! tip

    💡 The height and width are divided by 8 because the `vae` model has 3 down-sampling layers. You can check by running the following:

    ```python
    print(2 ** (len(vae.config.block_out_channels) - 1) == 8)
    ```

```python
latents = mindspore.ops.randn(
   (batch_size, unet.config.in_channels, height // 8, width // 8),
)
```

### Denoise the image

Start by scaling the input with the initial noise distribution, *sigma*, the noise scale value, which is required for improved schedulers like [`UniPCMultistepScheduler`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/unipc/#mindone.diffusers.UniPCMultistepScheduler):

```python
latents = latents * scheduler.init_noise_sigma
```

The last step is to create the denoising loop that'll progressively transform the pure noise in `latents` to an image described by your prompt. Remember, the denoising loop needs to do three things:

1. Set the scheduler's timesteps to use during denoising.
2. Iterate over the timesteps.
3. At each timestep, call the UNet model to predict the noise residual and pass it to the scheduler to compute the previous noisy sample.

```python
from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = mindspore.ops.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)[0]

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents)[0]
```

### Decode the image

The final step is to use the `vae` to decode the latent representation into an image and get the decoded output with `sample`:

```python
# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
image = vae.decode(latents)[0]
```

Lastly, convert the image to a `PIL.Image` to see your generated image!

```python
image = (image / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).to(mindspore.uint8).numpy()
image = Image.fromarray(image)
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img src="https://github.com/user-attachments/assets/627e641b-4c6c-4d78-9345-f7b686915eb6"/>
</div>

## Next steps

From basic to complex pipelines, you've seen that all you really need to write your own diffusion system is a denoising loop. The loop should set the scheduler's timesteps, iterate over them, and alternate between calling the UNet model to predict the noise residual and passing it to the scheduler to compute the previous noisy sample.

This is really what 🧨 Diffusers is designed for: to make it intuitive and easy to write your own diffusion system using models and schedulers.

For your next steps, feel free to:

* Learn how to [build and contribute a pipeline](../using-diffusers/contribute_pipeline.md) to 🧨 Diffusers. We can't wait and see what you'll come up with!
* Explore [existing pipelines](../api/pipelines/overview.md) in the library, and see if you can deconstruct and build a pipeline from scratch using the models and schedulers separately.
