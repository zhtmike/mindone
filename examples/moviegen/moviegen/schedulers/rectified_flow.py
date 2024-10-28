import logging
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

logger = logging.getLogger(__name__)

__all__ = ["RFLOW", "RFlowScheduler"]


def mean_flat(tensor: Tensor) -> Tensor:
    return mint.mean(tensor, dim=list(range(1, len(tensor.shape))))


class LogisticNormal(nn.Cell):
    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None:
        self.mean = loc
        self.std = scale
        self._min = Tensor(np.finfo(np.float32).tiny, dtype=ms.float32)
        self._max = Tensor(1.0 - np.finfo(np.float32).eps, dtype=ms.float32)

    def construct(self, shape: Tuple[int, ...]) -> Tensor:
        x = self.mean + self.std * mint.normal(size=shape)
        z = mint.clamp(mint.sigmoid(x), min=self._min, max=self._max)
        return mint.pad(z, [0, 1], value=1) * mint.pad(1 - z, [1, 0], value=1)


class RFLOW:
    def __init__(
        self,
        num_sampling_steps: int = 10,
        num_timesteps: int = 1000,
        **kwargs: Any,
    ) -> None:
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps

        self.scheduler = RFlowScheduler(
            num_timesteps=self.num_timesteps, num_sampling_steps=self.num_sampling_steps, **kwargs
        )

    def __call__(self, model: nn.Cell, z: Tensor, model_kwargs: Dict[str, Tensor]) -> Tensor:
        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        timesteps = [Tensor([t] * z.shape[0]) for t in timesteps]

        for i, t in tqdm(enumerate(timesteps), total=self.num_sampling_steps):
            pred = model(z, t, **model_kwargs)
            # FIXME: a tmp solution for inference with cfg==1.0
            pred = pred[:, :4]

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + pred * dt[:, None, None, None, None]

        return z


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps: int = 1000,
        sample_method: Literal["discrete-uniform", "uniform", "logit-normal"] = "uniform",
        loc: float = 0.0,
        scale: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.eps = eps

        if sample_method == "discrete-uniform":
            self._sample_func = self._discrete_sample
        elif sample_method == "uniform":
            self._sample_func = self._uniform_sample
        elif sample_method == "logit-normal":
            self.distribution = LogisticNormal(loc=loc, scale=scale)
            self._sample_func = self._logit_normal_sample
        else:
            raise ValueError(f"Unknown sample method: {sample_method}")

        self.criteria = nn.MSELoss()

    def _discrete_sample(self, size: int) -> Tensor:
        return ops.randint(0, self.num_timesteps, (size,), dtype=ms.int64)

    def _uniform_sample(self, size: int) -> Tensor:
        return mint.rand((size,), dtype=ms.float32) * self.num_timesteps

    def _logit_normal_sample(self, size: int) -> Tensor:
        return self.distribution((size,)) * self.num_timesteps

    def training_losses(
        self, model: nn.Cell, x: Tensor, text_embedding: Tensor, timestep: Optional[Tensor] = None
    ) -> Tensor:
        """
        x: (N, T, C, H, W) tensor of inputs (latent representations of video)
        text_embedding: (N, L, C') tensor of the text embedding
        timestep: (N,) tensor to indicate denoising step
        """
        if timestep is None:
            timestep = self._sample_func(x.shape[0])

        # TODO: change to mint.randn_like
        noise = mint.normal(size=x.shape).to(x.dtype)
        x_t = self.add_noise(x, noise, timestep)

        # force to be fp32
        model_output = model(x_t, timestep, text_embedding).to(ms.float32)
        v_t = (x - (1 - self.eps) * noise).to(ms.float32)

        # 3.1.2 Eqa (2)
        loss = self.criteria(model_output, v_t)
        return loss

    def add_noise(self, x: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """
        x: (N, T, C, H, W) tensor of ground truth
        noise: (N, T, C, H, W) tensor of white noise
        timesteps: (N,) tensor of timestamps with range [0, num_timesteps)
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = 1 - timesteps  # in range [1, 1/1000]

        timesteps = timesteps[:, None, None, None, None]

        # 3.1.2 First Eqa.
        return timesteps * x + (1 - (1 - self.eps) * timesteps) * noise
