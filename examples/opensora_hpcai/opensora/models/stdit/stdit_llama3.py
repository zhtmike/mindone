import copy
import os
from typing import Literal, Optional

from moviegen import llama3_1B, llama3_5B, llama3_30B
from moviegen.models import TextProjector
from moviegen.models.llama.block import LlamaRMSNorm

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, load_checkpoint, load_param_into_net


class STDiTLlama3Wrapper(nn.Cell):
    def __init__(self, model_size: Literal["1B", "5B", "30B"] = "1B", **kwargs):
        super().__init__(auto_prefix=False)

        attn_implementation = "flash_attention" if kwargs.get("enable_flashattn", False) else "eager"
        gradient_checkpointing = kwargs.get("use_recompute", False)
        model_parallelism = kwargs.get("enable_model_parallelism", False)

        self.model_size = model_size
        self.model_kwargs = dict(
            in_channels=4,
            out_channels=8,
            attn_implementation=attn_implementation,
            gradient_checkpointing=gradient_checkpointing,
            model_parallelism=model_parallelism,
        )

        if self.model_size == "1B":
            self.llama = llama3_1B(**self.model_kwargs)
        elif self.model_size == "5B":
            self.llama = llama3_5B(**self.model_kwargs)
        else:
            self.llama = llama3_30B(**self.model_kwargs)

        self.text_projector = TextProjector(
            out_features=self.llama.hidden_size,
            layer_norm=LlamaRMSNorm,
            norm_eps=self.llama.rms_norm_eps,
            dtype=self.llama.dtype,
        )

        self.patch_size = self.llama.patch_size
        self.hidden_size = self.llama.hidden_size
        self.num_heads = self.llama.num_attention_heads
        self.input_sq_size = None
        self.in_channels = self.llama.in_channels

    def construct(
        self,
        x: Tensor,
        timestep: Tensor,
        y: Tensor,
        mask: Optional[Tensor] = None,
        frames_mask: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        height: Optional[Tensor] = None,
        width: Optional[Tensor] = None,
        extra_text_embed1: Optional[Tensor] = None,
        extra_mask1: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        x = ops.transpose(x, (0, 2, 1, 3, 4))

        if extra_text_embed1 is not None:
            y = ops.squeeze(y, axis=1)
            # FIXME: placeholder for MetaCLIP
            metaclip_text_embed = ops.ones((extra_text_embed1.shape[0], 100, 1280), dtype=extra_text_embed1.dtype)
            text_embedding = self.text_projector(y, metaclip_text_embed, extra_text_embed1)
        else:
            text_embedding = ops.squeeze(y, axis=1)

        latent_embedding = x
        output = self.llama(latent_embedding, timestep, text_embedding)
        output = ops.transpose(output, (0, 2, 1, 3, 4))
        return output

    def load_from_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found. No checkpoint loaded!!")
        else:
            sd = load_checkpoint(ckpt_path)
            sd = {k.replace("network.llama.", "").replace("_backbone.", ""): v for k, v in sd.items()}

            m, u = load_param_into_net(self, sd, strict_load=True)
            print("net param not load: ", m, len(m))
            print("ckpt param not load: ", u, len(u))

    def load_from_single_checkpoint(self, ckpt_path):
        # temporarily solution to load checkpoint to model parallel
        model_parallelism = self.model_kwargs.get("model_parallelism", False)
        if not model_parallelism:
            return self.load_from_checkpoint(ckpt_path)

        model_kwargs = copy.deepcopy(self.model_kwargs)
        model_kwargs["model_parallelism"] = False

        if self.model_size == "1B":
            llama = llama3_1B(**model_kwargs)
        elif self.model_size == "5B":
            llama = llama3_5B(**model_kwargs)
        else:
            llama = llama3_30B(**model_kwargs)

        sd = load_checkpoint(ckpt_path)
        sd = {k.replace("network.llama.", "").replace("_backbone.", ""): v for k, v in sd.items()}
        m, u = load_param_into_net(llama, sd, strict_load=True)
        print("net param not load: ", m, len(m))
        print("ckpt param not load: ", u, len(u))

        param_dict = llama.parameters_dict()

        # filter tensor-parallel block
        names = ["gate_proj", "up_proj", "down_proj"]
        param_dict = {k: v for k, v in param_dict.items() if not any([name in k for name in names])}
        load_param_into_net(self, param_dict)

        # load tensor-parallel block
        for layer, target_layer in zip(self.llama.layers, llama.layers):
            layer.mlp.load_weight_from_non_parallel_cell(target_layer.mlp)
