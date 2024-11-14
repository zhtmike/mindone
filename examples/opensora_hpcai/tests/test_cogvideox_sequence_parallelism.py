import argparse

import numpy as np
from opensora.acceleration.parallel_states import create_parallel_group, get_sequence_parallel_group
from opensora.models.cogvideox import CogVideoXTransformer3DModel

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.communication import get_group_size, init

from mindone.utils.seed import set_random_seed


class MeanNet(nn.Cell):
    def __init__(self, net: nn.Cell) -> None:
        super().__init__()
        self.net = net

    def construct(self, *inputs):
        output = self.net(*inputs)
        return output.mean()


def get_sample_data(use_rotary_positional_embeddings=False):
    x = ops.rand([2, 16, 13, 60, 90], dtype=ms.float16)  # (B, C, T, H, W)
    timestep = ms.Tensor([2, 32], dtype=ms.int64)
    y = ops.rand(2, 226, 4096, dtype=ms.float16)
    if use_rotary_positional_embeddings:
        image_rotary_emb = ops.rand(2, 2, 17550, 64, dtype=ms.float32)
    else:
        image_rotary_emb = None
    return dict(x=x, timestep=timestep, y=y, image_rotary_emb=image_rotary_emb)


def get_model_config(use_rotary_positional_embeddings=False, enable_sequence_parallelism=False):
    config = {
        "num_layers": 1,
        "use_rotary_positional_embeddings": use_rotary_positional_embeddings,
        "enable_sequence_parallelism": enable_sequence_parallelism,
        "enable_flash_attention": True,
        "dtype": ms.float16,
    }
    return config


def run_model(mode: int = 0, use_rotary_positional_embeddings=True):
    ms.set_context(mode=mode)
    init()

    # prepare data
    set_random_seed(1024)
    data = get_sample_data(use_rotary_positional_embeddings=use_rotary_positional_embeddings)

    # single model
    set_random_seed(1024)
    non_dist_model_cfg = get_model_config(
        use_rotary_positional_embeddings=use_rotary_positional_embeddings, enable_sequence_parallelism=False
    )
    non_dist_model = CogVideoXTransformer3DModel(**non_dist_model_cfg)

    # sequence parallel model
    create_parallel_group(get_group_size())
    set_random_seed(1024)
    dist_model_cfg = get_model_config(
        use_rotary_positional_embeddings=use_rotary_positional_embeddings, enable_sequence_parallelism=True
    )
    dist_model = CogVideoXTransformer3DModel(**dist_model_cfg)

    for (_, w0), (_, w1) in zip(non_dist_model.parameters_and_names(), dist_model.parameters_and_names()):
        w1.set_data(w0)  # FIXME: seed does not work
        np.testing.assert_allclose(w0.value().asnumpy(), w1.value().asnumpy())

    # test forward
    non_dist_out = non_dist_model(**data)
    dist_out = dist_model(**data)

    np.testing.assert_allclose(non_dist_out.asnumpy(), dist_out.asnumpy(), atol=1e-2)
    print("Test 1 (Forward): Passed.")

    # test backward
    non_dist_mean_net = MeanNet(non_dist_model)
    dist_mean_net = MeanNet(dist_model)

    grad_fn = ops.value_and_grad(non_dist_mean_net, grad_position=None, weights=non_dist_mean_net.trainable_params())
    non_dist_loss, non_dist_grads = grad_fn(*data.values())

    grad_fn = ops.value_and_grad(dist_mean_net, grad_position=None, weights=dist_mean_net.trainable_params())
    dist_loss, dist_grads = grad_fn(*data.values())

    # take mean around different ranks
    sp_group = get_sequence_parallel_group()
    reduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=sp_group)
    num = get_group_size()
    syn_dist_grads = list()
    for x in dist_grads:
        syn_dist_grads.append(reduce(x) / num)

    np.testing.assert_allclose(non_dist_loss.asnumpy(), dist_loss.asnumpy(), atol=1e-2)

    for grad_0, grad_1 in zip(non_dist_grads, syn_dist_grads):
        np.testing.assert_allclose(grad_0.asnumpy(), grad_1.asnumpy(), atol=1e-2)
    print("Test 2 (Backward): Passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default=1, type=int, choices=[0, 1], help="Mode to test. (0: Graph Mode; 1: Pynative mode)"
    )
    args = parser.parse_args()
    run_model(mode=args.mode)
