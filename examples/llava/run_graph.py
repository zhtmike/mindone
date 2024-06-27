import time

from model.mistral import MistralForCausalLM

import mindspore as ms
import mindspore.ops as ops


def main():
    ms.set_seed(42)
    ms.set_context(mode=ms.GRAPH_MODE, jit_config=dict(jit_level="O0"))

    net = MistralForCausalLM(num_hidden_layers=1, dtype=ms.float16, attn_implementation="flash_attention")

    START = 2652
    NUM = 10

    # prepare the inputs
    masks = [ops.ones((1, START + i), dtype=ms.int32) for i in range(NUM)]
    position_ids = [ops.arange(0, START + i, dtype=ms.int32)[None, :] for i in range(NUM)]
    inputs_embeds = [ops.rand(1, START + i, 4096).to(ms.float16) for i in range(NUM)]

    print("start...")
    start = None
    for i in range(NUM):
        if i == 1:
            start = time.time()
        net(attention_mask=masks[i], position_ids=position_ids[i], inputs_embeds=inputs_embeds[i])

    end = time.time()
    print("Time elapse: ", end - start)


if __name__ == "__main__":
    main()
