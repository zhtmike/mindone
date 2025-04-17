from transformers import AutoProcessor

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

from mindone.transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from mindone.transformers.models.qwen2_vl.qwen_vl_utils import process_vision_info


def int64_to_int32(x: Tensor):
    if x.dtype == ms.int64:
        return x.to(ms.int32)
    return x


def main():
    ms.set_context(mode=ms.GRAPH_MODE)

    with nn.no_init_parameters():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            mindspore_dtype=ms.bfloat16,
            attn_implementation="flash_attention_2",  # paged_attention
        )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="np",
    )
    inputs = {k: int64_to_int32(Tensor(v)) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=False)

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output_text)


if __name__ == "__main__":
    main()
