python scripts/infer_t5.py \
    --csv_path /path/to/video_caption.csv \
    --output_path /path/to/text_embed_folder \
    --model_max_length 224 \
    --t5_model transformers \
    --dtype fp32 \
    --mode 1 \
    --require_mask False \
    --predict_empty_text_embedding True
