python scripts/infer_t5.py \
    --csv_path datasets/mixkit-100videos/video_caption_train.csv \
    --output_path datasets/mixkit-100videos/t5_224 \
    --model_max_length 224 \
    --t5_model transformers \
    --dtype fp32 \
    --mode 1 \
    --require_mask False \
    --predict_empty_text_embedding True
