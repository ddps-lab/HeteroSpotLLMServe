BUCKET_NAME=""
MODEL_NAME=""

python raw_s3_model_uploader.py \
    --model_name $MODEL_NAME \
    --s3-path $BUCKET_NAME
