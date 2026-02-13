BUCKET_NAME=""

python raw_s3_model_uploader.py \
    --model_name "meta-llama/Llama-3.1-70B-Instruct" \
    --s3-path $BUCKET_NAME
