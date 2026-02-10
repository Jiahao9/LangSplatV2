DATASET_ROOT_PATH=$1
DATASET_NAME=$2
INDEX=$3
TOPK=4

echo "Train Gaussian Splatting"

# Train Gaussian Splatting only
python train.py \
    -s $DATASET_ROOT_PATH/$DATASET_NAME \
    -m $DATASET_ROOT_PATH/$DATASET_NAME/gaussian_splatting \
    --vq_layer_num 1 \
    --codebook_size 64 \
    --cos_loss \
    --topk $TOPK  \

echo "Train LangSplatV2"

for level in 1 2 3
do
    python train.py \
        -s $DATASET_ROOT_PATH/$DATASET_NAME \
        -m output/${DATASET_NAME}_${INDEX} \
        --start_checkpoint $DATASET_ROOT_PATH/$DATASET_NAME/gaussian_splatting_-1/chkpnt30000.pth \
        --feature_level ${level} \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --cos_loss \
        --topk $TOPK  \
        --include_feature_in_training
done