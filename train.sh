DATASET_ROOT_PATH=$1
DATASET_NAME=$2
INDEX=$3
TOPK=4

## Train Gaussian Splatting only
## set self.include_feature = False in arguments/__init__.py
# python train.py \
#     -s $DATASET_ROOT_PATH/$DATASET_NAME \
#     -m output/${DATASET_NAME}_${INDEX} \
#     --vq_layer_num 1 \
#     --codebook_size 64 \
#     --cos_loss \
#     --topk $TOPK

for level in 1 2 3
do
    python train.py \
        -s $DATASET_ROOT_PATH/$DATASET_NAME \
        -m output/${DATASET_NAME}_${INDEX} \
        --start_checkpoint $DATASET_ROOT_PATH/$DATASET_NAME/output/chkpnt30000.pth \
        --feature_level ${level} \
        --vq_layer_num 1 \
        --codebook_size 64 \
        --cos_loss \
        --topk $TOPK
done










