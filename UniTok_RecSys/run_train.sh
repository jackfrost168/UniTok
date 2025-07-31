# export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=0
#export CUDA_VISIBLE_DEVICES=1

DATASET=Beauty
OUTPUT_DIR=./ckpt/$DATASET/

torchrun --nproc_per_node=1 --master_port=2320 ./finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 256 \
    --learning_rate 5e-4 \
    --epochs 200 \
    --index_file .index.moe10_2000.json \
    --temperature 1.0



# Clothing_Shoes_and_Jewelry, Health_and_Personal_Care, Sports_and_Outdoors

# Cell_Phones_and_Accessories, Grocery_and_Gourmet_Food, Instruments, Office_Products, Pet_Supplies, Tools_and_Home_Improvement, Toys_and_Games, Video_Games, Yelp