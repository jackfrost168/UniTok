#export CUDA_LAUNCH_BLOCKING=1

DATASET=Beauty
DATA_PATH=../data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/xxx.json
CKPT_PATH=./ckpt/$DATASET/

python test.py \
    --gpu_id 0 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 32 \
    --num_beams 20 \
    --test_prompt_ids 0 \
    --index_file .index.moe10_2000.json
    #--index_file .index.moe10_shared0.05_2000.json


# Beauty, Cell_Phones_and_Accessories, Grocery_and_Gourmet_Food, Instruments, Office_Products, Pet_Supplies, Tools_and_Home_Improvement, Toys_and_Games, Video_Games, Yelp
# Sports_and_Outdoors