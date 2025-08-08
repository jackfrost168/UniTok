python ./MoE_RQ-VAE/generate_indices.py\
    --dataset Beauty \
    --alpha 1e-1 \
    --beta 1e-4 \
    --epoch 10000 \
    --checkpoint best_model_moe10_shared0.05_hsic0.9_2000.pth # best_model-moe10-2000.pth # best_collision_model_moe10_7999.pth       #epoch_3999_collision_0.0085_model.pth      #epoch_9999_collision_0.0012_model.pth 

    # Cell_Phones_and_Accessories, Grocery_and_Gourmet_Food, Instruments, Pet_Supplies, Tools_and_Home_Improvement, Toys_and_Games, Yelp